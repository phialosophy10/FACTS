"""Train a 2D SRGAN or ESRGAN model on FACTS with an L1 pixel loss + adversarial loss.

Training has two phases:
  1. warm-up (--warmup_iters): generator only, L1 loss (PSNR-oriented pre-training)
  2. GAN training: L1 loss + adv_weight * adversarial loss
     SRGAN uses the standard GAN loss, ESRGAN the relativistic average GAN loss.

Training is resumed automatically from <run_dir>/checkpoint.pt if it exists.

Example:
    python train.py --root /path/to/FACTS --model ESRGAN --data real
"""
import argparse
import csv
import json
import os
import random
import time
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from facts import config
from facts.data import FemurVolume, Normalizer, PatchDataset, fixed_patches
from facts.models import Discriminator, build_generator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The discriminator scheduler is stepped (to stay aligned with the generator) during the
# warm-up phase, where the discriminator is not updated
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root directory")
    p.add_argument("--model", choices=["SRGAN", "ESRGAN"], required=True)
    p.add_argument("--data", choices=["real", "synth"], required=True, help="Input data used for training")
    p.add_argument("--runs_dir", default="runs", help="Output directory (run dir: <runs_dir>/<model>_<data>)")
    p.add_argument("--name", default=None, help="Run name (default: <model>_<data>)")
    p.add_argument("--train_femurs", nargs="*", default=config.TRAIN_FEMURS)
    p.add_argument("--axis", type=int, default=config.SLICE_AXIS, help="Slicing axis for 2D slices")
    p.add_argument("--iters", type=int, default=100_000, help="Total number of iterations (incl. warm-up)")
    p.add_argument("--warmup_iters", type=int, default=10_000, help="L1-only iterations before GAN training")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pix_weight", type=float, default=1.0, help="Weight of the L1 pixel loss")
    p.add_argument("--adv_weight", type=float, default=1e-3, help="Weight of the adversarial loss")
    p.add_argument("--n_blocks", type=int, default=None,
                   help="Residual blocks in the generator (default: SRGAN 16, ESRGAN 8)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--mmap", action="store_true",
                   help="Memory-map volumes instead of loading them into RAM (slow unless --axis 0)")
    p.add_argument("--val_every", type=int, default=2_000)
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    return p.parse_args()


# ----------------------------------------------------------------- losses ---
def d_loss_fn(model, real, fake):
    if model == "ESRGAN":  # relativistic average GAN
        return 0.5 * (F.binary_cross_entropy_with_logits(real - fake.mean(), torch.ones_like(real))
                      + F.binary_cross_entropy_with_logits(fake - real.mean(), torch.zeros_like(fake)))
    return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
            + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))


def g_adv_loss_fn(model, real, fake):
    if model == "ESRGAN":
        return 0.5 * (F.binary_cross_entropy_with_logits(real - fake.mean(), torch.zeros_like(real))
                      + F.binary_cross_entropy_with_logits(fake - real.mean(), torch.ones_like(fake)))
    return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))


def psnr(sr, hr):
    mse = F.mse_loss(sr.clamp(0, 1), hr.clamp(0, 1), reduction="none").mean(dim=(1, 2, 3))
    return (10 * torch.log10(1 / mse.clamp_min(1e-10))).mean().item()


# ---------------------------------------------------------------- helpers ---
def save_samples(path, batch, sr, n=4):
    n = min(n, sr.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        for j, (im, title) in enumerate([(batch["lr"][i, 0], "input"), (sr[i, 0], "SR"), (batch["hr"][i, 0], "HR")]):
            axes[i, j].imshow(im.float().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def infinite(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    name = args.name or f"{args.model.lower()}_{args.data}"
    run_dir = os.path.join(args.runs_dir, name)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Run: {run_dir} | device: {device} | AMP: {use_amp} ({amp_dtype})")

    # ---- data
    t0 = time.time()
    femurs = [FemurVolume(args.root, f, args.data, axis=args.axis, in_memory=not args.mmap, keep_mask=False)
              for f in args.train_femurs]
    print(f"Loaded {len(femurs)} femurs in {time.time() - t0:.0f} s")
    norm = Normalizer.from_femurs(femurs)
    print(f"Intensity normalisation: [{norm.lo:.1f}, {norm.hi:.1f}] -> [0, 1]")
    dataset = PatchDataset(femurs, norm, args.patch_size, length=args.iters * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        drop_last=True, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_batch = fixed_patches(femurs, norm, n=32, patch_size=args.patch_size, seed=args.seed)

    # ---- models
    G = build_generator(args.model, args.n_blocks).to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))
    milestones = [int(args.iters * 0.5), int(args.iters * 0.75)]
    sched_G = torch.optim.lr_scheduler.MultiStepLR(opt_G, milestones, 0.5)
    sched_D = torch.optim.lr_scheduler.MultiStepLR(opt_D, milestones, 0.5)
    scaler_G = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    scaler_D = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    start = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        for obj, key in [(G, "G"), (D, "D"), (opt_G, "opt_G"), (opt_D, "opt_D"), (sched_G, "sched_G"),
                         (sched_D, "sched_D"), (scaler_G, "scaler_G"), (scaler_D, "scaler_D")]:
            obj.load_state_dict(ckpt[key])
        start = ckpt["iter"]
        norm = Normalizer(**ckpt["normalizer"])
        print(f"Resumed from iteration {start}")
        dataset.norm = norm

    meta = {"model": args.model, "train_data": args.data, "n_blocks": args.n_blocks, "axis": args.axis,
            "normalizer": norm.state_dict(), "args": vars(args)}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def save_checkpoint(it):
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(), "sched_G": sched_G.state_dict(), "sched_D": sched_D.state_dict(),
                    "scaler_G": scaler_G.state_dict(), "scaler_D": scaler_D.state_dict(),
                    "iter": it, "normalizer": norm.state_dict()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    def save_generator(fname):
        torch.save({"G": G.state_dict(), **meta}, os.path.join(run_dir, fname))

    if args.wandb:
        import wandb
        wandb.init(project="FACTS", name=name, config=meta, resume="allow", id=name)

    log_path = os.path.join(run_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["iter", "lr", "loss_pix", "loss_adv", "loss_D", "val_l1", "val_psnr", "input_psnr", "sec_per_iter"])

    # ---- training loop
    G.train()
    D.train()
    data_iter = infinite(loader)
    running = {"pix": 0.0, "adv": 0.0, "D": 0.0, "n": 0}
    t_last = time.time()
    for it in range(start + 1, args.iters + 1):
        batch = next(data_iter)
        lr_img = batch["lr"].to(device, non_blocking=True)
        hr_img = batch["hr"].to(device, non_blocking=True)
        gan_phase = it > args.warmup_iters

        # Generator update
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            sr_img = G(lr_img)
            loss_pix = F.l1_loss(sr_img.float(), hr_img)
            loss_G = args.pix_weight * loss_pix
            if gan_phase:
                for p in D.parameters():
                    p.requires_grad_(False)
                pred_fake = D(sr_img).float()
                pred_real = D(hr_img).float().detach()
                loss_adv = g_adv_loss_fn(args.model, pred_real, pred_fake)
                loss_G = loss_G + args.adv_weight * loss_adv
        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()
        sched_G.step()

        # Discriminator update
        if gan_phase:
            for p in D.parameters():
                p.requires_grad_(True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                pred_real = D(hr_img).float()
                pred_fake = D(sr_img.detach()).float()
                loss_D = d_loss_fn(args.model, pred_real, pred_fake)
            opt_D.zero_grad(set_to_none=True)
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
            running["adv"] += loss_adv.item()
            running["D"] += loss_D.item()
        sched_D.step()
        running["pix"] += loss_pix.item()
        running["n"] += 1

        if it == args.warmup_iters:
            save_generator("generator_warmup.pt")

        # Monitoring
        if it % args.val_every == 0 or it == args.iters:
            G.eval()
            with torch.inference_mode():
                sr_val = torch.cat([G(b.to(device)).float() for b in val_batch["lr"].split(8)]).cpu()
            G.train()
            n = running["n"]
            row = [it, sched_G.get_last_lr()[0], running["pix"] / n, running["adv"] / n, running["D"] / n,
                   F.l1_loss(sr_val, val_batch["hr"]).item(), psnr(sr_val, val_batch["hr"]),
                   psnr(val_batch["lr"], val_batch["hr"]), (time.time() - t_last) / n]
            log.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
            log_file.flush()
            print(f"[{it}/{args.iters}] pix {row[2]:.4f} adv {row[3]:.4f} D {row[4]:.4f} | "
                  f"val L1 {row[5]:.4f} PSNR {row[6]:.2f} dB (input {row[7]:.2f} dB) | {row[8]:.3f} s/it")
            if args.wandb:
                wandb.log(dict(zip(["iter", "lr", "loss_pix", "loss_adv", "loss_D", "val_l1", "val_psnr",
                                    "input_psnr", "sec_per_iter"], row)), step=it)
            save_samples(os.path.join(run_dir, "samples", f"iter_{it:07d}.png"), val_batch, sr_val)
            running = {"pix": 0.0, "adv": 0.0, "D": 0.0, "n": 0}
            t_last = time.time()

        if it % args.save_every == 0 or it == args.iters:
            save_checkpoint(it)

    save_generator("generator.pt")
    log_file.close()
    print(f"Finished. Generator saved to {os.path.join(run_dir, 'generator.pt')}")


if __name__ == "__main__":
    main()
