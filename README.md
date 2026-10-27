# FACTS: Femur Archaeological CT Superresolution

FACTS is a benchmark for real-world 3D super-resolution. It contains 13 human proximal femurs, each scanned with both clinical CT (LR) and micro-CT (HR). The two scans of each bone are registered voxel to voxel. This repository contains the preprocessing, the 2D SRGAN/ESRGAN benchmark, and the [project page](https://phialosophy10.github.io/FACTS/).

* Data: <https://archive.compute.dtu.dk/files/public/projects/FACTS>
* Paper: [Superresolution of real-world multiscale bone CT verified with clinical bone measures](https://doi.org/10.1007/978-3-031-66958-3_12) (MIUA 2024)

## Repository layout

```
facts/                    library code
  config.py               femur IDs, train/test split, voxel size, synthetic parameters
  io.py                   paths and (memory-mapped) TIFF I/O
  histmatch.py            masked histogram matching and linear calibration
  qa.py                   preprocessing snapshot figures
  synthetic.py            Gaussian blur + 4x down/up-sampling
  data.py                 patch sampling inside the bone mask, intensity normalisation
  models.py               SRGAN / ESRGAN generators and discriminator (2D)
  metrics.py              PSNR, SSIM, LPIPS, BV/TV, Tb.Th
  inference.py            loading trained generators, tiled (batched) inference
  uncertainty.py          noise levels, blended noisy inputs, sparsification/AUSE
scripts/
  histogram_match.py      HR -> LR intensity scale     => f_NNN/HR/f_NNN_hm.tif
  make_synthetic.py       synthetic LR from HR_hm      => f_NNN/SY/f_NNN_hm.tif
  check_data.py           shapes, dtypes and intensity statistics per femur
  plot_histograms.py      intensity histograms of all volumes
  inspect_preprocessing.py  snapshot figures of the preprocessing
  make_results.py         metrics + figures -> project page (docs/)
train.py                  training (L1 + adversarial loss)
test.py                   inference + evaluation (incl. cross-testing and baseline)
uncertainty.py            input-noise uncertainty along a synthetic -> real input blend
jobs/                     LSF job scripts for DTU HPC
docs/                     GitHub Pages project page (Jekyll)
init.sh                   creates/activates the Python environment on DTU HPC
```

## Data

```
<root>/f_NNN/HR/f_NNN.tif       micro-CT
<root>/f_NNN/LR/f_NNN.tif       clinical CT, resliced to the micro-CT grid
<root>/f_NNN/MS/f_NNN.tif       bone mask
<root>/f_NNN/SY/f_NNN.tif       synthetic LR (from the original micro-CT)
created here:
<root>/f_NNN/HR/f_NNN_hm.tif    micro-CT, histogram matched to the clinical CT (uint16)
<root>/f_NNN/SY/f_NNN_hm.tif    synthetic LR made from HR_hm (uint16)
```

## Setup on DTU HPC

```bash
git clone https://github.com/phialosophy10/FACTS.git && cd FACTS
cp local.sh.example local.sh && nano local.sh   # set DATA_ROOT (path to the folder with f_001/, f_002/, ...)
source init.sh            # first call creates the venv, later calls only activate it
```

`local.sh` holds your personal settings for this project (`DATA_ROOT`, optionally `WANDB_ENTITY`). It is loaded by `init.sh`, so also by every job script, and it is not tracked by git. A `DATA_ROOT=...` given on the command line takes precedence over it.

`init.sh` loads a Python module, creates a venv in `~/venvs/facts`, installs PyTorch (CUDA wheel) and `requirements.txt`, and downloads the LPIPS weights. You can override the defaults:

```bash
PY_MODULE=python3/3.12.4 TORCH_CUDA=cu126 VENV_DIR=/work3/$USER/venvs/facts source init.sh
```

**Choosing versions:**
* Run `module avail python3` on a login node to see the available Python versions. Python 3.11 or 3.12 is a safe choice.
* PyTorch pip wheels ship their own CUDA runtime, so only the GPU driver matters. Open an interactive GPU node (e.g. `a100sh` or `voltash`) and run `nvidia-smi`. The "CUDA Version" in the header is the newest CUDA the driver supports, so pick a wheel at or below it: `cu126` for CUDA ≥ 12.6, `cu124` for ≥ 12.4, `cu121` for ≥ 12.1, `cu118` for 11.8.
* Check the install with `python -c "import torch; print(torch.cuda.is_available())"` on a GPU node.

## Running

Run all steps from the repository root, after `source init.sh` (which sets `$DATA_ROOT` from `local.sh`).

**1. Preprocessing** (CPU):

```bash
python scripts/plot_histograms.py --root $DATA_ROOT                               # look at all volumes first
python scripts/histogram_match.py --root $DATA_ROOT --method linear --offset 2048  # HR -> HR/f_NNN_hm.tif
python scripts/make_synthetic.py  --root $DATA_ROOT                               # SY/f_NNN_hm.tif
```

`histogram_match.py` brings the micro-CT to the intensity scale of the clinical CT, in one of two ways (`--method`):
* **`histogram`** (default): matches the micro-CT histogram inside the mask to the clinical-CT histogram. The micro-CT has sharp, separated peaks (marrow/air and bone) while the clinical CT is blurred, so this mapping is strongly non-linear: it stretches some intensity ranges, together with their noise, and compresses others.
* **`linear`**: `HR_hm = a · HR + b`. The values of `a` and `b` are fitted so that `a · blur(HR) + b` matches the clinical CT inside the mask (least squares on random blocks; the blur width is chosen by the best R²). It keeps the micro-CT's contrast and noise proportions. Because thin trabeculae are blurred in the clinical CT, HR_hm bone values end up higher than the clinical CT's, which is physically expected. The fit (`a`, `b`, blur width, R² for each width) is printed and saved in `HR/f_NNN_hm.json`.

If the clinical CT is in HU (negative values), uint16 cannot hold it, so add an `--offset` and use **the same offset for all femurs**. With `--method linear`, micro-CT noise in air/marrow maps below −1024 HU. The script reports how much of the mask is clipped and suggests a larger offset if needed (or use `--dtype float32`).

Both preprocessing scripts write snapshot figures to `results/preprocessing/snapshots/`. To produce them for already processed femurs, run `scripts/inspect_preprocessing.py`. The figures show one slice at every stage, the intensity mapping and its gain, a registration overlay and intensity profiles. `plot_histograms.py` writes histograms of all volumes and a CSV of statistics to `results/histograms/` (`--mods LR HR HR_hm SY_hm`, `--whole`, `--log`).

**2. Training** (4 runs: {SRGAN, ESRGAN} × {real, synth}):

```bash
python train.py --root $DATA_ROOT --model ESRGAN --data real     # -> runs/esrgan_real/
```

The defaults are 10k L1-only warm-up iterations, then 100k total iterations with L1 + 1e-3 · adversarial loss (no perceptual loss), batch 16, 128×128 patches, Adam lr 1e-4. Slices are taken along axis 2 (`--axis`). Training resumes automatically from `runs/<name>/checkpoint.pt`. The run directory also holds `log.csv` and sample images; `--wandb` also logs to Weights & Biases (see below).

By default the training volumes are loaded into RAM, cropped to the bone bounding box. This needs roughly 100–150 GB for the 10 training femurs. `--mmap` memory-maps the volumes instead, which is only fast with `--axis 0`.

**3. Testing and cross-testing:**

```bash
python test.py --root $DATA_ROOT --run runs/esrgan_real --input real
python test.py --root $DATA_ROOT --run runs/esrgan_real --input synth
python test.py --root $DATA_ROOT --baseline --input real          # LR baseline (no SR)
```

`test.py` writes to `runs/<name>/test_<input>/` (or `results/baseline/test_<input>/`):
* the SR volume `f_NNN.tif` (skip it with `--no_save`),
* `metrics.json`,
* an example slice for the figures.

Metrics are computed inside the mask on every 10th slice (`--eval_stride`): PSNR, SSIM, LPIPS, BV/TV and Tb.Th (2D local thickness).

**4. Uncertainty along synthetic → real input (optional):**

```bash
python uncertainty.py --root $DATA_ROOT                  # or: bash jobs/submit.sh jobs/uncertainty.sh
```

The input is blended as `x_α = (1 − α) · synthetic + α · real`, for α = 0, 0.1, …, 1. Each trained model super-resolves N = 10 noisy versions of every blend, on every 50th slice of the test femurs (`--n_samples`, `--slice_stride`). Noise is added in two parts:
* **Equalisation:** one fixed draw per α tops the blend's noise level up to that of the real CT. Every α then has the same total noise, so only the structure of the input changes. With `--noise fixed` this step is skipped.
* **Perturbation:** a fresh draw for each repetition, with the same strength for every α (`--perturb_scale` × the real CT's noise std, estimated from the data).

The uncertainty is the per-voxel standard deviation of the N outputs, and the error is the difference between their mean and the micro-CT. `results/uncertainty/` contains:
* `uncertainty.csv`: mean uncertainty and error per run, femur and α, the Spearman correlation between uncertainty and error, and AUSE (area between the sparsification curve and the oracle; lower means the uncertainty ranks errors better);
* `uncertainty_vs_alpha.png`;
* `examples_<run>.png`: input, SR, micro-CT, uncertainty and error maps at α = 0, 0.5 and 1.

The noise is white Gaussian, which is a simplification: real CT noise is spatially correlated.

**5. Project page:**

```bash
python scripts/make_results.py     # -> docs/_data/results.json, docs/assets/img/...
git add docs && git commit -m "Update results" && git push
```

### Logging with Weights & Biases

`train.py` and `test.py` log to [Weights & Biases](https://wandb.ai) with `--wandb`. The LSF jobs pass it by default; pass `WANDB=0` to `jobs/submit.sh` to switch it off.

**One-time setup** (on a login node, with the venv active):

```bash
pip install wandb      # only needed if the venv was created before wandb was added to requirements.txt
wandb login            # paste the API key from https://wandb.ai/authorize
```

The key is saved in `~/.netrc`, which the compute nodes can read too.

**What is logged:**
* **Training** (one W&B run per model): every 100 iterations (`--log_every`), the L1, adversarial and discriminator losses, the mean discriminator outputs for real and fake patches, and the learning rate. Every 2000 iterations (`--val_every`), the validation L1/PSNR, a sample image grid, the seconds per iteration and the GPU memory use.
  * The W&B run id is stored in `runs/<name>/wandb_id.txt`. A resubmitted job therefore continues the same W&B run, and its curves are plotted against `iter`.
* **Testing** (one W&B run per model and test input): metrics per femur, a results table, the mean metrics, and example slices (input | SR | micro-CT).
  * Test runs share the training run's *group*, so W&B shows them together.

**Options:**
* `--wandb_project` (default `FACTS`), `--wandb_entity` (your user or team) and `--wandb_tags`.
* The standard environment variables also work, e.g. `WANDB_ENTITY`.
* If the compute nodes cannot reach wandb.ai, pass `WANDB_MODE=offline` to `jobs/submit.sh`. Afterwards, upload the logs from a login node with `wandb sync runs/<name>/wandb/offline-run-*`.

### LSF jobs (DTU HPC)

DTU HPC does not pass environment variables from `bsub` on to the job. Settings are therefore given to `jobs/submit.sh`, which writes them into the job before submitting it. They take precedence over `local.sh`.

```bash
bash jobs/submit.sh jobs/preprocess.sh HM_ARGS="--offset 1024"            # all femurs
bash jobs/submit.sh jobs/preprocess.sh FEMURS="001" HM_ARGS="--offset 1024"
bash jobs/submit_all.sh                     # 4 training jobs + dependent test jobs + baseline
```

`HM_ARGS="--offset 1024"` is needed when the clinical CT is in HU (negative values). Settings that are the same for every job, such as `HM_ARGS`, can also go in `local.sh`.

Single jobs are submitted like this; anything after `--` is passed on to `bsub`:

```bash
bash jobs/submit.sh jobs/train.sh MODEL=ESRGAN DATA=real TRAIN_ARGS="--train_femurs 001 --name esrgan_real_f001" -- -J esrgan_real_f001
bash jobs/submit.sh jobs/test.sh RUN=runs/esrgan_real -- -J test_esrgan_real
```

Each job prints the settings it received at the top of its log (`logs/<job>_<jobid>.out`). If a training job hits the 24 h wall time, resubmit it with the same command; it resumes from the last checkpoint. The queue (`gpua100`), memory and wall time are set in the `#BSUB` headers. Check jobs with `bjobs` and queues with `bstat`/`bqueues`.

## Project page

The page in `docs/` is built by GitHub Pages: *Settings → Pages → Deploy from a branch → `main` / `/docs`*. The result tables and figures are filled in from `docs/_data/results.json`. To preview it locally, run `bundle exec jekyll serve -s docs` (requires Jekyll).

## Citing

See [`CITATION.cff`](CITATION.cff) or the [project page](https://phialosophy10.github.io/FACTS/#citing).
