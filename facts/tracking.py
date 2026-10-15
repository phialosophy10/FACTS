"""Optional experiment tracking with Weights & Biases (wandb).

The wandb run id is stored in the output directory, so a resumed training
job (e.g. resubmitted after hitting the wall-time limit) continues the same
wandb run instead of starting a new one.

Standard wandb environment variables are respected, e.g.
    WANDB_MODE=offline   log locally (sync later with `wandb sync <dir>`)
    WANDB_API_KEY=...    instead of `wandb login`
"""
import os
import uuid


def add_wandb_args(parser):
    g = parser.add_argument_group("Weights & Biases")
    g.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    g.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "FACTS"))
    g.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"),
                   help="wandb user or team (default: your default entity)")
    g.add_argument("--wandb_tags", nargs="*", default=[])


def init_wandb(args, out_dir, name, config, job_type, group=None):
    """Start (or resume) a wandb run whose id is kept in <out_dir>/wandb_id.txt."""
    import wandb

    id_file = os.path.join(out_dir, "wandb_id.txt")
    if os.path.exists(id_file):
        with open(id_file) as f:
            run_id = f.read().strip()
    else:
        run_id = uuid.uuid4().hex[:8]
        with open(id_file, "w") as f:
            f.write(run_id)

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name, id=run_id,
                     resume="allow", config=config, job_type=job_type, group=group,
                     tags=args.wandb_tags, dir=out_dir)
    return run
