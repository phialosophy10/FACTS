# FACTS: Femur Archaeological CT Superresolution

FACTS is a benchmark for real-world 3D super-resolution. It contains 13 human proximal femurs, each scanned with both clinical CT (LR) and micro-CT (HR). The two scans of each bone are registered voxel to voxel. This repository contains the preprocessing, the 2D SRGAN/ESRGAN benchmark, and the [project page](https://phialosophy10.github.io/FACTS/).

* Data: <https://archive.compute.dtu.dk/files/public/projects/FACTS>
* Paper: [Superresolution of real-world multiscale bone CT verified with clinical bone measures](https://doi.org/10.1007/978-3-031-66958-3_12) (MIUA 2024)

## Repository layout

```
facts/                    library code
  config.py               femur IDs, train/test split, voxel size, synthetic parameters
  io.py                   paths and (memory-mapped) TIFF I/O
  histmatch.py            masked histogram matching
  synthetic.py            Gaussian blur + 4x down/up-sampling
  data.py                 patch sampling inside the bone mask, intensity normalisation
  models.py               SRGAN / ESRGAN generators and discriminator (2D)
  metrics.py              PSNR, SSIM, LPIPS, BV/TV, Tb.Th
scripts/
  histogram_match.py      HR -> LR intensity scale     => f_NNN/HR/f_NNN_hm.tif
  make_synthetic.py       synthetic LR from HR_hm      => f_NNN/SY/f_NNN_hm.tif
  make_results.py         metrics + figures -> project page (docs/)
train.py                  training (L1 + adversarial loss)
test.py                   inference + evaluation (incl. cross-testing and baseline)
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
source init.sh            # first call creates the venv, later calls only activate it
```

`init.sh` loads a Python module, creates a venv in `~/venvs/facts`, installs PyTorch (CUDA wheel) and `requirements.txt`, and downloads the LPIPS weights. You can override the defaults:

```bash
PY_MODULE=python3/3.12.4 TORCH_CUDA=cu126 VENV_DIR=/work3/$USER/venvs/facts source init.sh
```

**Choosing versions:**
* Run `module avail python3` on a login node to see the available Python versions. Python 3.11 or 3.12 is a safe choice.
* PyTorch pip wheels ship their own CUDA runtime, so only the GPU driver matters. Open an interactive GPU node (e.g. `a100sh` or `voltash`) and run `nvidia-smi`. The "CUDA Version" in the header is the newest CUDA the driver supports, so pick a wheel at or below it: `cu126` for CUDA ≥ 12.6, `cu124` for ≥ 12.4, `cu121` for ≥ 12.1, `cu118` for 11.8.
* Check the install with `python -c "import torch; print(torch.cuda.is_available())"` on a GPU node.

## Running

Run all steps from the repository root.

**1. Preprocessing** (CPU):

```bash
python scripts/histogram_match.py --root $DATA_ROOT     # HR -> f_NNN_hm.tif
python scripts/make_synthetic.py  --root $DATA_ROOT     # SY/f_NNN_hm.tif
```

The histogram of the micro-CT inside the mask is matched to that of the clinical CT inside the same mask. The mapping is then applied to the whole micro-CT volume. If the clinical CT intensities are in HU (negative values), uint16 cannot hold them: the script stops and suggests `--offset 1024` (or use `--dtype float32`). Histograms and the mapping are saved to `results/preprocessing/`.

**2. Training** (4 runs: {SRGAN, ESRGAN} × {real, synth}):

```bash
python train.py --root $DATA_ROOT --model ESRGAN --data real     # -> runs/esrgan_real/
```

The defaults are 10k L1-only warm-up iterations, then 100k total iterations with L1 + 1e-3 · adversarial loss (no perceptual loss), batch 16, 128×128 patches, Adam lr 1e-4. Slices are taken along axis 2 (`--axis`). Training resumes automatically from `runs/<name>/checkpoint.pt`. The run directory also holds `log.csv` and sample images; `--wandb` logs to Weights & Biases.

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

**4. Project page:**

```bash
python scripts/make_results.py     # -> docs/_data/results.json, docs/assets/img/...
git add docs && git commit -m "Update results" && git push
```

### LSF jobs (DTU HPC)

```bash
mkdir -p logs
DATA_ROOT=/path/to/FACTS bsub < jobs/preprocess.sh
DATA_ROOT=/path/to/FACTS bash jobs/submit_all.sh       # 4 training jobs + dependent test jobs + baseline
```

Single jobs are submitted like this:

```bash
DATA_ROOT=... MODEL=ESRGAN DATA=real bsub -J train_esrgan_real < jobs/train.sh
```

If a training job hits the 24 h wall time, resubmit it; it resumes from the last checkpoint. The queue (`gpua100`), memory and wall time are set in the `#BSUB` headers. Check queues with `bstat`/`bqueues`.

## Project page

The page in `docs/` is built by GitHub Pages: *Settings → Pages → Deploy from a branch → `main` / `/docs`*. The result tables and figures are filled in from `docs/_data/results.json`. To preview it locally, run `bundle exec jekyll serve -s docs` (requires Jekyll).

## Citing

See [`CITATION.cff`](CITATION.cff) or the [project page](https://phialosophy10.github.io/FACTS/#citing).
