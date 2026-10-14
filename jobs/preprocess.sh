#!/bin/bash
# Histogram matching + synthetic LR for all femurs (CPU job).
# Submit from the repository root:   DATA_ROOT=/path/to/FACTS bsub < jobs/preprocess.sh
#BSUB -J facts_preproc
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/preproc_%J.out
#BSUB -e logs/preproc_%J.err

set -e
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root directory}"
source init.sh

python scripts/histogram_match.py --root "$DATA_ROOT"
python scripts/make_synthetic.py --root "$DATA_ROOT" --workers 8
