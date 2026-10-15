#!/bin/bash
# Histogram matching + synthetic LR for all femurs (CPU job).
# Submit from the repository root:   bsub < jobs/preprocess.sh
# Only some femurs:    FEMURS="001 002" bsub < jobs/preprocess.sh
# DATA_ROOT is read from local.sh (or give it on the command line: DATA_ROOT=... bsub < ...)
# Extra arguments for histogram_match.py (e.g. --offset 1024) in HM_ARGS
#BSUB -J facts_preproc
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/preproc_%J.out
#BSUB -e logs/preproc_%J.err

set -e
source init.sh
echo "Settings: DATA_ROOT=$DATA_ROOT | FEMURS=${FEMURS:-<all>} | HM_ARGS=${HM_ARGS:-<none>}"
: "${DATA_ROOT:?Set DATA_ROOT in local.sh (see local.sh.example)}"

FEMUR_ARGS=""
if [ -n "$FEMURS" ]; then FEMUR_ARGS="--femurs $FEMURS"; fi

python scripts/histogram_match.py --root "$DATA_ROOT" $FEMUR_ARGS $HM_ARGS
python scripts/make_synthetic.py --root "$DATA_ROOT" $FEMUR_ARGS --workers 8
python scripts/check_data.py --root "$DATA_ROOT" $FEMUR_ARGS
