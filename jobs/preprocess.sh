#!/bin/bash
# Histogram matching + synthetic LR for all femurs (CPU job).
# Submit from the repository root:   bash jobs/submit.sh jobs/preprocess.sh
# Only some femurs:    bash jobs/submit.sh jobs/preprocess.sh FEMURS="001 002"
# Extra arguments for histogram_match.py (e.g. --method linear --offset 2048) in HM_ARGS.
# OVERWRITE=1 recomputes femurs that were already processed.
# DATA_ROOT (and defaults for FEMURS/HM_ARGS) are read from local.sh.
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
echo "Settings: DATA_ROOT=$DATA_ROOT | FEMURS=${FEMURS:-<all>} | HM_ARGS=${HM_ARGS:-<none>} | OVERWRITE=${OVERWRITE:-0}"
: "${DATA_ROOT:?Set DATA_ROOT in local.sh (see local.sh.example)}"

FEMUR_ARGS=""
if [ -n "$FEMURS" ]; then FEMUR_ARGS="--femurs $FEMURS"; fi
# OVERWRITE=1 recomputes existing HR_hm/SY_hm files
OVERWRITE_ARG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_ARG="--overwrite"; fi

python scripts/histogram_match.py --root "$DATA_ROOT" $FEMUR_ARGS $OVERWRITE_ARG $HM_ARGS
python scripts/make_synthetic.py --root "$DATA_ROOT" $FEMUR_ARGS $OVERWRITE_ARG --workers 8
python scripts/check_data.py --root "$DATA_ROOT" $FEMUR_ARGS
