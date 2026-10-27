#!/bin/bash
# Uncertainty sweep from synthetic to real input for the trained models (see uncertainty.py).
# Submit from the repository root:   bash jobs/submit.sh jobs/uncertainty.sh
# Extra uncertainty.py arguments in UNC_ARGS, e.g. UNC_ARGS="--n_samples 20 --slice_stride 25".
#BSUB -J facts_uncertainty
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o logs/uncertainty_%J.out
#BSUB -e logs/uncertainty_%J.err

set -e
source init.sh
echo "Settings: DATA_ROOT=$DATA_ROOT | UNC_ARGS=${UNC_ARGS:-<none>}"
: "${DATA_ROOT:?Set DATA_ROOT in local.sh (see local.sh.example)}"
nvidia-smi

python uncertainty.py --root "$DATA_ROOT" $UNC_ARGS
