#!/bin/bash
# Test a trained model on real and synthetic input (cross-testing). Submit e.g.
#   DATA_ROOT=/path/to/FACTS RUN=runs/esrgan_real bsub -J test_esrgan_real < jobs/test.sh
# With RUN=baseline the inputs themselves are evaluated (LR reference).
#BSUB -J facts_test
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o logs/test_%J.out
#BSUB -e logs/test_%J.err

set -e
: "${DATA_ROOT:?Set DATA_ROOT}" "${RUN:?Set RUN (run directory or 'baseline')}"
source init.sh

# Weights & Biases logging is on by default (WANDB=0 to disable)
WANDB_FLAG=""
if [ "${WANDB:-1}" = "1" ]; then WANDB_FLAG="--wandb"; fi

for INPUT in real synth; do
    if [ "$RUN" = "baseline" ]; then
        python test.py --root "$DATA_ROOT" --baseline --input "$INPUT" $WANDB_FLAG $TEST_ARGS
    else
        python test.py --root "$DATA_ROOT" --run "$RUN" --input "$INPUT" $WANDB_FLAG $TEST_ARGS
    fi
done
