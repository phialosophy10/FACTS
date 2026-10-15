#!/bin/bash
# Train one model. Submit from the repository root, e.g.
#   MODEL=ESRGAN DATA=real bsub -J esrgan_real < jobs/train.sh
# DATA_ROOT is read from local.sh (or give it on the command line: DATA_ROOT=... bsub < ...)
# Training resumes from the last checkpoint, so the job can simply be resubmitted
# if it hits the wall-time limit. Extra train.py arguments can be passed in TRAIN_ARGS.
#BSUB -J facts_train
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -W 24:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

set -e
source init.sh
echo "Settings: DATA_ROOT=$DATA_ROOT | MODEL=$MODEL | DATA=$DATA | TRAIN_ARGS=${TRAIN_ARGS:-<none>} | WANDB=${WANDB:-1}"
: "${DATA_ROOT:?Set DATA_ROOT in local.sh (see local.sh.example)}" "${MODEL:?Set MODEL (SRGAN/ESRGAN)}" "${DATA:?Set DATA (real/synth)}"
nvidia-smi

# Weights & Biases logging is on by default (WANDB=0 to disable); run `wandb login` once first
WANDB_FLAG=""
if [ "${WANDB:-1}" = "1" ]; then WANDB_FLAG="--wandb"; fi

python train.py --root "$DATA_ROOT" --model "$MODEL" --data "$DATA" --workers 7 $WANDB_FLAG $TRAIN_ARGS
