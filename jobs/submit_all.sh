#!/bin/bash
# Submit the full benchmark: 4 trainings (SRGAN/ESRGAN x real/synth), the matching
# test jobs (each tested on both real and synthetic input) and the baseline.
# Run from the repository root:   bash jobs/submit_all.sh   (DATA_ROOT from local.sh)
# Set PREPROCESS=1 to also run the preprocessing first.
set -e
if [ -z "$DATA_ROOT" ] && [ -f local.sh ]; then source local.sh; fi
: "${DATA_ROOT:?Set DATA_ROOT in local.sh (see local.sh.example)}"
mkdir -p logs
export DATA_ROOT

DEP=""
if [ "${PREPROCESS:-0}" = "1" ]; then
    bsub -J facts_preproc < jobs/preprocess.sh
    DEP="-w done(facts_preproc)"
fi

for MODEL in SRGAN ESRGAN; do
    for DATA in real synth; do
        NAME=$(echo "${MODEL}_${DATA}" | tr '[:upper:]' '[:lower:]')
        MODEL=$MODEL DATA=$DATA bsub -J "train_$NAME" $DEP < jobs/train.sh
        RUN=runs/$NAME bsub -J "test_$NAME" -w "done(train_$NAME)" < jobs/test.sh
    done
done
RUN=baseline bsub -J test_baseline $DEP < jobs/test.sh
