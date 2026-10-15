#!/bin/bash
# Submit the full benchmark: 4 trainings (SRGAN/ESRGAN x real/synth), the matching
# test jobs (each tested on both real and synthetic input) and the baseline.
# Run from the repository root:   bash jobs/submit_all.sh   (DATA_ROOT from local.sh)
# Set PREPROCESS=1 to also run the preprocessing first.
set -e

DEP=()
if [ "${PREPROCESS:-0}" = "1" ]; then
    bash jobs/submit.sh jobs/preprocess.sh -- -J facts_preproc
    DEP=(-w "done(facts_preproc)")
fi

for MODEL in SRGAN ESRGAN; do
    for DATA in real synth; do
        NAME=$(echo "${MODEL}_${DATA}" | tr '[:upper:]' '[:lower:]')
        bash jobs/submit.sh jobs/train.sh MODEL=$MODEL DATA=$DATA -- -J "train_$NAME" "${DEP[@]}"
        bash jobs/submit.sh jobs/test.sh RUN=runs/$NAME -- -J "test_$NAME" -w "done(train_$NAME)"
    done
done
bash jobs/submit.sh jobs/test.sh RUN=baseline -- -J test_baseline "${DEP[@]}"
