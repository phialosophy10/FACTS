#!/bin/bash
# Submit a job script with settings written into the job itself, so they arrive in the job
# even when the cluster does not pass environment variables on from bsub (as on DTU HPC).
# Settings given here take precedence over those in local.sh.
#
# Usage:  bash jobs/submit.sh <job script> [NAME=value ...] [-- <extra bsub options>]
#
# Examples:
#   bash jobs/submit.sh jobs/train.sh MODEL=ESRGAN DATA=real TRAIN_ARGS="--train_femurs 001 --name esrgan_real_f001"
#   bash jobs/submit.sh jobs/test.sh RUN=runs/esrgan_real -- -J test_esrgan_real -w "done(train_esrgan_real)"
#   bash jobs/submit.sh jobs/preprocess.sh FEMURS="001 002" HM_ARGS="--offset 1024"
set -e
JOB="$1"
shift || true
if [ -z "$JOB" ] || [ ! -f "$JOB" ]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

EXPORTS=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
        *=*) EXPORTS+="export ${1%%=*}=$(printf '%q' "${1#*=}")"$'\n' ;;
        *) echo "Expected NAME=value, got '$1'"; exit 1 ;;
    esac
    shift
done
[ "$1" = "--" ] && shift

mkdir -p logs
# Insert the exports right after 'source init.sh', so they take precedence over local.sh
grep -q '^source init.sh' "$JOB" || { echo "$JOB has no 'source init.sh' line"; exit 1; }
awk -v exports="$EXPORTS" '{print} /^source init.sh/ && !done {printf "%s", exports; done=1}' "$JOB" | bsub "$@"
