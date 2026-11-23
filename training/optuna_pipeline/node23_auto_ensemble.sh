#!/bin/bash
# Watcher: wait for the phase-1 Optuna pipeline (all workers exit), then run
# the phase-2 ensemble builder for every endpoint that has a usable ckpt.
LOGDIR=${WORK_DIR}/s3/ensemble_logs
mkdir -p "$LOGDIR"
echo "$(date '+%F %T') waiting for phase-1 workers to finish..."
while pgrep -f '[n]ode23_parallel_worker' > /dev/null; do
    sleep 60
done
sleep 15
echo "$(date '+%F %T') phase-1 done. ckpts: $(ls ${WORK_DIR}/s3/complete/ckpt/*.json 2>/dev/null | wc -l)"
ls ${WORK_DIR}/s3/complete/ckpt/*.json | sed 's/.*ckpt\///; s/\.json//' \
  | xargs -P 4 -I{} sh -c "python3 ${WORK_DIR}/s3/node23_ensemble.py {} > $LOGDIR/{}.log 2>&1"
echo "$(date '+%F %T') ensemble phase complete:"
grep -h 'DONE' $LOGDIR/*.log | sort
