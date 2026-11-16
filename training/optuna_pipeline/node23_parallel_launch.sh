#!/bin/bash
# Launch parallel endpoint workers: 8 concurrent processes, each gets ~8 cores
OUTDIR=/data/qlyu/s3/parallel_logs
mkdir -p $OUTDIR

WORKER=/data/qlyu/s3/node23_parallel_worker.py
ENDPOINTS=$(ls /data/qlyu/s3_per/*__clean.tsv | sed 's/.*\///;s/__clean.tsv//' | sort)

echo "Starting parallel workers for all endpoints..."
echo "Endpoints: $ENDPOINTS"
echo ""

MAX_PARALLEL=8
RUNNING=0

for EP in $ENDPOINTS; do
    # Skip if already done
    if [ -f "/data/qlyu/s3/complete/ckpt/${EP}.json" ]; then
        echo "[SKIP] $EP (already done)"
        continue
    fi

    # Wait if too many running
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        sleep 5
        RUNNING=$(pgrep -cf "node23_parallel_worker" || echo 0)
        RUNNING=${RUNNING:-0}
        if [ "$RUNNING" -lt "$MAX_PARALLEL" ]; then
            break
        fi
    done

    echo "[START] $EP"
    nohup python3 $WORKER "$EP" > "$OUTDIR/${EP}.log" 2>&1 &
    RUNNING=$((RUNNING + 1))
    echo "  PID=$! | Running=$RUNNING/$MAX_PARALLEL"
done

echo ""
echo "All workers launched. Waiting for completion..."
wait

echo ""
echo "All endpoints complete!"
echo "Results in /data/qlyu/s3/complete/models/"
ls -la /data/qlyu/s3/complete/models/
