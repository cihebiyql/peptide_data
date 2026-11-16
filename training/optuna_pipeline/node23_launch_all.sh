#!/bin/bash
# Launch ALL endpoints in parallel (no limit)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for f in /data/qlyu/s3_integrated/*__clean.tsv; do
    ep=$(basename "$f" .tsv | sed 's/__clean//')
    ckpt="/data/qlyu/s3/complete/ckpt/${ep}.json"
    if [ ! -f "$ckpt" ]; then
        if ! pgrep -f "worker.py ${ep}$" > /dev/null 2>&1; then
            nohup python3 /data/qlyu/s3/node23_parallel_worker.py "$ep" > "/data/qlyu/s3/parallel_logs/${ep}.log" 2>&1 &
            echo "START: $ep"
        else
            echo "RUN: $ep"
        fi
    else
        echo "DONE: $ep"
    fi
done
echo "---"
echo "Total workers: $(pgrep -cf node23_parallel_worker)"
sleep 5
uptime
