#!/bin/bash
# Wait for cache expansion to complete, then launch all workers
echo "Waiting for cache expansion to complete..."
while pgrep -f "expand_cache" > /dev/null 2>&1; do
    sleep 30
done
echo "Cache expansion done. Launching workers..."
sleep 5
bash /data/qlyu/s3/node23_launch_all.sh
