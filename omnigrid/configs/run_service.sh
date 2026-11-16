#!/bin/bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
while true; do
  python3 /data/qlyu/s3/omnigrid/node23_omnigrid_service.py >> /data/qlyu/s3/omnigrid/service.log 2>&1
  echo restarted >> /data/qlyu/s3/omnigrid/service.log
  sleep 5
done
