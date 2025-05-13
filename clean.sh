#!/bin/bash

KEEP_PID=1521650
echo "[INFO] Keeping PID=$KEEP_PID and killing other GPU-bound zombie Python processes..."

# 找出佔用 GPU 的所有 Python PID
for pid in $(fuser -v /dev/nvidia* 2>/dev/null | grep python | awk '{print $2}' | sort -u); do
    if [ "$pid" != "$KEEP_PID" ]; then
        echo "[KILL] PID $pid"
        kill -9 $pid
    else
        echo "[SKIP] PID $pid (active job)"
    fi
done
