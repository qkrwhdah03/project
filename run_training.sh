#!/bin/bash
cd /root/ydk/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate project

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True는 CUDA driver error를 일으킴
# 제거하거나 다른 설정 사용
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 타임아웃 설정 (5시간), unbuffered 출력
timeout 18000 python -u train_dolly.py > train.log 2>&1

echo "Training completed or timed out at $(date)" >> train.log