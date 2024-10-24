#!/bin/bash

# Python 脚本路径
PYTHON_SCRIPT="train_imitation.py"

# 算法列表
# ALGOS=("mybc" "demodice" "iswbc")
ALGOS=("mybc" "metademodice" "metaiswbc")
# 环境列表
ENVS=("Ant-v2" "Hopper-v2" "HalfCheetah-v2" "Walker2d-v2")
# ENVS=("Hopper-v2" "Walker2d-v2")
# 随机种子列表
# SEEDS=(2022 2023 2024 2025 2026)
SEEDS=(2022)
# GPU 设备列表
DEVICES=(0 1)

# 初始化 GPU 计数器
gpu_index=0

# 遍历所有算法、环境和随机种子
for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # 分配 GPU 设备
            device=${DEVICES[gpu_index]}
            
            # 构建参数集并执行 Python 脚本
            echo "Running: algo=$algo, env=$env, seed=$seed, device=$device"
            CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --algo "$algo" --env-id "$env" --seed "$seed" &  # 在后台执行
            # 更新 GPU 计数器 (循环使用 0-3)
            gpu_index=$(( (gpu_index + 1) % 2 ))
        done
    done
done

# 等待所有后台任务完成
wait
echo "All processes finished."

