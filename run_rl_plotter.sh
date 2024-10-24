#!/bin/bash

# 定义路径数组
paths=(
    "/home/fanjiangdong/workspace/ILMAR(d4rl)/plots/Ant-v2"
    "/home/fanjiangdong/workspace/ILMAR(d4rl)/plots/Hopper-v2"
    "/home/fanjiangdong/workspace/ILMAR(d4rl)/plots/HalfCheetah-v2"
    "/home/fanjiangdong/workspace/ILMAR(d4rl)/plots/Walker2d-v2"
)

# 遍历每个路径
for path in "${paths[@]}"; do
    # 进入指定路径
    cd "$path" || { echo "无法进入目录 $path"; exit 1; }

    # 执行 rl_plotter 命令
    rl_plotter --save --show --avg_group --shaded_err --no_legend_group_num --legend_outside
done
