#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 添加参数处理
PRETRAINED=0
PRETRAINED_PATH=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained)
            PRETRAINED=1
            shift
            ;;
        --pretrained_path)
            PRETRAINED_PATH="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 构建torchrun命令
CMD="torchrun --nproc_per_node=8 --master_port=29500 mainold.py"

# 根据参数添加选项
if [ $PRETRAINED -eq 1 ]; then
    CMD="$CMD --pretrained"
fi

if [ ! -z "$PRETRAINED_PATH" ]; then
    CMD="$CMD --pretrained_path $PRETRAINED_PATH"
fi

# 执行命令
eval $CMD 