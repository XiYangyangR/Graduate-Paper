#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
REVERSE_MODEL="outputs/exp2_reverse/final"
MONO_DATA="data/mono.hani"

if [ ! -f "$MONO_DATA" ]; then
    echo "错误: 未找到哈尼语单语文件 $MONO_DATA"
    echo "请将单语数据放入 data/mono.hani，每行一句哈尼语。"
    exit 1
fi

python -m src.back_translation \
    --config config/exp2_reverse.yaml \
    --model $REVERSE_MODEL \
    --mono $MONO_DATA \
    --output_prefix augmented_data/pseudo/pseudo \
    --batch_size 32