#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m src.train config/experiment1.yaml

# 从配置文件中读取 output_dir
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('config/experiment1.yaml'))['training']['output_dir'])")
FINAL_DIR="${OUTPUT_DIR}/final"

python -m src.evaluate config/experiment1.yaml ${FINAL_DIR}