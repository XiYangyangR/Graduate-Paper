#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m src.train config/experiment2.yaml
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('config/experiment2.yaml'))['training']['output_dir'])")
FINAL_DIR="${OUTPUT_DIR}/final"
python -m src.evaluate config/experiment2.yaml ${FINAL_DIR}