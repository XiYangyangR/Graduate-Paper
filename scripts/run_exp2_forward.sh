#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m src.train config/experiments2.yaml
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('config/experiments2.yaml'))['training']['output_dir'])")
FINAL_DIR="${OUTPUT_DIR}/final"
python -m src.evaluate config/experiments2.yaml ${FINAL_DIR}