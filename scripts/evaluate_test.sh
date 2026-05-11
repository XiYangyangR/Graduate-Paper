#!/bin/bash
CONFIG=${1:-"config/experiment1.yaml"}

if [ -z "$2" ]; then
    MODEL_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['output_dir'])" )
    MODEL_DIR="${MODEL_DIR}/final"
else
    MODEL_DIR="$2"
fi

python -m src.evaluate $CONFIG $MODEL_DIR