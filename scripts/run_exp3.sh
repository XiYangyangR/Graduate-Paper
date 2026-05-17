#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "=== 阶段 1：训练 3 个不同种子的反向模型 ==="
python -m src.train config/exp3_bwd_seed42.yaml
python -m src.train config/exp3_bwd_seed43.yaml
python -m src.train config/exp3_bwd_seed44.yaml

echo "=== 阶段 2：训练 3 个不同种子的正向模型 ==="
python -m src.train config/exp3_fwd_seed42.yaml
python -m src.train config/exp3_fwd_seed43.yaml
python -m src.train config/exp3_fwd_seed44.yaml

echo "=== 阶段 3：执行数据多样化生成与合并 ==="
python -m src.data_diversification

echo "=== 阶段 4：训练最终的 Data Diversification 模型 ==="
# 注意：你需要再建一个 exp3_final.yaml（数据路径指向 augmented_data/exp3_merged/）
# python -m src.train config/exp3_final.yaml