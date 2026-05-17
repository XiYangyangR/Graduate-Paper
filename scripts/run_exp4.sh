#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "========================================================="
echo " 🚀 阶段 1：调用现有 LoRA 模型进行多路径 MC Dropout 采样"
echo "========================================================="
python -m src.mc_sampling

echo "========================================================="
echo " 🏋️ 阶段 2：基于隐式集成增强语料训练最终模型"
echo "========================================================="
python -m src.train config/experiments4.yaml

echo "========================================================="
echo " 📊 阶段 3：在独立测试集上评估实验四最终性能"
echo "========================================================="
FINAL_DIR="outputs/exp4_mc_final/final"
python -m src.evaluate config/experiments4.yaml ${FINAL_DIR}

echo "🎉 实验四全流水线执行完毕！请通过日志查看 BLEU 和 chrF 指标。"