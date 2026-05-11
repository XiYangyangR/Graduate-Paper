# 中文-哈尼语翻译实验一：直接微调预训练模型

## 环境准备
pip install -r requirements.txt

## 数据准备
1. 将原始平行语料（如 all.zh, all.hani）放入 data/ 目录。
2. 运行划分脚本：
   python scripts/split_data.py --src data/all.zh --tgt data/all.hani --output data

## 训练
bash scripts/run_exp1.sh

## 查看 TensorBoard
tensorboard --logdir=outputs/exp1_nllb/logs --port=6008

## 单独评估
bash scripts/evaluate_test.sh outputs/exp1_nllb/final

## 对比不同预训练模型
修改 config/experiment1.yaml 中的 model.name 和 model.tgt_lang 即可。