#!/bin/bash

echo "========================================"
echo "神经网络与深度学习课程设计 - 依赖安装"
echo "========================================"
echo ""

echo "[1/3] 安装基础依赖包..."
pip install torch torchvision torchaudio
pip install Pillow numpy matplotlib tqdm
pip install nltk
pip install rouge-score
pip install pycocoevalcap
pip install requests

echo ""
echo "[2/3] 安装大模型微调依赖（可选，仅Model_finetune_Qwen2-VL需要）..."
echo "如果需要使用Qwen2-VL模型，请取消下面命令的注释"
# pip install transformers
# pip install accelerate
# pip install peft

echo ""
echo "[3/3] 下载NLTK数据..."
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "注意："
echo "1. 如果使用GPU，请根据CUDA版本安装对应的PyTorch版本"
echo "2. GloVe词向量需要手动下载：https://nlp.stanford.edu/projects/glove/"
echo "3. 大模型微调需要较大显存，请根据实际情况安装transformers等库"

