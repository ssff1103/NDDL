import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider  # Install pycocoevalcap
from collections import Counter



# 读取数据
def read_test_output(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 计算 BLEU 分数
def calculate_bleu(data):
    references = [[item['true_caption'].split()] for item in data]
    hypotheses = [item['generated_caption'].split() for item in data]
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

# 计算 METEOR 分数
def calculate_meteor(data):
    scores = []
    for item in data:
        true_caption = item['true_caption'].split()
        generated_caption = item['generated_caption'].split()
        score = meteor_score([true_caption], generated_caption)
        scores.append(score)
    return sum(scores) / len(scores)

# 计算 ROUGE 分数
def calculate_rouge(data):
    rouge_scorer = Rouge()
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    for item in data:
        pred = item['generated_caption']
        ref = item['true_caption']
        pred_text = " ".join([str(word) for word in pred])
        ref_text = " ".join([str(word) for word in ref]) 
        scores = rouge_scorer.get_scores(pred_text, ref_text, avg=True)
        #rouge_1_score = scores['rouge-1']['f']
        #rouge_2_score = scores['rouge-2']['f']
        #rouge_l_score = scores['rouge-l']['f']
        rouge_1_scores.append(scores['rouge-1']['f'])
        rouge_2_scores.append(scores['rouge-2']['f'])
        rouge_l_scores.append(scores['rouge-l']['f'])
    n = len(data)
    #print(len(data))
    return sum(rouge_1_scores) / n, sum(rouge_2_scores) / n, sum(rouge_l_scores) / n

# 主函数
def main():
    file_path = 'output_llm.json'  # 替换为实际文件路径
    data = read_test_output(file_path)

    bleu = calculate_bleu(data)
    meteor = calculate_meteor(data)
    rouge_1, rouge_2, rouge_l = calculate_rouge(data)
    print(f"BLEU: {bleu:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"ROUGE-1: {rouge_1:.4f}")
    print(f"ROUGE-2: {rouge_2:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")

if __name__ == "__main__":
    main()