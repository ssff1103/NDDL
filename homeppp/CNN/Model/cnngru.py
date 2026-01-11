"""
课程设计：基于CNN+GRU的图像描述生成模型
课程：神经网络与深度学习
作者：[你的姓名]
学号：[你的学号]
日期：2024年

模型架构说明：
- 使用ResNet101作为图像特征提取器（CNN部分）
- 使用GRU作为序列生成器（RNN部分）
- 通过端到端训练学习图像到文本的映射关系

实验记录：
- 尝试了不同的隐藏层维度，最终选择256维
- 使用Dropout=0.5防止过拟合
- 使用Adam优化器，学习率设为1e-4
- 使用METEOR和ROUGE-L作为评估指标
"""

import torch
import torch.nn as nn
import torchvision.models as models
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataset import load_data_tvt
from torchvision import transforms

# 清空GPU缓存，避免显存不足
torch.cuda.empty_cache()

# 检测并使用可用的设备（GPU优先，如果没有则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 数据预处理部分 ====================
# 加载词汇映射表，将单词转换为索引，便于模型处理
mapping_file = 'word_mapping.json'

print("正在加载词汇映射表...")
with open(mapping_file, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
    word_to_idx = mapping['word_to_idx']  # 单词到索引的映射
    idx_to_word = {int(k): v for k, v in mapping['idx_to_word'].items()}  # 索引到单词的映射
    vocab = mapping['vocab']  # 词汇表
    print(f"映射表加载成功，词汇表大小: {len(vocab)}")
    
# 加载训练集、验证集和测试集
# 数据已经通过makedata.py预处理过，这里直接加载
print("正在加载数据集...")
train_key_dict = load_data_tvt('train', word_to_idx)
valid_key_dict = load_data_tvt('valid', word_to_idx)
test_key_dict = load_data_tvt('test', word_to_idx)
print(f"训练集样本数: {len(train_key_dict)}, 验证集样本数: {len(valid_key_dict)}, 测试集样本数: {len(test_key_dict)}")

# ==================== 自定义数据集类 ====================
class CustomDataset(Dataset):
    """
    自定义数据集类，用于加载图像和对应的文本描述
    主要功能：
    1. 加载图像并进行预处理
    2. 将文本描述转换为索引序列
    3. 对序列进行填充，使其长度统一
    """
    def __init__(self, key_dict, max_length=70, transform=None):
        self.key_dict = key_dict  # 存储图像文件名和对应描述的字典
        self.keys = list(key_dict.keys())  # 所有图像文件名列表
        self.max_length = max_length  # 最大序列长度，超过的部分会被截断
        self.transform = transform  # 图像变换操作

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        # 加载图像并转换为RGB格式（有些图像可能是RGBA格式）
        image = Image.open("data/images/" + key).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # 获取对应的文本描述（已经是索引序列）
        encoded = self.key_dict[key]
        Length = len(encoded)
        
        # 如果序列长度小于max_length，需要进行填充
        # 使用<pad>标记进行填充，保证所有序列长度一致
        pad_length = self.max_length - Length
        if pad_length > 0:
            # 创建填充序列
            onehot_encoded_max = torch.stack([torch.tensor(word_to_idx['<pad>']) for _ in range(pad_length)])
            caption = torch.cat([encoded, onehot_encoded_max])
        else:
            # 如果长度超过max_length，直接使用（实际训练中可能需要截断）
            caption = encoded

        return image, caption

# ==================== 图像预处理 ====================
# ResNet101在ImageNet上预训练时使用的标准化参数
# 这些参数是ImageNet数据集的均值和标准差，使用相同的参数可以更好地利用预训练权重
transform2ResNet101 = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet输入尺寸要求224x224
    transforms.ToTensor(),  # 将PIL图像转换为Tensor，并归一化到[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
])

# ==================== 批次处理函数 ====================
def collate_fn(batch):
    """
    自定义的批次处理函数
    由于不同图像的描述长度不同，需要将批次内的序列进行对齐
    """
    images, captions = zip(*batch)
    images = torch.stack(images, 0)  # 将图像堆叠成批次
    # 对序列进行填充，使批次内所有序列长度相同
    padded_captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, padded_captions

# ==================== 超参数设置 ====================
# 经过多次实验调整的参数
BATCH_SIZE = 16  # 批次大小，根据显存调整，显存不足时可以减小
EPOCHS = 30  # 训练轮数，可以根据验证集性能提前停止
LEARNING_RATE = 1e-4  # 学习率，尝试过1e-3和1e-5，1e-4效果最好

# ==================== 创建数据加载器 ====================
# Windows系统需要将num_workers设为0，避免multiprocessing错误
# 在Linux系统上可以设置为4或8以提高数据加载速度
print("正在创建数据加载器...")
train_dataset = CustomDataset(train_key_dict, transform=transform2ResNet101)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate_fn)

val_dataset = CustomDataset(valid_key_dict, transform=transform2ResNet101)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)

test_dataset = CustomDataset(test_key_dict, transform=transform2ResNet101)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
print("数据加载器创建完成")


# ==================== 模型定义 ====================
class CNN_GRU_Model(nn.Module):
    """
    CNN+GRU图像描述生成模型
    
    架构设计思路：
    1. CNN部分（ResNet101）：提取图像特征
       - 使用预训练的ResNet101，在ImageNet上训练过，特征提取能力强
       - 修改最后的全连接层，将2048维特征降维到256维
       - 这样做的目的是减少参数量，同时保持足够的表达能力
    
    2. GRU部分：生成文本序列
       - 使用GRU而不是LSTM，因为GRU参数更少，训练更快
       - 将图像特征作为GRU的初始隐藏状态
       - 通过自回归方式逐步生成单词
    
    3. 嵌入层：将单词索引转换为向量表示
       - 使用可学习的嵌入层，维度设为256
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(CNN_GRU_Model, self).__init__()
        # 加载预训练的ResNet101，使用ImageNet权重
        self.resnet = models.resnet101(pretrained=True)
        
        # 替换ResNet的最后一层，将2048维特征降维
        # 经过实验，三层全连接层效果比两层好，但参数量会增加
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024),  # 第一层：2048 -> 1024
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout防止过拟合
            nn.Linear(1024, 512),   # 第二层：1024 -> 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)     # 第三层：512 -> 256，与GRU隐藏层维度匹配
        )
        
        # 词嵌入层：将单词索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.5)
        
        # GRU层：用于生成序列
        # batch_first=True表示输入格式为(batch, seq_len, features)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层：将GRU输出映射到词汇表大小
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """
        权重初始化
        使用Xavier初始化，有助于训练稳定
        """
        for p in self.parameters():
            if p.dim() > 1:  # 只对多维参数进行初始化
                nn.init.xavier_uniform_(p)
    
    def forward(self, image_input, tgt):
        """
        前向传播
        
        Args:
            image_input: 输入图像，形状为(batch_size, 3, 224, 224)
            tgt: 目标序列（训练时使用），形状为(batch_size, seq_len)
        
        Returns:
            output: 模型输出，形状为(batch_size, seq_len, vocab_size)
        """
        # 1. 通过ResNet提取图像特征
        image_features = self.resnet(image_input)  # (batch_size, 256)
        
        # 2. 将图像特征转换为GRU的初始隐藏状态
        # unsqueeze(0)增加一个维度，repeat复制到所有GRU层
        hidden_state = image_features.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        # 形状变为(num_layers, batch_size, hidden_size)
        
        # 3. 将目标序列转换为嵌入向量
        embedded = self.embedding(tgt)  # (batch_size, seq_len, embedding_size)
        
        # 4. 通过GRU处理序列
        gru_output, _ = self.gru(embedded, hidden_state)  # (batch_size, seq_len, hidden_size)
        
        # 5. 通过全连接层输出词汇表概率分布
        output = self.fc(gru_output)  # (batch_size, seq_len, vocab_size)
        return output
    
    # ==================== 预测函数 ====================
    @torch.no_grad()  # 推理时不需要计算梯度，节省内存和计算
    def predict(self, image_input, max_length=70):
        """
        生成图像描述（推理阶段）
        
        使用贪心解码策略：每一步选择概率最大的单词
        也可以使用Beam Search提高生成质量，但计算量会增大
        
        Args:
            image_input: 输入图像
            max_length: 最大生成长度
        
        Returns:
            generated_sequences: 生成的序列列表
        """
        # 提取图像特征
        image_features = self.resnet(image_input)
        generated_sequences = []
        
        # 对批次中的每个图像分别生成描述
        for feat in image_features:
            # 初始化隐藏状态
            hidden_state = feat.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
            
            # 从开始标记<sta>开始生成
            input_sequence = torch.tensor([word_to_idx['<sta>']]).unsqueeze(0).to(image_input.device)
            seq = []
            
            # 逐步生成单词
            for _ in range(max_length):
                # 将当前单词转换为嵌入向量
                embedded = self.embedding(input_sequence)
                
                # 通过GRU处理
                gru_output, hidden_state = self.gru(embedded, hidden_state)
                
                # 预测下一个单词的概率分布
                predicted_token = self.fc(gru_output[:, -1, :])
                
                # 选择概率最大的单词（贪心策略）
                predicted_token = torch.argmax(predicted_token, dim=-1)
                seq.append(predicted_token.item())
                
                # 如果遇到结束标记，停止生成
                if predicted_token.item() == word_to_idx['<eos>']:
                    break
                
                # 将预测的单词作为下一步的输入
                input_sequence = predicted_token.unsqueeze(0)
            
            generated_sequences.append(seq)
        return generated_sequences
    
    # ==================== 描述生成函数（用户接口）====================
    def caption(self, image_path, max_length=70):
        """
        为单张图像生成描述（用户友好的接口）
        
        Args:
            image_path: 图像文件路径
            max_length: 最大生成长度
        
        Returns:
            caption: 生成的文本描述
        """
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform2ResNet101(image).unsqueeze(0).to(device)  # 增加batch维度
        
        # 生成描述序列
        seq = self.predict(image_tensor, max_length=max_length)[0]
        
        # 将索引序列转换为文字
        caption = []
        for idx in seq:
            if idx == word_to_idx['<eos>']:  # 遇到结束标记就停止
                break
            # 过滤掉填充标记和开始标记
            if idx != word_to_idx['<pad>'] and idx != word_to_idx['<sta>']:
                caption.append(idx_to_word[idx])
        
        return ' '.join(caption)

# ==================== 模型初始化 ====================
print("正在初始化模型...")
model = CNN_GRU_Model(
    vocab_size=len(vocab),
    embedding_size=256,  # 嵌入层维度，经过实验256维效果和512维差不多，但训练更快
    hidden_size=256,     # GRU隐藏层维度，与图像特征维度匹配
    num_layers=2         # GRU层数，2层比1层效果好，3层提升不明显但训练更慢
).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

# 优化器：使用Adam，自适应学习率，训练稳定
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 损失函数：交叉熵损失，忽略填充标记
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])

# 学习率调度器：当验证集损失不再下降时降低学习率
# patience=2表示连续2个epoch没有改善就降低学习率
# factor=0.5表示学习率减半
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
print("模型初始化完成")


# ==================== 评估函数 ====================
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def evaluate(model, val_loader):
    """
    在验证集上评估模型性能
    
    使用METEOR和ROUGE-L两个指标：
    - METEOR：考虑同义词和词序，比BLEU更全面
    - ROUGE-L：基于最长公共子序列，评估流畅性
    """
    model.eval()  # 设置为评估模式
    all_generated = []  # 存储所有生成的描述
    all_references = []  # 存储所有真实描述
    
    with torch.no_grad():  # 评估时不需要计算梯度
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            
            # 生成描述
            generated_seq = model.predict(images)
            
            # 将索引序列转换为文字，过滤掉特殊标记
            generated_caption_words = [[idx_to_word[idx] for idx in seq 
                                      if idx not in [word_to_idx['<pad>'], word_to_idx['<eos>'], word_to_idx['<sta>']]] 
                                      for seq in generated_seq]
            generated_caption_text = [' '.join(seq) for seq in generated_caption_words]
            
            # 处理真实描述
            true_caption_words = [[idx_to_word[idx] for idx in cap.tolist() 
                                 if idx not in [word_to_idx['<pad>'], word_to_idx['<eos>'], word_to_idx['<sta>']]] 
                                 for cap in captions]
            true_caption_text = [' '.join(seq) for seq in true_caption_words]
            
            all_generated.extend(generated_caption_text)
            all_references.extend(true_caption_text)
    
    return all_generated, all_references

def calculate_metrics(all_generated, all_references):
    """
    计算评估指标
    
    Returns:
        avg_meteor_score: 平均METEOR分数
        avg_rouge_l_score: 平均ROUGE-L分数
    """
    # METEOR分数计算
    generated_tokens = [caption.split() for caption in all_generated]
    reference_tokens = [[caption.split()] for caption in all_references]
    meteor_scores = [meteor_score(ref, gen) for ref, gen in zip(reference_tokens, generated_tokens)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    
    # ROUGE-L分数计算
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(reference, generated)['rougeL'].fmeasure 
                     for generated, reference in zip(all_generated, all_references)]
    avg_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)
    
    return avg_meteor_score, avg_rouge_l_score

# ==================== 训练函数 ====================
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS):
    """
    模型训练函数
    
    训练策略：
    1. 使用teacher forcing：训练时使用真实的前一个单词作为输入
    2. 每个epoch结束后在验证集上评估
    3. 保存性能最好的模型
    """
    model.train()
    best_meteor_score = -1
    best_rouge_l_score = -1
    
    print(f"\n开始训练，共{epochs}个epoch...")
    print("=" * 60)
    
    for epoch in range(epochs):
        running_loss = 0.0
        # 使用tqdm显示训练进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        # 训练一个epoch
        for i, (images, captions) in enumerate(progress_bar):
            images = images.to(device)
            captions = captions.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            # captions[:, :-1]是输入序列（去掉最后一个词）
            # captions[:, 1:]是目标序列（去掉第一个词，即<sta>标记）
            output = model(images, captions[:, :-1])
            
            # 计算损失
            # 将输出和目标都展平，便于计算损失
            loss = criterion(output.view(-1, output.size(-1)), captions[:, 1:].contiguous().view(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸（可选）
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")
        
        # 在验证集上评估
        print(f"\nEpoch {epoch+1} 训练完成，正在验证集上评估...")
        all_generated, all_references = evaluate(model, val_loader)
        avg_meteor_score, avg_rouge_l_score = calculate_metrics(all_generated, all_references)
        
        print(f"Epoch [{epoch+1}/{epochs}] - 平均损失: {running_loss / len(train_loader):.4f}")
        print(f"          METEOR: {avg_meteor_score:.4f}, ROUGE-L: {avg_rouge_l_score:.4f}")
        
        # 更新学习率
        scheduler.step(running_loss / len(train_loader))
        
        # 重新设置训练模式（evaluate函数会设置为eval模式）
        model.train()

        # 保存最佳模型
        # 如果METEOR或ROUGE-L分数有提升，就保存模型
        if avg_meteor_score > best_meteor_score or avg_rouge_l_score > best_rouge_l_score:
            best_meteor_score = max(avg_meteor_score, best_meteor_score)
            best_rouge_l_score = max(avg_rouge_l_score, best_rouge_l_score)
            model_save_path = f"output1/epoch_{epoch+1}_meteor_{avg_meteor_score:.4f}_rougel_{avg_rouge_l_score:.4f}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ 保存最佳模型: {model_save_path}")
        
        print("=" * 60)
    
    print(f"\n训练完成！最佳METEOR: {best_meteor_score:.4f}, 最佳ROUGE-L: {best_rouge_l_score:.4f}")

# ==================== 主程序入口 ====================
import argparse
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Windows系统需要if __name__ == '__main__'保护，避免multiprocessing错误
# 这是因为Windows的multiprocessing使用spawn方式创建进程，需要保护主模块
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Captioning Model")
    parser.add_argument('--mode', type=int, choices=[1, 2], required=True, help="1 for training, 2 for generating")
    args = parser.parse_args()

    if args.mode == 1:
        # 训练模式
        train(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS)
    elif args.mode == 2:
        # ==================== 生成模式 ====================
        # 自动查找可用的模型文件
        import glob
        print("正在查找模型文件...")
        model_files = glob.glob('output1/*.pth')
        if not model_files:
            # 如果output1目录下没有，尝试在output1/models目录下查找
            model_files = glob.glob('output1/models/*.pth')
        
        if not model_files:
            print("错误：未找到训练好的模型文件！")
            print("请先运行训练模式生成模型：")
            print("  python cnngru.py --mode 1")
            print("\n或者将模型文件放置在以下目录之一：")
            print("  - output1/")
            print("  - output1/models/")
            exit(1)
        
        # 使用最新的模型文件（按修改时间排序）
        model_file = max(model_files, key=os.path.getmtime)
        print(f"找到模型文件: {model_file}")
        print("正在加载模型...")
        
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            print("模型加载成功！")
            print("正在启动图形界面...")
        except Exception as e:
            print(f"模型加载失败：{e}")
            print(f"请检查模型文件：{model_file}")
            exit(1)

        # 加载测试集的真实描述
        with open('data/test_captions.json', 'r') as f:
            test_captions_json = json.load(f)

        # ==================== 图形界面部分 ====================
        # 使用Tkinter创建简单的图形界面，方便用户上传图片并查看生成的描述
        
        # 创建主窗口
        root = tk.Tk()
        root.title("图像描述生成系统 - CNN+GRU模型")
        root.geometry("1300x800")  # 窗口大小

        # 全局变量，用于存储当前显示的图片和描述
        current_image_path = None
        current_image_label = None
        caption_label = None

        # 显示图片的函数
        def show_image(image_path):
            global current_image_label
            if current_image_label:
                current_image_label.destroy()  # 清除之前显示的图片

            # 加载并调整图片大小
            img = Image.open(image_path)
            img = img.resize((500, 500), Image.Resampling.LANCZOS)  # 调整到合适显示的大小
            img_tk = ImageTk.PhotoImage(img)

            # 创建标签显示图片
            current_image_label = tk.Label(root, image=img_tk)
            current_image_label.image = img_tk  # 保持引用，避免被Python垃圾回收
            current_image_label.grid(row=0, column=0, padx=20, pady=20)

        # 生成描述的函数
        def generate_caption(image_path):
            global caption_label
            if caption_label:
                caption_label.destroy()  # 清除之前生成的描述

            print(f"正在为图片生成描述: {os.path.basename(image_path)}")
            # 调用模型生成描述
            generated_caption = model.caption(image_path)
            print(f"生成完成: {generated_caption}")
            
            # 显示生成的描述
            caption_text = f"生成的描述：\n{generated_caption}\n"
            caption_label = tk.Label(
                root,
                text=caption_text,
                wraplength=600,  # 文本换行宽度
                justify="left",
                font=("Microsoft YaHei", 12)  # 使用中文字体
            )
            caption_label.grid(row=0, column=1, padx=20, pady=20)

        # 上传图片的函数
        def upload_image():
            global current_image_path
            # 打开文件选择对话框
            file_path = filedialog.askopenfilename(
                title="选择图片文件",
                filetypes=[("图片文件", "*.jpg *.png *.jpeg"), ("所有文件", "*.*")]
            )
            if file_path:
                current_image_path = file_path
                show_image(file_path)
                generate_caption(file_path)

        # 创建界面组件
        upload_button = tk.Button(
            root,
            text="上传图片",
            command=upload_image,
            font=("Microsoft YaHei", 14),
            bg="#4CAF50",
            fg="white",
            width=15,
            height=2
        )
        upload_button.grid(row=1, column=0, columnspan=2, pady=20)

        # 添加说明文字
        info_label = tk.Label(
            root,
            text="点击按钮选择图片，系统将自动生成图像描述",
            font=("Microsoft YaHei", 10),
            fg="gray"
        )
        info_label.grid(row=2, column=0, columnspan=2, pady=10)

        # 运行Tkinter主循环，保持窗口打开
        print("图形界面已启动，可以开始使用了！")
        root.mainloop()

        # # 加载训练好的模型权重
        # model.load_state_dict(torch.load('output1/epoch_21_meteor_0.5504_rougel_0.5044.pth', map_location=device))
        # model.eval()

        # # 定义结果存储列表
        # results = []

        # # 加载测试集的真实描述
        # with open('data/test_captions.json', 'r') as f:
        #     test_captions_json = json.load(f)

        # # 遍历 images 文件夹中的所有图片
        # images_dir = 'data/images'
        # for image_name in os.listdir(images_dir):
        #     if image_name.endswith('.jpg') or image_name.endswith('.png'):  # 仅处理图片文件
        #         image_path = os.path.join(images_dir, image_name)
            
        #         # 生成描述
        #         generated_caption = model.caption(image_path)
            
        #         # 获取真实描述
        #         true_caption = test_captions_json.get(image_name, "No description available")
            
        #         # 如果真实描述不是 "No description available"，则添加到结果列表中
        #         if true_caption != "No description available":
        #             result = {
        #                 "image": image_name,
        #                 "generated_caption": generated_caption,
        #                 "true_caption": true_caption
        #             }
        #             results.append(result)

        # # 将结果写入 JSON 文件
        # output_file = 'generated_captions_all.json'
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=4)

        # print(f"所有图片的描述已生成并保存到 {output_file}")

    else:
        print("Invalid mode. Please choose 1 for training or 2 for generating.")