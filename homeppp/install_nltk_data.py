"""
NLTK数据下载脚本
运行此脚本自动下载项目所需的NLTK数据
"""
import nltk
import sys

def download_nltk_data():
    """下载必要的NLTK数据"""
    print("开始下载NLTK数据...")
    
    data_list = ['wordnet', 'punkt', 'omw-1.4']
    
    for data_name in data_list:
        try:
            print(f"正在下载 {data_name}...")
            nltk.download(data_name, quiet=False)
            print(f"✓ {data_name} 下载完成")
        except Exception as e:
            print(f"✗ {data_name} 下载失败: {e}")
            sys.exit(1)
    
    print("\n所有NLTK数据下载完成！")

if __name__ == "__main__":
    download_nltk_data()

