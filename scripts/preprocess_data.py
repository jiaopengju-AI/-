import sys
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

from src.data.dataset import load_data, parse_xml_file
from src.data.vocab import Vocab

def preprocess_data(data_dir, output_dir, max_length=128, max_samples=None):
    """预处理数据，生成模型可以直接使用的文件"""
    print("开始预处理数据...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    (de_train, en_train), (de_dev, en_dev), (de_test, en_test) = load_data(data_dir, max_samples)
    
    print(f"训练集大小: {len(de_train)}")
    print(f"验证集大小: {len(de_dev)}")
    print(f"测试集大小: {len(de_test)}")
    
    # 如果验证集或测试集为空，从训练集分割一部分
    if len(de_dev) == 0 and len(de_train) > 100:
        # 分割10%的训练数据作为验证集
        split_idx = int(len(de_train) * 0.9)
        de_dev = de_train[split_idx:]
        en_dev = en_train[split_idx:]
        de_train = de_train[:split_idx]
        en_train = en_train[:split_idx]
        print(f"从训练集分割验证集，新训练集大小: {len(de_train)}，新验证集大小: {len(de_dev)}")
    
    if len(de_test) == 0 and len(de_train) > 100:
        # 分割10%的训练数据作为测试集
        split_idx = int(len(de_train) * 0.9)
        de_test = de_train[split_idx:]
        en_test = en_train[split_idx:]
        de_train = de_train[:split_idx]
        en_train = en_train[:split_idx]
        print(f"从训练集分割测试集，新训练集大小: {len(de_train)}，新测试集大小: {len(de_test)}")
    
    # 如果训练集仍然很大，但验证集和测试集为空，则强制分割
    if len(de_dev) == 0 and len(de_test) == 0 and len(de_train) > 20:
        # 分割训练集，确保验证集和测试集都有数据
        split_idx1 = int(len(de_train) * 0.8)
        split_idx2 = int(len(de_train) * 0.9)
        de_dev = de_train[split_idx1:split_idx2]
        en_dev = en_train[split_idx1:split_idx2]
        de_test = de_train[split_idx2:]
        en_test = en_train[split_idx2:]
        de_train = de_train[:split_idx1]
        en_train = en_train[:split_idx1]
        print(f"强制分割训练集，新训练集大小: {len(de_train)}，新验证集大小: {len(de_dev)}，新测试集大小: {len(de_test)}")
    
    # 构建词汇表
    print("构建德语词汇表...")
    de_vocab = Vocab(min_freq=2)
    de_vocab.build_vocab(de_train)
    
    print("构建英语词汇表...")
    en_vocab = Vocab(min_freq=2)
    en_vocab.build_vocab(en_train)
    
    print(f"德语词汇表大小: {de_vocab.vocab_size}")
    print(f"英语词汇表大小: {en_vocab.vocab_size}")
    
    # 保存词汇表 - 只保存必要的字典数据
    print("保存词汇表...")
    de_vocab_data = {
        'word2idx': de_vocab.word2idx,
        'idx2word': de_vocab.idx2word,
        'vocab_size': de_vocab.vocab_size,
        'pad_token': de_vocab.pad_token,
        'unk_token': de_vocab.unk_token,
        'bos_token': de_vocab.bos_token,
        'eos_token': de_vocab.eos_token
    }
    
    en_vocab_data = {
        'word2idx': en_vocab.word2idx,
        'idx2word': en_vocab.idx2word,
        'vocab_size': en_vocab.vocab_size,
        'pad_token': en_vocab.pad_token,
        'unk_token': en_vocab.unk_token,
        'bos_token': en_vocab.bos_token,
        'eos_token': en_vocab.eos_token
    }
    
    with open(os.path.join(output_dir, 'de_vocab.pkl'), 'wb') as f:
        pickle.dump(de_vocab_data, f)
    
    with open(os.path.join(output_dir, 'en_vocab.pkl'), 'wb') as f:
        pickle.dump(en_vocab_data, f)
    
    # 处理训练数据
    print("处理训练数据...")
    train_data = process_sentences(de_train, en_train, de_vocab, en_vocab, max_length)
    torch.save(train_data, os.path.join(output_dir, 'train_data.pt'))
    
    # 处理验证数据
    print("处理验证数据...")
    dev_data = process_sentences(de_dev, en_dev, de_vocab, en_vocab, max_length)
    torch.save(dev_data, os.path.join(output_dir, 'dev_data.pt'))
    
    # 处理测试数据
    print("处理测试数据...")
    test_data = process_sentences(de_test, en_test, de_vocab, en_vocab, max_length)
    torch.save(test_data, os.path.join(output_dir, 'test_data.pt'))
    
    print("数据预处理完成!")
    return True

def process_sentences(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length):
    """处理句子对，转换为模型可以直接使用的格式"""
    data = {
        'de_sequences': [],
        'en_sequences': []
    }
    
    for src_sentence, tgt_sentence in tqdm(zip(src_sentences, tgt_sentences), total=len(src_sentences)):
        # 编码源语言和目标语言句子
        src_indices = src_vocab.encode(src_sentence)
        tgt_indices = tgt_vocab.encode(tgt_sentence)
        
        # 添加开始和结束标记
        tgt_indices = [tgt_vocab.word2idx[tgt_vocab.bos_token]] + tgt_indices + [tgt_vocab.word2idx[tgt_vocab.eos_token]]
        
        # 截断或填充
        src_indices = pad_or_truncate(src_indices, max_length, src_vocab)
        tgt_indices = pad_or_truncate(tgt_indices, max_length, tgt_vocab)
        
        data['de_sequences'].append(src_indices)
        data['en_sequences'].append(tgt_indices)
    
    # 转换为张量
    data['de_sequences'] = torch.tensor(data['de_sequences'], dtype=torch.long)
    data['en_sequences'] = torch.tensor(data['en_sequences'], dtype=torch.long)
    
    return data

def pad_or_truncate(indices, max_length, vocab):
    """填充或截断序列"""
    if len(indices) > max_length:
        return indices[:max_length]
    else:
        return indices + [vocab.word2idx[vocab.pad_token]] * (max_length - len(indices))

if __name__ == "__main__":
    # 设置路径
    data_dir = os.path.join('..', 'src', 'data', 'de-en')
    output_dir = os.path.join('..', 'processed_data')
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    args = parser.parse_args()
    
    # 预处理数据
    preprocess_data(data_dir, output_dir, args.max_length, args.max_samples)