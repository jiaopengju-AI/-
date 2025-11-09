import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from .vocab import Vocab
import random

class TranslationDataset(Dataset):
    """翻译数据集类"""
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=128, data_augmentation=False, word_dropout_rate=0.1, word_shuffle_dist=3):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        self.data_augmentation = data_augmentation
        self.word_dropout_rate = word_dropout_rate
        self.word_shuffle_dist = word_shuffle_dist
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # 应用数据增强
        if self.data_augmentation:
            src_sentence = self._word_dropout(src_sentence)
            src_sentence = self._word_shuffle(src_sentence)
            tgt_sentence = self._word_dropout(tgt_sentence)
            tgt_sentence = self._word_shuffle(tgt_sentence)
        
        # 编码源语言和目标语言句子
        src_indices = self.src_vocab.encode(src_sentence)
        tgt_indices = self.tgt_vocab.encode(tgt_sentence)
        
        # 添加开始和结束标记
        tgt_indices = [self.tgt_vocab.word2idx[self.tgt_vocab.bos_token]] + tgt_indices + [self.tgt_vocab.word2idx[self.tgt_vocab.eos_token]]
        
        # 截断或填充
        src_indices = self._pad_or_truncate(src_indices, self.src_vocab)
        tgt_indices = self._pad_or_truncate(tgt_indices, self.tgt_vocab)
        
        return {
            'de_sequences': torch.tensor(src_indices, dtype=torch.long),
            'en_sequences': torch.tensor(tgt_indices, dtype=torch.long)
        }
    
    def _word_dropout(self, sentence):
        """词丢弃数据增强"""
        words = sentence.split()
        if len(words) <= 1:
            return sentence
            
        # 随机丢弃一些词
        new_words = []
        for word in words:
            if random.random() > self.word_dropout_rate:
                new_words.append(word)
        
        # 确保至少保留一个词
        if not new_words:
            new_words = [random.choice(words)]
            
        return ' '.join(new_words)
    
    def _word_shuffle(self, sentence):
        """词洗牌数据增强"""
        words = sentence.split()
        if len(words) <= 1:
            return sentence
            
        # 随机交换相邻词
        for _ in range(len(words) // 2):
            if random.random() < 0.5:  # 50%的概率进行交换
                i = random.randint(0, len(words) - 1)
                j = min(i + random.randint(1, self.word_shuffle_dist), len(words) - 1)
                words[i], words[j] = words[j], words[i]
                
        return ' '.join(words)
    
    def _pad_or_truncate(self, indices, vocab):
        """填充或截断序列"""
        if len(indices) > self.max_length:
            return indices[:self.max_length]
        else:
            return indices + [vocab.word2idx[vocab.pad_token]] * (self.max_length - len(indices))

def parse_xml_file(file_path):
    """解析XML格式的数据文件"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    sentences = []
    for doc in root.findall('doc'):
        for seg in doc.findall('seg'):
            sentences.append(seg.text.strip())
    
    return sentences

def load_data(data_dir, max_samples=None):
    """加载数据"""
    # 加载德语和英语训练句子
    with open(os.path.join(data_dir, 'train.tags.de-en.de'), 'r', encoding='utf-8') as f:
        de_sentences = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(data_dir, 'train.tags.de-en.en'), 'r', encoding='utf-8') as f:
        en_sentences = [line.strip() for line in f.readlines()]
    
    # 限制数据量以进行快速测试
    if max_samples is not None:
        de_sentences = de_sentences[:max_samples]
        en_sentences = en_sentences[:max_samples]
    
    # 尝试加载验证集（XML格式）
    de_dev, en_dev = [], []
    dev_file_de = os.path.join(data_dir, 'IWSLT17.TED.dev2010.de-en.de.xml')
    dev_file_en = os.path.join(data_dir, 'IWSLT17.TED.dev2010.de-en.en.xml')
    
    if os.path.exists(dev_file_de) and os.path.exists(dev_file_en):
        de_dev = parse_xml_file(dev_file_de)
        en_dev = parse_xml_file(dev_file_en)
        
        # 限制验证集数据量
        if max_samples is not None:
            de_dev = de_dev[:max(10, max_samples//10)]  # 确保至少有10个样本
            en_dev = en_dev[:max(10, max_samples//10)]
    else:
        print(f"警告: 验证集文件不存在: {dev_file_de} 或 {dev_file_en}")
        # 如果验证集文件不存在，从训练集中分割一部分作为验证集
        if max_samples is not None:
            val_size = max(10, max_samples//10)
        else:
            val_size = max(10, len(de_sentences)//10)
        de_dev = de_sentences[-val_size:]
        en_dev = en_sentences[-val_size:]
        de_sentences = de_sentences[:-val_size]
        en_sentences = en_sentences[:-val_size]
    
    # 尝试加载测试集（XML格式）
    de_test, en_test = [], []
    test_file_de = os.path.join(data_dir, 'IWSLT17.TED.tst2010.de-en.de.xml')
    test_file_en = os.path.join(data_dir, 'IWSLT17.TED.tst2010.de-en.en.xml')
    
    if os.path.exists(test_file_de) and os.path.exists(test_file_en):
        de_test = parse_xml_file(test_file_de)
        en_test = parse_xml_file(test_file_en)
        
        # 限制测试集数据量
        if max_samples is not None:
            de_test = de_test[:max(10, max_samples//10)]  # 确保至少有10个样本
            en_test = en_test[:max(10, max_samples//10)]
    else:
        print(f"警告: 测试集文件不存在: {test_file_de} 或 {test_file_en}")
        # 如果测试集文件不存在，从训练集中分割一部分作为测试集
        if max_samples is not None:
            test_size = max(10, max_samples//10)
        else:
            test_size = max(10, len(de_sentences)//10)
        de_test = de_sentences[-test_size:]
        en_test = en_sentences[-test_size:]
        de_sentences = de_sentences[:-test_size]
        en_sentences = en_sentences[:-test_size]
    
    return (de_sentences, en_sentences), (de_dev, en_dev), (de_test, en_test)

def create_data_loaders(data_dir, batch_size=32, max_length=128, max_samples=None, data_augmentation=False, word_dropout_rate=0.1, word_shuffle_dist=3):
    """创建数据加载器"""
    # 加载数据
    (de_train, en_train), (de_dev, en_dev), (de_test, en_test) = load_data(data_dir, max_samples)
    
    # 打印实际数据量
    print(f"实际训练数据量: {len(de_train)}")
    print(f"实际验证数据量: {len(de_dev)}")
    print(f"实际测试数据量: {len(de_test)}")
    
    # 构建词汇表
    de_vocab = Vocab(min_freq=2)
    de_vocab.build_vocab(de_train)
    
    en_vocab = Vocab(min_freq=2)
    en_vocab.build_vocab(en_train)
    
    # 创建数据集
    train_dataset = TranslationDataset(de_train, en_train, de_vocab, en_vocab, max_length, data_augmentation, word_dropout_rate, word_shuffle_dist)
    dev_dataset = TranslationDataset(de_dev, en_dev, de_vocab, en_vocab, max_length, False, 0, 0)  # 验证集不使用数据增强
    test_dataset = TranslationDataset(de_test, en_test, de_vocab, en_vocab, max_length, False, 0, 0)  # 测试集不使用数据增强
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader, de_vocab, en_vocab