import os
import re
from collections import Counter

class Vocab:
    """词汇表类"""
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.word2freq = {}
        self.vocab_size = 0
        self.min_freq = min_freq
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
    
    def build_vocab(self, sentences):
        """构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for sentence in sentences:
            words = self.tokenize(sentence)
            word_counts.update(words)
        
        # 添加特殊标记
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            self.word2idx[token] = self.vocab_size
            self.idx2word[self.vocab_size] = token
            self.word2freq[token] = float('inf')
            self.vocab_size += 1
        
        # 添加满足最小频率的词
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.word2freq[word] = count
                self.vocab_size += 1
    
    def tokenize(self, sentence):
        """分词"""
        # 简单的按空格和标点分词
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        return sentence.lower().split()
    
    def encode(self, sentence):
        """将句子转换为索引序列"""
        words = self.tokenize(sentence)
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        return indices
    
    def decode(self, indices):
        """将索引序列转换为句子"""
        words = [self.idx2word[idx] for idx in indices if idx not in [self.word2idx[self.pad_token]]]
        return ' '.join(words)
    
    def __len__(self):
        return self.vocab_size