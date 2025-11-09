import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class PreprocessedTranslationDataset(Dataset):
    """预处理的翻译数据集类"""
    def __init__(self, data_file):
        # 加载预处理的数据
        self.data = torch.load(data_file)
        self.de_sequences = self.data['de_sequences']
        self.en_sequences = self.data['en_sequences']
    
    def __len__(self):
        return len(self.de_sequences)
    
    def __getitem__(self, idx):
        return {
            'de_sequences': self.de_sequences[idx],
            'en_sequences': self.en_sequences[idx]
        }

class EmptyDataset(Dataset):
    """空数据集类，用于处理验证集或测试集为空的情况"""
    def __init__(self):
        pass
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise IndexError("索引超出范围，数据集为空")

class SimpleVocab:
    """简化的词汇表类，用于加载预处理的词汇表数据"""
    def __init__(self, vocab_data):
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_size = vocab_data['vocab_size']
        self.pad_token = vocab_data['pad_token']
        self.unk_token = vocab_data['unk_token']
        self.bos_token = vocab_data['bos_token']
        self.eos_token = vocab_data['eos_token']

def load_vocab(vocab_file):
    """加载词汇表"""
    with open(vocab_file, 'rb') as f:
        vocab_data = pickle.load(f)
    return SimpleVocab(vocab_data)

def create_preprocessed_data_loaders(data_dir, batch_size=32):
    """创建预处理数据的加载器"""
    # 加载词汇表
    de_vocab = load_vocab(os.path.join(data_dir, 'de_vocab.pkl'))
    en_vocab = load_vocab(os.path.join(data_dir, 'en_vocab.pkl'))
    
    # 创建训练数据集
    train_dataset = PreprocessedTranslationDataset(os.path.join(data_dir, 'train_data.pt'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 尝试创建验证数据集
    dev_file = os.path.join(data_dir, 'dev_data.pt')
    if os.path.exists(dev_file):
        dev_dataset = PreprocessedTranslationDataset(dev_file)
        if len(dev_dataset) > 0:
            dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        else:
            dev_loader = DataLoader(EmptyDataset(), batch_size=batch_size, shuffle=False)
            print("警告: 验证集为空")
    else:
        dev_loader = DataLoader(EmptyDataset(), batch_size=batch_size, shuffle=False)
        print("警告: 验证集文件不存在")
    
    # 尝试创建测试数据集
    test_file = os.path.join(data_dir, 'test_data.pt')
    if os.path.exists(test_file):
        test_dataset = PreprocessedTranslationDataset(test_file)
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = DataLoader(EmptyDataset(), batch_size=batch_size, shuffle=False)
            print("警告: 测试集为空")
    else:
        test_loader = DataLoader(EmptyDataset(), batch_size=batch_size, shuffle=False)
        print("警告: 测试集文件不存在")
    
    return train_loader, dev_loader, test_loader, de_vocab, en_vocab