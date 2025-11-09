import sys
import os
# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

from src.data.preprocessed_dataset import create_preprocessed_data_loaders, load_vocab
from src.config import Config

def test_preprocessed_data():
    """测试预处理数据的内容"""
    config = Config()
    
    try:
        # 加载词汇表
        de_vocab = load_vocab(os.path.join(config.preprocessed_data_dir, 'de_vocab.pkl'))
        en_vocab = load_vocab(os.path.join(config.preprocessed_data_dir, 'en_vocab.pkl'))
        
        print(f"德语词汇表大小: {de_vocab.vocab_size}")
        print(f"英语词汇表大小: {en_vocab.vocab_size}")
        
        # 打印一些词汇表内容
        print("\n德语词汇表示例:")
        for i, word in enumerate(list(de_vocab.word2idx.keys())[:10]):
            print(f"  {word}: {de_vocab.word2idx[word]}")
        
        print("\n英语词汇表示例:")
        for i, word in enumerate(list(en_vocab.word2idx.keys())[:10]):
            print(f"  {word}: {en_vocab.word2idx[word]}")
        
        # 创建数据加载器
        train_loader, dev_loader, test_loader, _, _ = create_preprocessed_data_loaders(
            config.preprocessed_data_dir, 
            config.batch_size
        )
        
        print(f"\n训练集批次数量: {len(train_loader)}")
        print(f"验证集批次数量: {len(dev_loader)}")
        print(f"测试集批次数量: {len(test_loader)}")
        
        # 获取一个批次的数据
        batch = next(iter(train_loader))
        print(f"\n德语序列形状: {batch['de_sequences'].shape}")
        print(f"英语序列形状: {batch['en_sequences'].shape}")
        
        # 打印一些序列示例
        print("\n德语序列示例:")
        de_seq = batch['de_sequences'][0].tolist()
        print(f"  索引: {de_seq[:20]}...")
        print(f"  文本: {' '.join([de_vocab.idx2word.get(idx, '<unk>') for idx in de_seq if idx != de_vocab.word2idx[de_vocab.pad_token]])}")
        
        print("\n英语序列示例:")
        en_seq = batch['en_sequences'][0].tolist()
        print(f"  索引: {en_seq[:20]}...")
        print(f"  文本: {' '.join([en_vocab.idx2word.get(idx, '<unk>') for idx in en_seq if idx != en_vocab.word2idx[en_vocab.pad_token]])}")
        
        print("\n预处理数据测试成功!")
        return True
    
    except Exception as e:
        print(f"预处理数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_preprocessed_data()