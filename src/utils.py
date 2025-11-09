import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """改进的标签平滑损失函数，更符合经典实现"""
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        # 对模型输出应用log_softmax，因为kl_div函数期望输入是log概率
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        
        # 计算KL散度，并正确归一化
        # 使用mean而不是sum，避免梯度爆炸
        loss = torch.nn.functional.kl_div(log_probs, true_dist, reduction='none')
        # 忽略padding位置的损失
        mask = target != self.padding_idx
        loss = loss.sum(dim=-1) * mask.float()
        return loss.sum() / mask.sum().clamp(min=1)

def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, optimizer, epoch, loss, path):
    """保存模型"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"模型已保存到 {path}")

def load_model(model, optimizer, path):
    """加载模型"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"模型已从 {path} 加载，epoch: {epoch}, loss: {loss:.4f}")
        return epoch, loss
    else:
        print(f"未找到模型文件: {path}")
        return 0, float('inf')

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存到 {save_path}")

class WarmupScheduler:
    """学习率预热调度器"""
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0, decay_rate=0.9, decay_steps=10000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.decay_rate = decay_rate  # 学习率衰减率
        self.decay_steps = decay_steps  # 衰减步数
        self.step_num = 0
    
    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """获取当前学习率"""
        step = self.step_num
        d_model = self.d_model
        warmup_steps = self.warmup_steps
        factor = self.factor
        
        # 基础学习率
        lr = factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        
        # 添加学习率衰减
        if step > warmup_steps:
            decay_factor = self.decay_rate ** ((step - warmup_steps) // self.decay_steps)
            lr *= decay_factor
        
        return lr
    
    def state_dict(self):
        """返回调度器的状态字典"""
        return {
            'step_num': self.step_num,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            'decay_rate': self.decay_rate,
            'decay_steps': self.decay_steps
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器的状态字典"""
        self.step_num = state_dict['step_num']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']
        self.decay_rate = state_dict['decay_rate']
        self.decay_steps = state_dict['decay_steps']

def evaluate_bleu(model, test_loader, tgt_vocab, device, use_beam_search=False, beam_size=5):
    """评估BLEU分数，使用改进的解码策略"""
    from nltk.translate.bleu_score import corpus_bleu
    
    references = []
    hypotheses = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src = batch['de_sequences'].to(device)
            tgt = batch['en_sequences'].to(device)
            
            # 生成预测
            batch_size = src.size(0)
            start_symbol = tgt_vocab.word2idx[tgt_vocab.bos_token]
            end_symbol = tgt_vocab.word2idx[tgt_vocab.eos_token]
            
            for i in range(batch_size):
                # 使用改进的解码策略
                if use_beam_search:
                    output = model.beam_search_decode(
                        src[i:i+1], 
                        None, 
                        max_len=50, 
                        start_symbol=start_symbol, 
                        end_symbol=end_symbol,
                        beam_size=beam_size,
                        repetition_penalty=1.2,
                        min_length=4
                    )
                else:
                    output = model.greedy_decode(
                        src[i:i+1], 
                        None, 
                        max_len=50, 
                        start_symbol=start_symbol, 
                        end_symbol=end_symbol,
                        repetition_penalty=1.2,
                        min_length=4
                    )
                
                # 将索引转换为单词
                tgt_tokens = tgt[i].tolist()
                ref_tokens = [tgt_vocab.idx2word[idx] for idx in tgt_tokens 
                             if idx != tgt_vocab.word2idx[tgt_vocab.pad_token] 
                             and idx != tgt_vocab.word2idx[tgt_vocab.bos_token]
                             and idx != tgt_vocab.word2idx[tgt_vocab.eos_token]]
                
                hyp_tokens = [tgt_vocab.idx2word[idx] for idx in output[0].tolist() 
                             if idx != tgt_vocab.word2idx[tgt_vocab.pad_token] 
                             and idx != tgt_vocab.word2idx[tgt_vocab.bos_token]
                             and idx != tgt_vocab.word2idx[tgt_vocab.eos_token]]
                
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)
    
    # 计算BLEU分数
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

def ablation_study_results(results, save_path):
    """绘制消融实验结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制不同配置的验证损失
    for config_name, losses in results.items():
        plt.plot(losses, label=config_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Ablation Study Results')
    plt.legend()
    plt.grid(True)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"消融实验结果已保存到 {save_path}")