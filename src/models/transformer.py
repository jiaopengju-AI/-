import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import EncoderLayer, DecoderLayer
from .modules import PositionalEncoding, RelativePositionalEncoding, MultiHeadAttention, SparseAttention

class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout, use_relative_pos=False, use_sparse=False, window_size=8):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # 位置编码
        if use_relative_pos:
            self.pos_encoding = RelativePositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_sparse, window_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None, return_attention=False):
        # 词嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        if isinstance(self.pos_encoding, RelativePositionalEncoding):
            # 相对位置编码在注意力层内部处理
            x = self.dropout(x)
        else:
            # 绝对位置编码
            x = self.pos_encoding(x)
            x = self.dropout(x)
        
        # 存储注意力权重
        attention_weights = []
        
        # 通过各编码器层
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, mask)
        
        # 添加最终层归一化
        x = self.layer_norm(x)
        
        if return_attention:
            return x, attention_weights
        else:
            return x

class Decoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout, use_relative_pos=False, use_sparse=False, window_size=8):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # 位置编码
        if use_relative_pos:
            self.pos_encoding = RelativePositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_sparse, window_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None, return_attention=False):
        # 词嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        if isinstance(self.pos_encoding, RelativePositionalEncoding):
            # 相对位置编码在注意力层内部处理
            x = self.dropout(x)
        else:
            # 绝对位置编码
            x = self.pos_encoding(x)
            x = self.dropout(x)
        
        # 存储注意力权重
        attention_weights = []
        
        # 通过各解码器层
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, memory, src_mask, tgt_mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, memory, src_mask, tgt_mask)
        
        # 添加最终层归一化
        x = self.layer_norm(x)
        
        if return_attention:
            return x, attention_weights
        else:
            return x

class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_len=5000, dropout=0.1, use_relative_pos=False, 
                 use_sparse=False, window_size=8):
        super().__init__()
        
        # 编码器
        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, num_encoder_layers, d_ff, 
            max_seq_len, dropout, use_relative_pos, use_sparse, window_size
        )
        
        # 解码器
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, num_decoder_layers, d_ff, 
            max_seq_len, dropout, use_relative_pos, use_sparse, window_size
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, src, tgt):
        """生成注意力掩码"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, tgt_len, 1]
        
        # 创建后续掩码，防止当前位置关注到后面的位置
        subsequent_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        tgt_mask = tgt_mask & subsequent_mask.to(tgt.device)
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, return_attention=False):
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器
        if return_attention:
            memory, enc_attention = self.encoder(src, src_mask, return_attention=True)
        else:
            memory = self.encoder(src, src_mask)
        
        # 解码器
        if return_attention:
            output, dec_attention = self.decoder(tgt, memory, src_mask, tgt_mask, return_attention=True)
            attention_weights = enc_attention + dec_attention
        else:
            output = self.decoder(tgt, memory, src_mask, tgt_mask)
            attention_weights = []
        
        # 输出投影
        output = self.output_projection(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output
    
    def encode(self, src, src_mask=None, return_attention=False):
        """编码输入序列"""
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        return self.encoder(src, src_mask, return_attention)
    
    def decode(self, tgt, memory, tgt_mask=None, return_attention=False):
        """解码目标序列"""
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            subsequent_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
            tgt_mask = tgt_mask & subsequent_mask.to(tgt.device)
        
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, return_attention=return_attention)
        if return_attention:
            return self.output_projection(output[0]), output[1]
        else:
            return self.output_projection(output)
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol, end_symbol, repetition_penalty=1.2, min_length=4):
        """改进的贪心解码生成序列，添加重复惩罚"""
        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        
        # 记录已生成的词
        generated_words = set()
        
        for i in range(max_len-1):
            out = self.decode(ys, memory)
            prob = out[:, -1]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0 and i >= min_length:
                for word_id in generated_words:
                    prob[0, word_id] /= repetition_penalty
            
            # 获取最高概率的词
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            # 记录已生成的词
            generated_words.add(next_word)
            
            ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type_as(src.data)], dim=1)
            
            if next_word == end_symbol:
                break
                
        return ys
    
    def beam_search_decode(self, src, src_mask, max_len, start_symbol, end_symbol, beam_size=5, length_penalty=0.6, repetition_penalty=1.2, min_length=4):
        """改进的束搜索解码生成序列，添加重复惩罚"""
        memory = self.encode(src, src_mask)
        
        # 初始化束
        beams = [([start_symbol], 0.0, set())]  # (序列, 分数, 已生成的词)
        
        for _ in range(max_len - 1):
            new_beams = []
            
            for seq, score, gen_words in beams:
                if seq[-1] == end_symbol:
                    new_beams.append((seq, score, gen_words))
                    continue
                
                # 解码当前序列
                ys = torch.tensor([seq], dtype=torch.long).to(src.device)
                out = self.decode(ys, memory)
                prob = F.log_softmax(out[:, -1], dim=-1)
                
                # 应用重复惩罚
                if repetition_penalty != 1.0 and len(seq) >= min_length:
                    for word_id in gen_words:
                        prob[0, word_id] /= repetition_penalty
                
                # 获取top-k候选词
                topk_probs, topk_indices = torch.topk(prob, beam_size)
                
                for i in range(beam_size):
                    next_word = topk_indices[0, i].item()
                    next_prob = topk_probs[0, i].item()
                    new_seq = seq + [next_word]
                    new_score = score + next_prob
                    new_gen_words = gen_words.copy()
                    new_gen_words.add(next_word)
                    
                    # 应用长度惩罚
                    new_score = new_score / ((len(new_seq) + 5) ** length_penalty / (6 ** length_penalty))
                    
                    new_beams.append((new_seq, new_score, new_gen_words))
            
            # 选择top-k束
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
        
        # 返回最佳序列
        return torch.tensor([beams[0][0]], dtype=torch.long).to(src.device)