import torch.nn as nn
from .modules import *

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout, use_sparse=False, window_size=8):
        super().__init__()
        if use_sparse:
            self.self_attn = SparseAttention(d_model, n_heads, window_size)
        else:
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
    
    def forward(self, x, mask=None, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.self_attn(x, x, x, mask)
            x = self.sublayer1(x, lambda x: attn_output)
            x = self.sublayer2(x, self.ffn)
            return x, attn_weights
        else:
            x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask)[0])
            return self.sublayer2(x, self.ffn)

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout, use_sparse=False, window_size=8):
        super().__init__()
        if use_sparse:
            self.self_attn = SparseAttention(d_model, n_heads, window_size)
            self.cross_attn = SparseAttention(d_model, n_heads, window_size)
        else:
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask, return_attention=False):
        if return_attention:
            # 带掩码的自注意力
            self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
            x = self.sublayer1(x, lambda x: self_attn_output)
            
            # 编码器-解码器注意力
            cross_attn_output, cross_attn_weights = self.cross_attn(x, memory, memory, src_mask)
            x = self.sublayer2(x, lambda x: cross_attn_output)
            
            x = self.sublayer3(x, self.ffn)
            
            # 返回自注意力和交叉注意力的权重
            return x, (self_attn_weights, cross_attn_weights)
        else:
            # 带掩码的自注意力
            x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
            # 编码器-解码器注意力
            x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask)[0])
            return self.sublayer3(x, self.ffn)