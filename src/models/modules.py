import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class RelativePositionalEncoding(nn.Module):
    """相对位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # 创建相对位置嵌入
        self.relative_positions_k = nn.Parameter(torch.zeros(2 * max_len - 1, d_model))
        self.relative_positions_v = nn.Parameter(torch.zeros(2 * max_len - 1, d_model))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.relative_positions_k)
        nn.init.xavier_uniform_(self.relative_positions_v)
    
    def forward(self, q, k, v):
        """
        q, k, v: [batch_size, num_heads, seq_len, d_head]
        """
        batch_size, num_heads, seq_len, d_head = q.size()
        
        # 生成相对位置索引
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec[None, :] - range_vec[:, None]  # [seq_len, seq_len]
        distance_mat = distance_mat + self.max_len - 1  # 将[-seq_len+1, seq_len-1]映射到[0, 2*seq_len-2]
        
        # 获取相对位置嵌入
        relative_pos_k = self.relative_positions_k[distance_mat]  # [seq_len, seq_len, d_model]
        relative_pos_v = self.relative_positions_v[distance_mat]  # [seq_len, seq_len, d_model]
        
        # 调整形状以匹配多头注意力
        relative_pos_k = relative_pos_k.view(seq_len, seq_len, num_heads, d_head).permute(2, 0, 1, 3)
        relative_pos_v = relative_pos_v.view(seq_len, seq_len, num_heads, d_head).permute(2, 0, 1, 3)
        
        # 计算相对位置得分
        q_transposed = q.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, d_head]
        relative_pos_scores = torch.matmul(q_transposed, relative_pos_k.transpose(-2, -1))  # [batch_size, seq_len, seq_len, num_heads]
        relative_pos_scores = relative_pos_scores.permute(0, 3, 1, 2)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用相对位置到value
        relative_pos_v = relative_pos_v.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, num_heads, seq_len, seq_len, d_head]
        v = v.unsqueeze(3)  # [batch_size, num_heads, seq_len, 1, d_head]
        relative_pos_v = torch.matmul(v, relative_pos_v.transpose(-2, -1)).squeeze(3)  # [batch_size, num_heads, seq_len, seq_len]
        
        return relative_pos_scores, relative_pos_v

class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def forward(self, query, key, value, mask=None, relative_pos_scores=None, relative_pos_v=None):
        batch_size = query.size(0)
        
        # 保存残差连接
        residual = query
        
        # 层归一化
        query = self.layer_norm(query)
        
        # 线性变换并分割成多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, self.attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, relative_pos_scores, relative_pos_v
        )
        
        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 线性变换
        output = self.w_o(attention_output)
        
        # 残差连接
        return residual + output, self.attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None, relative_pos_scores=None, relative_pos_v=None):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加相对位置分数（如果有）
        if relative_pos_scores is not None:
            scores = scores + relative_pos_scores
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        if relative_pos_v is not None:
            output = torch.matmul(attention_weights, V) + relative_pos_v
        else:
            output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class SparseAttention(nn.Module):
    """稀疏注意力"""
    def __init__(self, d_model, n_heads, window_size=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 保存残差连接
        residual = query
        
        # 层归一化
        query = self.layer_norm(query)
        
        # 线性变换并分割成多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算稀疏注意力
        attention_output, self.attention_weights = self.sparse_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 线性变换
        output = self.w_o(attention_output)
        
        # 残差连接
        return residual + output, self.attention_weights
    
    def sparse_dot_product_attention(self, Q, K, V, mask=None):
        batch_size, n_heads, seq_len, d_k = Q.size()
        
        # 创建窗口掩码
        window_mask = self._create_window_mask(seq_len).to(Q.device)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用窗口掩码
        scores = scores.masked_fill(window_mask == 0, -1e4)
        
        # 应用额外掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def _create_window_mask(self, seq_len):
        """创建窗口掩码"""
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > self.window_size:
                    mask[i, j] = 0
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

class PositionWiseFFN(nn.Module):
    """位置前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()  # 使用GELU激活函数
    
    def forward(self, x):
        # 保存残差连接
        residual = x
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 前馈网络
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        
        # 残差连接
        return residual + x

class SublayerConnection(nn.Module):
    """残差连接和层归一化"""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))