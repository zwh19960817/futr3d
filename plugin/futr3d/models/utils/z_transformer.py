import torch
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 att_dropout=0.1,
                 out_dropout=0.1,
                 average_attn_weights=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(att_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.average_attn_weights = average_attn_weights
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        assert self.embed_dim == self.num_heads * self.head_dim, \
            'embed_dim <{}> must be divisible by num_heads <{}>'.format(self.embed_dim, self.num_heads)
        self.fuse_heads = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                identity=None,
                query_pos=None,
                key_pos=None):
        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        tgt_len, bsz, embed_dim = query.shape  # [查询数量 batch数量 特征维度]
        src_len, _, _ = key.shape  # [被查询数量,_,_]
        # 默认和query进行shortcut(要在位置编码前,因为output为输出特征,特征和原特征shortcut,下一层再重新加位置编码,否则不就重了)
        if identity is None:
            identity = query
        # 位置编码
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        # 特征划分为self.num_heads 份 [tgt,b,embed_dim] -> [b,n_h, tgt, d_h]
        # [n,b,n_h*d_h] -> [b,n_h,n,d_h] 主要是target和source之前的特征匹配和提取, batch和n_h维度不处理
        query = query.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        key = key.contiguous().view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        value = value.contiguous().view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # [b,n_h,tgt_len,src_len]
        # Scaled Dot-Product Attention
        attention = query @ key.transpose(-2, -1)
        attention /= self.scale  # 参考: https://blog.csdn.net/zwhdldz/article/details/135462127
        attention = torch.softmax(attention, dim=-1)  # 行概率矩阵
        attention = self.att_dropout(input=attention)  # 正则化方法 DropKey，用于缓解 Vision Transformer 中的过拟合问题
        # [b,n_h,tgt_len,d_h] = [b,n_h,tgt_len,src_len] * [b,n_h,src_len,d_h]
        output = attention @ value
        # [b,n_h,tgt_len,d_h] -> [b,tgt_len,embed_dim]
        output = output.permute(0, 2, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        # 头之间通过全连接融合一下
        output = self.fuse_heads(output)
        output = self.out_dropout(output)
        # shortcut
        output = output + identity
        # 多头head求平均
        if self.average_attn_weights:
            attention = attention.sum(dim=1) / self.num_heads
        # [tgt_len,b,embed_dim],[b,tgt_len,src_len]
        return output, attention


if __name__ == '__main__':
    query = torch.rand(size=(10, 2, 64))
    key = torch.rand(size=(5, 2, 64))
    value = torch.rand(size=(5, 2, 64))
    query_pos = torch.rand(size=(10, 2, 64))
    key_pos = torch.rand(size=(5, 2, 64))

    att = MultiheadAttention(64, 4)
    # 返回特征采样结果和attention矩阵
    output = att(query=query, key=key, value=value,query_pos=query_pos,key_pos=key_pos)
    pass
