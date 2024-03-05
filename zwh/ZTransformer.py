import torch
from torch import nn
from ZMultiheadAttention import MultiheadAttention  # 来自上一次写的attension


class FFN(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 feedforward_channels=1024,
                 act_cfg='ReLU',
                 ffn_drop=0.,
                 ):
        super(FFN, self).__init__()
        self.l1 = nn.Linear(in_features=embed_dim, out_features=feedforward_channels)
        if act_cfg == 'ReLU':
            self.act1 = nn.ReLU(inplace=True)
        else:
            self.act1 = nn.SiLU(inplace=True)
        self.d1 = nn.Dropout(p=ffn_drop)
        self.l2 = nn.Linear(in_features=feedforward_channels, out_features=embed_dim)
        self.d2 = nn.Dropout(p=ffn_drop)

    def forward(self, x):
        tmp = self.d1(self.act1(self.l1(x)))
        tmp = self.d2(self.l2(tmp))
        x = tmp + x
        return x


# transfomer encode和decode的最小循环单元,用于打包self_attention或者cross_attention
class BaseTransformerLayer(nn.Module):
    def __init__(self,
                 attn_cfgs=[dict(embed_dim=64, num_heads=4), dict(embed_dim=64, num_heads=4)],
                 fnn_cfg=dict(embed_dim=64, feedforward_channels=128, act_cfg='ReLU', ffn_drop=0.),
                 operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')):
        super(BaseTransformerLayer, self).__init__()
        self.attentions = nn.ModuleList()
        # 搭建att层
        for attn_cfg in attn_cfgs:
            self.attentions.append(MultiheadAttention(**attn_cfg))
        self.embed_dims = self.attentions[0].embed_dim

        # 统计norm数量 并搭建
        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(normalized_shape=self.embed_dims))

        # 统计ffn数量 并搭建
        self.ffns = nn.ModuleList()
        self.ffns.append(FFN(**fnn_cfg))
        self.operation_order = operation_order

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None):
        attn_index = 0
        norm_index = 0
        ffn_index = 0
        for order in self.operation_order:
            if order == 'self_attn':
                temp_key = temp_value = query  # 不用担心三个值一样,在attention里面会重映射qkv
                query, attention = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    query_pos=query_pos,
                    key_pos=query_pos)
                attn_index += 1
            elif order == 'cross_attn':
                query, attention = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    key_pos=key_pos)
                attn_index += 1
            elif order == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif order == 'ffn':
                query = self.ffns[ffn_index](query)
                ffn_index += 1
        return query


if __name__ == '__main__':
    query = torch.rand(size=(10, 2, 64))
    key = torch.rand(size=(5, 2, 64))
    value = torch.rand(size=(5, 2, 64))
    query_pos = torch.rand(size=(10, 2, 64))
    key_pos = torch.rand(size=(5, 2, 64))
    # encoder 通常是6个encoder_layer组成 每个encoder_layer['self_attn', 'norm', 'ffn', 'norm']
    encoder_layer = BaseTransformerLayer(attn_cfgs=[dict(embed_dim=64, num_heads=4)],
                                         fnn_cfg=dict(embed_dim=64, feedforward_channels=1024, act_cfg='ReLU',
                                                      ffn_drop=0.),
                                         operation_order=('self_attn', 'norm', 'ffn', 'norm'))

    encoder_layer_output = encoder_layer(query=query, query_pos=query_pos, key_pos=key_pos)

    # decoder 通常是6个decoder_layer组成 每个decoder_layer['self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm']
    decoder_layer = BaseTransformerLayer(attn_cfgs=[dict(embed_dim=64, num_heads=4), dict(embed_dim=64, num_heads=4)],
                                         fnn_cfg=dict(embed_dim=64, feedforward_channels=1024, act_cfg='ReLU',
                                                      ffn_drop=0.),
                                         operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))

    decoder_layer_output = decoder_layer(query=query, key=key, value=value, query_pos=query_pos, key_pos=key_pos)

    pass
