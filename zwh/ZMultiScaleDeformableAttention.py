import torch
import torch.nn.functional as F
import torch.nn as nn


def multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
    batch, _, num_head, embeding_dim_perhead = value.shape
    _, query_size, _, level_num, sample_num, _ = sampling_locations.shape
    split_list = []
    for h, w in spatial_shapes:
        split_list.append(int(h * w))
    value_list = value.split(split_size=tuple(split_list), dim=1)
    # [0,1)分布变成 [-1,1)分布,因为要调用F.grid_sample函数
    sampling_grid = 2 * sampling_locations - 1
    output_list = []
    for level_id, (h, w) in enumerate(spatial_shapes):
        h = int(h)
        w = int(w)
        # batch, value_len, num_head, embeding_dim_perhead
        # batch, num_head, embeding_dim_perhead, value_len
        # batch*num_head, embeding_dim_perhead, h, w
        value_l = value_list[level_id].permute(0, 2, 3, 1).view(batch * num_head, embeding_dim_perhead, h, w)
        # batch,query_size,num_head,level_num,sample_num,2
        # batch,query_size,num_head,sample_num,2
        # batch,num_head,query_size,sample_num,2
        # batch*num_head,query_size,sample_num,2
        sampling_grid_l = sampling_grid[:, :, :, level_id, :, :].permute(0, 2, 1, 3, 4).view(batch * num_head,
                                                                                             query_size, sample_num, 2)
        # batch*num_head embeding_dim,,query_size, sample_num
        output = F.grid_sample(input=value_l,
                               grid=sampling_grid_l,
                               mode='bilinear',
                               padding_mode='zeros',
                               align_corners=False)
        output_list.append(output)
    # batch*num_head, embeding_dim_perhead,query_size, level_num, sample_num
    outputs = torch.stack(output_list, dim=-2)
    # batch,query_size,num_head,level_num,sample_num
    # batch,num_head,query_size,level_num,sample_num
    # batch*num_head,1,query_size,level_num,sample_num
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).view(batch * num_head, 1, query_size, level_num,
                                                                      sample_num)
    outputs = outputs * attention_weights
    # batch*num_head, embeding_dim_perhead,query_size
    # batch,num_head, embeding_dim_perhead,query_size
    # batch,query_size,num_head, embeding_dim_perhead
    # batch,query_size,num_head*embeding_dim_perhead
    outputs = outputs.sum(-1).sum(-1).view(batch, num_head, embeding_dim_perhead, query_size).permute(0, 3, 1, 2). \
        view(batch, query_size, num_head * embeding_dim_perhead)
    return outputs.contiguous()


if __name__ == '__main__':
    batch = 1
    num_head = 8
    embeding_dim = 256
    query_size = 900
    spatial_shapes = torch.Tensor([[180, 180], [90, 90], [45, 45], [23, 23]])
    value_len = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().int()
    value = torch.rand(size=(batch, value_len, embeding_dim))
    query_embeding = torch.rand(size=(batch, query_size, embeding_dim * 2 + 3))
    query = query_embeding[..., :embeding_dim]
    query_pos = query_embeding[..., embeding_dim:2 * embeding_dim]
    reference_poins = query_embeding[..., 2 * embeding_dim:]
    # 讨论1:在deformale-att中这个query并不会和value交互生成att-weights,att-weights只和query有关,
    # 也就是推理过程att-weights(包括sampling_locations)是固定的.
    # 据作者解释这是因为采用前者的方式计算的attention权重存在退化问题,
    # 即最后得到的attention权重与并没有随key的变化而变化。
    # 因此，这两种计算attention权重的方式最终得到的结果相当，
    # 而后者耗时更短、计算代价更小，所以作者选择直接对query做projection得到attention权重。
    # 讨论2:在query固定情况下,第一个layer的att-weights无法改变,
    # 但是第二个layer的query与value有关,att-weights则会发生变化.so the self-att in frist layer is not nesscerary
    level_num = 4
    sample_num = 4
    sampling_offsets_net = nn.Linear(in_features=embeding_dim, out_features=num_head * level_num * sample_num * 2)
    sampling_offsets = sampling_offsets_net(query).view(batch, query_size, num_head, level_num, sample_num, 2)
    sampling_location = reference_poins[:, :, None, None, None, :2] + sampling_offsets
    attention_weights_net = nn.Linear(in_features=embeding_dim, out_features=num_head * level_num * sample_num)
    attention_weights = attention_weights_net(query).view(batch, query_size, num_head, level_num * sample_num)
    attention_weights = attention_weights.softmax(dim=-1).view(batch, query_size, num_head, level_num,
                                                               sample_num)  # sum of 16 points weight is equal to 1
    embeding_dim_perhead = embeding_dim // num_head
    value = value.view(batch, value_len, num_head, -1)

    output = multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, sampling_location, attention_weights)
    pass
