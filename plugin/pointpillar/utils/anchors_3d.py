import numpy as np
import torch
from .geometry import limit_period, iou2d_nearest


class Anchors3d:
    def __init__(self, levels_range, levels_cz, levels_size, rotations):
        super(Anchors3d, self).__init__()
        ranges = []  # [x_min,y_min,z_c,x_max,y_max,z_c]
        for level_cz in levels_cz:
            ranges.append([levels_range[0], levels_range[1], level_cz, levels_range[3], levels_range[4], level_cz])

        assert len(ranges) == len(levels_size)
        self.ranges = ranges
        self.sizes = levels_size
        self.rotations = rotations

    def get_anchors(self, feature_map_size, anchor_range, anchor_size, rotations):
        '''
        feature_map_size: (x_l, y_l)
        anchor_range: [x1, y1, z1, x2, y2, z2]
        anchor_size: [l, w, h]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 2, 7)  y方向第y_l个 x方向第x_l个 某朝向 的anchor有7个信息 (中心xyz，长宽高，朝向)
        '''
        device = feature_map_size.device
        # grid的边角位置   [n,m] -> [n,n+1,n+2,,,m]
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_map_size[0] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_map_size[1] + 1, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], 1 + 1, device=device)
        # grid的边长的一半
        x_shift = (x_centers[1] - x_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        z_shift = (z_centers[1] - z_centers[0]) / 2
        # grid的中心位置
        x_centers = x_centers[:feature_map_size[0]] + x_shift  # (feature_map_size[1], )
        y_centers = y_centers[:feature_map_size[1]] + y_shift  # (feature_map_size[0], )
        z_centers = z_centers[:1] + z_shift  # (1, )

        # [feature_map_size[1], feature_map_size[0], 1, 2] * 4
        meshgrids = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        meshgrids = list(meshgrids)
        for i in range(len(meshgrids)):
            meshgrids[i] = meshgrids[i][..., None]  # [feature_map_size[1], feature_map_size[0], 1, 2, 1] 尾部再加一维

        anchor_size = anchor_size[None, None, None, None, :]
        repeat_shape = [feature_map_size[0], feature_map_size[1], 1, len(rotations), 1]
        anchor_size = anchor_size.repeat(repeat_shape)  # [feature_map_size[1], feature_map_size[0], 1, 2, 3]
        meshgrids.insert(3, anchor_size)  # 横坐标216单位，纵坐标248单位，z方向1个单位，朝向1个单位，尺寸3个单位
        anchors = torch.cat(meshgrids, dim=-1).permute(2, 0, 1, 3,
                                                       4).contiguous()  # [1, feature_map_size[0], feature_map_size[1], 2, 7]
        return anchors.squeeze(0)

    def get_multi_anchors(self, feature_map_size):
        '''
        feature_map_size: (y_l, x_l)
        ranges: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2]]
        sizes: [[w, l, h], [w, l, h], [w, l, h]]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 3, 2, 7)
        '''
        device = feature_map_size.device
        ranges = torch.tensor(self.ranges, device=device)
        sizes = torch.tensor(self.sizes, device=device)
        rotations = torch.tensor(self.rotations, device=device)
        multi_anchors = []
        for i in range(len(ranges)):
            # 不同的范围内，生成不同尺度的anchor  [x:248,y:216,z/scale(不同高度存放不同尺度，本代码1和2高度相同):3,yaw:2,info:7]
            anchors = self.get_anchors(feature_map_size=feature_map_size,
                                       anchor_range=ranges[i],
                                       anchor_size=sizes[i],
                                       rotations=rotations)
            multi_anchors.append(anchors[:, :, None, :, :])
        multi_anchors = torch.cat(multi_anchors, dim=2)

        return multi_anchors


def anchors2bboxes(anchors, deltas):
    '''
    anchors: (M, 7),  (x, y, z, l, w, h, theta)
    deltas: (M, 7)
    return: (M, 7)
    '''
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x = deltas[:, 0] * da + anchors[:, 0]
    y = deltas[:, 1] * da + anchors[:, 1]
    # z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2
    z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2]

    l = anchors[:, 3] * torch.exp(deltas[:, 3])
    w = anchors[:, 4] * torch.exp(deltas[:, 4])
    h = anchors[:, 5] * torch.exp(deltas[:, 5])

    # z = z - h / 2

    theta = anchors[:, 6] + deltas[:, 6]

    bboxes = torch.stack([x, y, z, l, w, h, theta], dim=1)
    return bboxes


def bboxes2deltas(bboxes, anchors):
    '''
    bboxes: (M, 7), (x, y, z, l, w, h, theta)
    anchors: (M, 7)
    return: (M, 7)
    '''
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    dx = (bboxes[:, 0] - anchors[:, 0]) / da
    dy = (bboxes[:, 1] - anchors[:, 1]) / da

    # zb = bboxes[:, 2] + bboxes[:, 5] / 2  # bottom center
    # za = anchors[:, 2] + anchors[:, 5] / 2 # bottom center
    dz = (bboxes[:, 2] - anchors[:, 2]) / anchors[:, 5]  # bottom center

    dl = torch.log(bboxes[:, 3] / anchors[:, 3])
    dw = torch.log(bboxes[:, 4] / anchors[:, 4])
    dh = torch.log(bboxes[:, 5] / anchors[:, 5])
    dtheta = bboxes[:, 6] - anchors[:, 6]

    deltas = torch.stack([dx, dy, dz, dl, dw, dh, dtheta], dim=1)
    return deltas


def anchor_target(batched_anchors, batched_gt_bboxes, batched_gt_labels, assigners, nclasses):
    '''
    batched_anchors: [(y_l, x_l, 3, 2, 7), (y_l, x_l, 3, 2, 7), ... ]
    batched_gt_bboxes: [(n1, 7), (n2, 7), ...]
    batched_gt_labels: [(n1, ), (n2, ), ...]
    return: 
           dict = {batched_anchors_labels: (bs, n_anchors),
                   batched_labels_weights: (bs, n_anchors),
                   batched_anchors_reg: (bs, n_anchors, 7),
                   batched_reg_weights: (bs, n_anchors),
                   batched_anchors_dir: (bs, n_anchors),
                   batched_dir_weights: (bs, n_anchors)}
    '''
    assert len(batched_anchors) == len(batched_gt_bboxes) == len(batched_gt_labels)
    batch_size = len(batched_anchors)
    n_assigners = len(assigners)
    batched_labels, batched_label_weights = [], []
    batched_bbox_reg, batched_bbox_reg_weights = [], []
    batched_dir_labels, batched_dir_labels_weights = [], []
    for i in range(batch_size):
        anchors = batched_anchors[i]
        gt_bboxes, gt_labels = batched_gt_bboxes[i], batched_gt_labels[i]
        # what we want to get next ?
        # 1. identify positive anchors and negative anchors  -> cls
        # 2. identify the regresstion values  -> reg
        # 3. indentify the direction  -> dir_cls
        multi_labels, multi_label_weights = [], []
        multi_bbox_reg, multi_bbox_reg_weights = [], []
        multi_dir_labels, multi_dir_labels_weights = [], []
        d1, d2, d3, d4, d5 = anchors.size()
        for j in range(n_assigners):  # multi anchors
            assigner = assigners[j]
            pos_iou_thr, neg_iou_thr, min_iou_thr = \
                assigner['pos_iou_thr'], assigner['neg_iou_thr'], assigner['min_iou_thr']
            cur_anchors = anchors[:, :, j, :, :].reshape(-1, 7)  # 取第j个anchor

            assigned_gt_inds = -torch.ones_like(cur_anchors[:, 0], dtype=torch.long)  # anchor 样本默认-1 就是非正且非负的样本

            if len(gt_bboxes)==0:
                assigned_gt_inds = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)  # anchor 样本默认-1 就是非正且非负的样本
            else:
                overlaps = iou2d_nearest(gt_bboxes, cur_anchors) # n个真值和m个anchor  -> n*m的iou矩阵

                max_overlaps, max_overlaps_idx = torch.max(overlaps, dim=0)  # anchor和哪个预测目标iou最大
                gt_max_overlaps, _ = torch.max(overlaps, dim=1)  # 预测目标和哪个anchor iou最大

                # a. negative anchors -- smaller IOU with label
                assigned_gt_inds[max_overlaps < neg_iou_thr] = 0

                # b. positive anchors
                # rule 1:如果anchor与所有的gt_bboxes中的最大iou大于pos_iou_thr(0.5), 那么此anchor为正anchor, /
                # 且此anchor负责与其有最大iou的gt_bbox (类别, bbox框, 旋转角度的类别) /
                # 加1,是因为0是负anchor
                assigned_gt_inds[max_overlaps >= pos_iou_thr] = max_overlaps_idx[max_overlaps >= pos_iou_thr] + 1
                # rule 2: 对每一个gt_bbox, 选择与其有最大iou的anchor, 如果其iou大于min_iou_thr(0.35), /
                # 那么此anchor为正anchor, 且此anchor对该gt_bbox负责(类别, bbox框, 旋转角度的类别)
                # support one bbox to multi anchors, only if the anchors are with the highest iou.
                # rule2 may modify the labels generated by rule 1
                for i in range(len(gt_bboxes)):
                    if gt_max_overlaps[i] >= min_iou_thr:
                        assigned_gt_inds[overlaps[i] == gt_max_overlaps[i]] = i + 1

            # 正样本 和 负样本
            pos_flag = assigned_gt_inds > 0
            neg_flag = assigned_gt_inds == 0
            # 1. anchor labels
            assigned_gt_labels = torch.zeros_like(cur_anchors[:, 0],
                                                  dtype=torch.long) + nclasses  # [m,]anchor 对应的样本label,默认为nclass,即负样本
            assigned_gt_labels[pos_flag] = gt_labels[assigned_gt_inds[pos_flag] - 1].long()
            assigned_gt_labels_weights = torch.zeros_like(cur_anchors[:, 0])#标记正负样本的label权重
            assigned_gt_labels_weights[pos_flag] = 1
            assigned_gt_labels_weights[neg_flag] = 1

            # 2. anchor regression
            assigned_gt_reg_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_reg_weights[pos_flag] = 1#标记正样本的box回归权重

            assigned_gt_reg = torch.zeros_like(cur_anchors)
            positive_anchors = cur_anchors[pos_flag]
            corr_gt_bboxes = (gt_bboxes[assigned_gt_inds[pos_flag] - 1]).to(torch.float32)# 正样本所对应的真值
            assigned_gt_reg[pos_flag] = bboxes2deltas(corr_gt_bboxes, positive_anchors)# 正样本和真值的差

            # 3. anchor direction
            assigned_gt_dir_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_dir_weights[pos_flag] = 1

            assigned_gt_dir = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)
            dir_cls_targets = limit_period(corr_gt_bboxes[:, 6].cpu(), 0, 2 * np.pi).to(corr_gt_bboxes)
            dir_cls_targets = torch.floor(dir_cls_targets / np.pi).long()
            assigned_gt_dir[pos_flag] = torch.clamp(dir_cls_targets, min=0, max=1)
            # [l,w,1,2]个anchor的 label,正负样本 box差值,正样本 航向角度差,正样本
            multi_labels.append(assigned_gt_labels.reshape(d1, d2, 1, d4))
            multi_label_weights.append(assigned_gt_labels_weights.reshape(d1, d2, 1, d4))
            multi_bbox_reg.append(assigned_gt_reg.reshape(d1, d2, 1, d4, -1))
            multi_bbox_reg_weights.append(assigned_gt_reg_weights.reshape(d1, d2, 1, d4))
            multi_dir_labels.append(assigned_gt_dir.reshape(d1, d2, 1, d4))
            multi_dir_labels_weights.append(assigned_gt_dir_weights.reshape(d1, d2, 1, d4))

        multi_labels = torch.cat(multi_labels, dim=-2).reshape(-1)
        multi_label_weights = torch.cat(multi_label_weights, dim=-2).reshape(-1)
        multi_bbox_reg = torch.cat(multi_bbox_reg, dim=-3).reshape(-1, d5)
        multi_bbox_reg_weights = torch.cat(multi_bbox_reg_weights, dim=-2).reshape(-1)
        multi_dir_labels = torch.cat(multi_dir_labels, dim=-2).reshape(-1)
        multi_dir_labels_weights = torch.cat(multi_dir_labels_weights, dim=-2).reshape(-1)

        batched_labels.append(multi_labels)
        batched_label_weights.append(multi_label_weights)
        batched_bbox_reg.append(multi_bbox_reg)
        batched_bbox_reg_weights.append(multi_bbox_reg_weights)
        batched_dir_labels.append(multi_dir_labels)
        batched_dir_labels_weights.append(multi_dir_labels_weights)

    rt_dict = dict(
        batched_labels=torch.stack(batched_labels, 0),  # (bs, y_l * x_l * 3 * 2)
        batched_label_weights=torch.stack(batched_label_weights, 0),  # (bs, y_l * x_l * 3 * 2)
        batched_bbox_reg=torch.stack(batched_bbox_reg, 0),  # (bs, y_l * x_l * 3 * 2, 7)
        batched_bbox_reg_weights=torch.stack(batched_bbox_reg_weights, 0),  # (bs, y_l * x_l * 3 * 2)
        batched_dir_labels=torch.stack(batched_dir_labels, 0),  # (bs, y_l * x_l * 3 * 2)
        batched_dir_labels_weights=torch.stack(batched_dir_labels_weights, 0)  # (bs, y_l * x_l * 3 * 2)
    )

    return rt_dict
