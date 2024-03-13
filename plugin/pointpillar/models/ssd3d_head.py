import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import anchor_target, anchors2bboxes, limit_period, Anchors3d
from ..ops import nms_cuda
from mmdet3d.models.builder import HEADS
from mmcv.runner import BaseModule


@HEADS.register_module()
class SSD3dHEAD(BaseModule):
    def __init__(self,
                 in_channel,
                 n_anchors,
                 n_classes,
                 anchor_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SSD3dHEAD, self).__init__(init_cfg=init_cfg)
        self.n_classes = n_classes
        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]
        self.anchors_generator = Anchors3d(**anchor_cfg)
        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
            ]

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return:
              bbox_cls_pred: (bs, n_anchors*3, 248, 216)
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred

    def loss(self,
             bbox_cls_pred,
             bbox_pred,
             bbox_dir_cls_pred,
             listed_gt_bboxes,
             listed_labels,
             img_metas,
             gt_bboxes_ignore=None):
        device = bbox_cls_pred.device
        batch_size = len(bbox_cls_pred)
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)  # [l/2,w/2,n,2,7]
        listed_anchors = [anchors for _ in range(batch_size)]

        listed_gt_bboxes_np = []
        for gt_box in listed_gt_bboxes:
            listed_gt_bboxes_np.append(gt_box.tensor[:, :7].to(device))

        anchor_target_dict = anchor_target(batched_anchors=listed_anchors,
                                           batched_gt_bboxes=listed_gt_bboxes_np,
                                           batched_gt_labels=listed_labels,
                                           assigners=self.assigners,
                                           nclasses=self.n_classes)

        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

        batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
        batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
        batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
        # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
        batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
        # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < self.n_classes)  # 如果这个格子被真值击中,则为正样本
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]
        # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < self.n_classes).sum()  # 正样本的数量
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels[batched_bbox_labels < 0] = self.n_classes
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

        loss_func = Loss()
        loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                              bbox_pred=bbox_pred,
                              bbox_dir_cls_pred=bbox_dir_cls_pred,
                              batched_labels=batched_bbox_labels,
                              num_cls_pos=num_cls_pos,
                              batched_bbox_reg=batched_bbox_reg,
                              batched_dir_labels=batched_dir_labels)
        return loss_dict

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216)
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return:
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, )
        '''
        # 0. pre-process
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.n_classes)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. 根据类别分数  选择前100个box  第一次过滤:选前100
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. offset + anchor = box
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1)  # (n_anchors, (x_min))

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.n_classes):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr  # 第二次分数过滤
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d,
                                 scores=cur_bbox_cls_pred,
                                 thresh=self.nms_thr,
                                 pre_maxsize=None,
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(
                cur_bbox_pred)  # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        result = {}
        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            result = {
                'lidar_bboxes': [],
                'labels': [],
                'scores': []
            }
        else:
            ret_bboxes = torch.cat(ret_bboxes, 0)
            ret_labels = torch.cat(ret_labels, 0)
            ret_scores = torch.cat(ret_scores, 0)
            if ret_bboxes.size(0) > self.max_num:
                final_inds = ret_scores.topk(self.max_num)[1]
                ret_bboxes = ret_bboxes[final_inds]
                ret_labels = ret_labels[final_inds]
                ret_scores = ret_scores[final_inds]
            result = {
                'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
                'labels': ret_labels.detach().cpu().numpy(),
                'scores': ret_scores.detach().cpu().numpy()
            }
        return result

    def predict(self, pred, gt_bboxes_ignore=None):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return:
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ]
        '''
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = pred
        device = bbox_cls_pred.device
        batch_size = len(bbox_cls_pred)
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)  # [l/2,w/2,n,2,7]
        listed_anchors = [anchors for _ in range(batch_size)]

        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i],
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                                                      anchors=listed_anchors[i])
            results.append(result)
        return results


class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1 / 9, cls_w=1.0, reg_w=2.0, dir_w=0.2):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none',
                                              beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()

    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                bbox_dir_cls_pred,
                batched_labels,
                num_cls_pos,
                batched_bbox_reg,
                batched_dir_labels):
        '''
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        '''
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float()  # (n, 3)

        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
                  (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels)  # (n, 3)
        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos

        # 2. regression loss
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)

        # 3. direction cls loss
        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)

        # 4. total loss
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss

        loss_dict = {'cls_loss': cls_loss,
                     'reg_loss': reg_loss,
                     'dir_cls_loss': dir_cls_loss}
            # ,'total_loss': total_loss}
        return loss_dict
