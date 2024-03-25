import torch
from torch import Tensor
import numpy as np
from .view import view, view_pr_curve


def iou(pred_boxes, gt_boxes):
    '''
    pred_boxes (m,4)
    gt_boxes (n,4)
    '''
    assert len(pred_boxes[0]) == 4 and len(gt_boxes[0]) == 4
    pred_size = len(pred_boxes)
    gt_size = len(gt_boxes)
    pred_boxes = Tensor(pred_boxes)
    gt_boxes = Tensor(gt_boxes)
    pred_x1y1x1y2 = torch.stack((pred_boxes[:, 0] - pred_boxes[:, 2] * 0.5,
                                 pred_boxes[:, 1] - pred_boxes[:, 3] * 0.5,
                                 pred_boxes[:, 0] + pred_boxes[:, 2] * 0.5,
                                 pred_boxes[:, 1] + pred_boxes[:, 3] * 0.5), axis=1)
    gt_x1y1x1y2 = torch.stack((gt_boxes[:, 0] - gt_boxes[:, 2] * 0.5,
                               gt_boxes[:, 1] - gt_boxes[:, 3] * 0.5,
                               gt_boxes[:, 0] + gt_boxes[:, 2] * 0.5,
                               gt_boxes[:, 1] + gt_boxes[:, 3] * 0.5), axis=1)
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    area_wl = torch.min(pred_x1y1x1y2[:, None, 2:], gt_x1y1x1y2[None, :, 2:]) - \
              torch.max(pred_x1y1x1y2[:, None, :2], gt_x1y1x1y2[None, :, :2])
    area = area_wl[:, :, 0].clamp(min=0) * area_wl[:, :, 1].clamp(min=0)
    iou = area / (pred_area[:, None] + gt_area[None,] - area)
    return iou.numpy()


def compute_ap(recall_array, precision_array, title, save_dir='data/tmp/'):
    precision_array = np.concatenate(([1.], precision_array, [0.]))
    recall_array = np.concatenate(([0.], recall_array, [1.]))
    view_pr_curve(recall_array, precision_array, title + '-ori', save_dir=save_dir, dpi=300, is_show=False)
    # 修正曲线
    # make precision_array monotone decreasion
    for i in range(len(precision_array) - 1, 0, -1):
        precision_array[i - 1] = np.maximum(precision_array[i - 1], precision_array[i])
    recall_array[-1] = recall_array[-2]
    view_pr_curve(recall_array, precision_array, title, save_dir=save_dir, dpi=300, is_show=False)
    ap = np.sum((recall_array[1:] - recall_array[:-1]) *
                (precision_array[:-1] + precision_array[1:]) * 0.5)
    return ap


class CYWEvaluation():
    def __init__(self, iou_thresholds: list = [0.5], classes: list = ['car', 'pedestrian']):
        self.iou_thresholds = iou_thresholds
        self.classes = classes

    def eval(self, pred_boxes, gt_boxes, sample_size=0, save_dir=None, save_detail=False):
        if save_dir != None:
            if sample_size != 0:
                for i in range(sample_size):
                    view(pred_boxes=pred_boxes[pred_boxes[:, 0] == i][:, 1:],
                         gt_boxes=gt_boxes[gt_boxes[:, 0] == i][:, 1:],
                         save_dir=save_dir,
                         image_idx=str(i))
        results = {}
        for iou_threshold in self.iou_thresholds:
            iou_threshold = round(iou_threshold, 3)
            ap_and_gts = {}
            sample_idx_unique = np.unique(gt_boxes[:, 0])  # 取所有的image_idx集合, 为什么不取index?因为有时image_idx不是序号,而是一些编码
            for class_idx, class_name in enumerate(self.classes):
                # 取出当前类别的所以boxes
                pred_boxes_cls = pred_boxes[pred_boxes[:, 1] == class_idx]
                gt_boxes_cls = gt_boxes[gt_boxes[:, 1] == class_idx]
                tp_and_tn = len(gt_boxes_cls)  # tp + tn 即真值数量
                if tp_and_tn == 0:  # if no gt_boxes, the average precious is 0
                    ap_and_gts[class_name] = 0, 0
                    continue
                # pred 根据iou区分的tp和fp,后面需要用score确定哪些pred是有效的
                # 初始化全为0,即误检
                fp_or_tp = []
                # collect tp and fp from pred_boxes in all samples
                for sample_idx in sample_idx_unique:
                    # 取出当前类别中当前图像的预测和真值
                    pred_boxes_cls_img = pred_boxes_cls[pred_boxes_cls[:, 0] == sample_idx]
                    gt_boxes_cls_img = gt_boxes_cls[gt_boxes_cls[:, 0] == sample_idx]
                    if len(pred_boxes_cls_img) == 0:  # if no pred, over
                        continue
                    elif len(gt_boxes_cls_img) == 0:  # if no gt ,preds belong to false
                        for _ in range(len(pred_boxes_cls_img)):
                            fp_or_tp.append(0)
                        continue
                    over_laps = iou(pred_boxes=pred_boxes_cls_img[:, [2, 3, 5, 6]],
                                    gt_boxes=gt_boxes_cls_img[:, [2, 3, 5, 6]])
                    corr_gts = np.argmax(over_laps, axis=1)
                    corr_iou = np.max(over_laps, axis=1)
                    visited_gt = []
                    if save_dir is not None and save_detail:
                        pred_boxes_view = []
                    for id, pred_box in enumerate(pred_boxes_cls_img):
                        if corr_iou[id] >= iou_threshold and corr_gts[id] not in visited_gt:
                            visited_gt.append(corr_gts[id])  # if pred got gt, the gt should be ignore
                            fp_or_tp.append(1)
                            if save_dir is not None and save_detail:
                                pred_boxes_view.append(pred_box)
                        else:
                            fp_or_tp.append(0)
                    if save_dir is not None and save_detail:
                        if len(pred_boxes_view) == 0:
                            pred_boxes_view = []
                        else:
                            pred_boxes_view = np.array(pred_boxes_view)[:, 1:]
                        view(pred_boxes=pred_boxes_view, gt_boxes=gt_boxes_cls_img[:, 1:],
                             image_idx='iou<' + str(iou_threshold) + '>-sample<' + str(
                                 int(sample_idx)) + '>-cls<' + str(
                                 class_idx) + '>', save_dir=save_dir, is_show=False)
                fp_or_tp = np.array(fp_or_tp)
                scores = pred_boxes_cls[:, -1]
                index = np.argsort(-scores, )

                # sort fp_or_tp by decending order of scores
                fp_or_tp = fp_or_tp[index]

                # get list of tp_num and pred_num from ordered score list,
                # while the pred_boxes are splited to be valid or unvalid by ordered score in list
                # 意思是根据>=score的是有效pred,有效pred中,0是f,1是检测到了目标, 1的数量除以总的有效pred就是precious
                # 1的数量除以真值数量就是recall
                tp_num_list = np.cumsum(fp_or_tp)
                pred_num_list = np.cumsum(np.ones_like(fp_or_tp))

                precision_array = tp_num_list / pred_num_list
                recall_array = tp_num_list / tp_and_tn

                ap = compute_ap(recall_array, precision_array,
                                'iou<' + str(iou_threshold) + '>-cls<' + class_name + '>', save_dir=save_dir)
                ap_and_gts[class_name] = round(ap, 3), tp_and_tn

            # mAP
            aps = 0.0
            class_labeled = 0
            for ap, tp_and_tn in ap_and_gts.values():
                if tp_and_tn > 0:
                    class_labeled += 1
                    aps += ap
            mAP = aps / class_labeled if class_labeled != 0 else 0

            result = {}
            result['AP'] = ap_and_gts
            result['mAP'] = round(mAP, 3)
            results[str(iou_threshold)] = result
        return results


if __name__ == "__main__":
    classes = ['car', 'pedestrian', 'bus', 'bicyle']
    # image_idx cls_id x y z l w h yaw score(for pred_boxes)
    pred_boxes = np.array([[0, 1, 13, 13, 0, 6, 6, 2, 0, 0.9],
                           [0, 0, 35, 30, 0, 10, 8, 2, 0, 0.9],
                           [0, 0, 12, 30, 0, 6, 10, 2, 0, 0.5],
                           [1, 0, 20, 20, 0, 8, 6, 2, 0, 0.4],
                           [1, 0, 42, 40, 0, 8, 6, 2, 0, 0.5],
                           [1, 0, 41, 40, 0, 8, 6, 2, 0, 0.5],
                           [3, 2, 42, 40, 0, 8, 6, 2, 0, 0.5]])
    gt_boxes = np.array([[0, 1, 10, 10, 0, 6, 6, 2, 0],
                         [0, 0, 30, 30, 0, 10, 8, 2, 0],
                         [0, 0, 10, 30, 0, 6, 10, 2, 0],
                         [1, 0, 20, 20, 0, 10, 6, 2, 0],
                         [1, 0, 40, 40, 0, 8, 6, 2, 0],
                         [1, 0, 10, 10, 0, 8, 6, 2, 0],
                         [2, 0, 30, 30, 0, 8, 6, 2, 0],
                         [2, 0, 40, 30, 0, 8, 6, 2, 0],
                         [2, 0, 30, 40, 0, 8, 6, 2, 0],
                         [2, 3, 10, 10, 0, 8, 6, 2, 0],
                         [2, 0, 20, 30, 0, 8, 6, 2, 0]])

    for i in range(4):
        view(pred_boxes=pred_boxes[pred_boxes[:, 0] == i][:, 1:],
             gt_boxes=gt_boxes[gt_boxes[:, 0] == i][:, 1:],
             image_idx=str(i))

    iou_thresholds = np.arange(1, 20) * 0.05
    iou_thresholds = [0.5]

    cymmap = CYWEvaluation(iou_thresholds, classes)
    resluts = cymmap.eval(pred_boxes, gt_boxes)

    pass
