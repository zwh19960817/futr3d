import cv2
import numpy as np
import matplotlib.pyplot as plt


def view_pr_curve(recall_array, precision_array, title='', save_dir='data/tmp/', dpi=800, is_show=False):
    assert len(recall_array) == len(precision_array)
    plt.figure(dpi=300)
    plt.plot(recall_array, precision_array, '-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title('Precision Recall curve for [{}]'.format(title))
    if save_dir != None:
        plt.savefig(save_dir + 'Precision Recall curve for [{}]'.format(title) + '.png', dpi=dpi)
    if is_show:
        plt.show()


def view(pred_boxes, gt_boxes, image_idx=None, save_dir='data/tmp/', dpi=800, is_show=False):
    # 创建一个米色的空白图像
    # 米色的RGB值
    BEIGE_COLOR = (245, 245, 220)
    image = np.full((1000, 1000, 3), BEIGE_COLOR, dtype=np.uint8)
    if len(pred_boxes) == 0:
        gt_boxes_and_pred_boxes = [gt_boxes, []]
    elif len(gt_boxes) == 0:
        gt_boxes_and_pred_boxes = [[], pred_boxes[:, :-1]]
    else:
        gt_boxes_and_pred_boxes = [gt_boxes, pred_boxes[:, :-1]]
    for i, boxes in enumerate(gt_boxes_and_pred_boxes):
        # 遍历数据列表并绘制边界框
        for item in boxes:
            cls, x, y, z, l, w, h, r = item
            x = float(x) * 10 + 500
            y = float(y) * 10 + 500
            z = float(z)
            l = float(l) * 5
            w = float(w) * 5
            h = float(h)
            r = float(r)
            # 计算边界框的左上角和右下角坐标
            x1, y1 = x - l / 2, y - w / 2
            x2, y2 = x + l / 2, y + w / 2

            # 计算边界框的中心点
            center_x, center_y = float(x + l / 2), float(y + w / 2)

            # 计算文本放置的位置（确保文本不会与边界框重叠）
            text_x = x1 if r == 0 else x1 + 10 * (np.sin(-r))
            text_y = y1 - 10 if r == 0 else y1 + 10 * (np.cos(r))

            # 使用OpenCV绘制旋转的矩形（边界框）
            angle = -r * 180 / np.pi  # 将弧度转换为角度
            box_points = cv2.boxPoints(((x, y), (l, w), angle))
            box_points = np.int0(box_points)

            # 绘制边界框
            cv2.drawContours(image, [box_points], -1, (i * 255, 50 * int(cls), (1-i%2) * 100), 2 - i)  # 粗的是真值

            # 添加类别文本（确保文本在图像内）
            # if 0 <= text_x < image.shape[1] and 0 <= text_y < image.shape[0]:
            #     plt.text(text_x, text_y, cls, color='black', bbox=dict(facecolor='white', alpha=0.5))

    if image_idx != None:
        from PIL import Image
        image = Image.fromarray(image)
        image.save(save_dir + str(image_idx) + '.png')
    # 显示图像
    if is_show:
        plt.figure(dpi=300)  # 设置DPI为300
        plt.imshow(image)
        # if image_idx != None:
        #     plt.savefig(out_dir + str(image_idx) + '.png', dpi=dpi)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
