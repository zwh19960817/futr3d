import cv2
import os

# 1. 设置PNG图像所在的文件夹路径和视频输出路径
input_folder = '/home/zwh/Downloads/hhh'  # 替换为你的PNG图像文件夹路径
output_video = '/home/zwh/Downloads/hhh/output_video.mp4'  # 输出视频文件名

# 2. 获取PNG图像列表
images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
images.sort()  # 确保图像按正确的顺序排序

# 3. 设置视频编码器（例如，使用'mp4v'编码器）和帧率（例如，30fps）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
fps = 2.0  # 视频帧率
frame_width = 1920  # 视频帧宽度，根据你的图像大小设置
frame_height = 1080  # 视频帧高度，根据你的图像大小设置

# 4. 创建VideoWriter对象
video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 5. 遍历PNG图像并写入视频
for image_name in images:
    image_path = os.path.join(input_folder, image_name)
    frame = cv2.imread(image_path)

    # 调整图像大小以匹配视频帧大小（如果需要）
    frame = cv2.resize(frame, (frame_width, frame_height))
    cv2.waitKey(3)
    # 写入帧到视频
    video.write(frame)

# 6. 释放VideoWriter对象
video.release()
print(f"Video saved to {output_video}")