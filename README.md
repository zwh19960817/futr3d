# FUTR3D: A Unified Sensor Fusion Framework for 3D Detection
This repo implements the paper FUTR3D: A Unified Sensor Fusion Framework for 3D Detection. [Paper](https://arxiv.org/abs/2203.10642) - [project page](https://tsinghua-mars-lab.github.io/futr3d/)

We built our implementation upon MMdetection3D 1.0.0rc6. The major part of the code is in the directory `plugin/futr3d`. 

## Environment
### Prerequisite
<ol>
<li> mmcv-full>=1.5.2, <=1.7.0 </li>
<li> mmdet>=2.24.0, <=3.0.0</li>
<li> mmseg>=0.20.0, <=1.0.0</li>
<li> nuscenes-devkit</li>
</ol>

### Installation

There is no neccesary to install mmdet3d separately, please install based on this repo:

```
cd futr3d
pip3 install -v -e .
```


### Data

 Please follow the mmdet3d to process the data. https://mmdetection3d.readthedocs.io/en/stable/datasets/nuscenes_det.html

## Train

For example, to train FUTR3D with LiDAR only on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075_900q.py 8
```

For LiDAR-Cam and Cam-Radar version, we need pre-trained model. 

The Cam-Radar uses DETR3D model as pre-trained model, please check [DETR3D](https://github.com/WangYueFt/detr3d).

The LiDAR-Cam uses fused LiDAR-only and Cam-only model as pre-trained model. You can use

```
python tools/fuse_model.py --img <cam checkpoint path> --lidar <lidar checkpoint path> --out <out model path>
```
to fuse cam-only and lidar-only models.

## Evaluate

For example, to evalaute FUTR3D with LiDAR-cam on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_cam/lidar_0075_cam_res101.py ../lidar_cam.pth 8 --eval bbox
```


## Results

### LiDAR & Cam
| models      | mAP         | NDS | Link |
| ----------- | ----------- | ----| ---- |
| [Res101 + VoxelNet](./plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py)  | 67.4 | 70.9 | [model](https://drive.google.com/file/d/1hdsrQhWOD6CjgoTgyi1i3KV94IRt2OhF/view?usp=share_link)|
| [VoVNet + VoxelNet](./plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py)   | 70.3 | 73.1 | [model](https://drive.google.com/file/d/1DgrzSoZSlTT_RDNGplHUMXatboKlkCqq/view?usp=share_link) |


### Cam & Radar
| models      | mAP         | NDS | Link |
| ----------- | ----------- | ----| ----- |
| [Res101 + Radar](./plugin/futr3d/configs/cam_radar/cam_res101_radar.py)  | 39.9  | 50.8 | [model](https://drive.google.com/file/d/1LX3kflWap_qWjTNy3Zy9gL9_IXANkUo1/view?usp=share_link) |

### LiDAR only

| models      | mAP         | NDS | Link |
| ----------- | ----------- | ----|  ----|
| [32 beam VoxelNet](./plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py)  | 63.3 | 68.9 | [model](https://drive.google.com/file/d/16y3o4Gn6NmNexM9E_ye4OnTeQNKWkOCk/view?usp=share_link)|
| [4 beam VoxelNet](./plugin/futr3d/configs/lidar_only/lidar_0075v_900q_4b.py)   | 44.3 | 56.4 |
| [1 beam VoxelNet](./plugin/futr3d/configs/lidar_only/lidar_0075v_900q_1b.py)   | 16.9 | 39.2 |

### Cam only
The camera-only version of FUTR3D is the same as DETR3D. Please check [DETR3D](https://github.com/WangYueFt/detr3d) for detail implementation.

## Acknowledgment

For the implementation, we rely heavily on [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), and [DETR3D](https://github.com/WangYueFt/detr3d)


## Related projects 
1. [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://tsinghua-mars-lab.github.io/detr3d/)
2. [MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries](https://tsinghua-mars-lab.github.io/mutr3d/)
3. For more projects on Autonomous Driving, check out our Visual-Centric Autonomous Driving (VCAD) project page [webpage](https://tsinghua-mars-lab.github.io/vcad/) 


## Reference

```
@article{chen2022futr3d,
  title={FUTR3D: A Unified Sensor Fusion Framework for 3D Detection},
  author={Chen, Xuanyao and Zhang, Tianyuan and Wang, Yue and Wang, Yilun and Zhao, Hang},
  journal={arXiv preprint arXiv:2203.10642},
  year={2022}
}
```

Contact: Xuanyao Chen at: `xuanyaochen19@fudan.edu.cn` or `ixyaochen@gmail.com`
