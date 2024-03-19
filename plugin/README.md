# FUTR3d环境参考:
https://zhuanlan.zhihu.com/p/671956617
#数据集
##nuscenes pipeline
### 新增pipeline
NormalizeGround 用于调整z方向的高度,使得地面点云高度为0.

nuscenes:offset_z = 1.86
###注意事项 LoadPointsFromFile   LoadPointsFromMultiSweeps   ObjectSample 的三者的点维度选择:
load_dim固定取5(与bin/pcd格式有关)
use_dim[0:x,1:y,2:z,3:intensity,4:time]:LoadPointsFromFile可以随便选,
LoadPointsFromMultiSweeps至少选择[0,1,2,4],ObjectSample需要配合前面两个,
所以建议直接全部选择(见配置文件).如果需要只选某几个维度,用另外pipeline处理.

#可视化
##数据集可视化
重写了数据集的可视化,使用方法:
```
plugin/tools/browse_dataset.py 
plugin/pointpillar/configs/pillar_ori.py
--task det
--output-dir /home/xxx/tmp
--online
--aug
--dataset_fun train
```
--online 用于可视化  
--aug 数据增强,如果aug使用train的pipeline,不选择就使用val的pipeline    
--dataset_fun 选择数据集(train/val/test)

#模型
##注意事项
hand调用Voxelization时候注意别用成了mmcv的,否则输出的维度顺序对不上