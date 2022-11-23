## PointNet

- PointNet的相关论文参考文件夹中的pdf： [Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf](Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) 
- PointNet的源码在github上，pytorch版本：[yanx27/Pointnet_Pointnet2_pytorch: PointNet and PointNet++ implemented by pytorch (pure python) and on ModelNet, ShapeNet and S3DIS. (github.com)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)，可以用git clone命令copy
- 百度paddle框架的代码在这个网站中：[【点云分割】：基于PointNet实现点云分割 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/580974019)

- 我自己写的代码在文件夹”RepeatPointNet“下，里面的readme文件有一些解释，目前的情况是：
  - 数据预处理的部分是可以用的：data_predeal和data_visualization
  - 虽然dataset的生成文件（dataset_pointcloud）也可以运行，但是转换为dataloader后喂到网络（pointnet和pointnet2）中会报错，网络本身也存在问题
  - train_seg是最终训练要运行的文件，由于前面有问题，也运行不起来