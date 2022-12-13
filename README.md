## PointNet

#### 2022-11-23更新
- PointNet的相关论文参考文件夹中的pdf： [Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf](Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) 
- PointNet的源码在github上，pytorch版本：[yanx27/Pointnet_Pointnet2_pytorch: PointNet and PointNet++ implemented by pytorch (pure python) and on ModelNet, ShapeNet and S3DIS. (github.com)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)，可以用git clone命令copy
- [vinits5/learning3d: This is a complete package of recent deep learning methods for 3D point clouds in pytorch (with pretrained models). (github.com)](https://github.com/vinits5/learning3d)中有PointNet的pretrained模型，但我没用过
- 百度paddle框架的代码在这个网站中：[【点云分割】：基于PointNet实现点云分割 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/580974019)主要想复现这个代码
- 我自己写的代码在文件夹”RepeatPointNet“下，里面的readme文件有一些解释，目前的情况是：
  - 数据预处理的部分是可以用的：data_predeal和data_visualization
  - 虽然dataset的生成文件（dataset_pointcloud）也可以运行，但是转换为dataloader后喂到网络（pointnet和pointnet2）中会报错，网络本身也存在问题
  - train_seg是最终训练要运行的文件，由于前面有问题，也运行不起来


#### 2022-12-13更新
- 已经完成PointNet相关代码修改，待模型训练
- 训练过程如下：
  - pyfile: data_predeal.py
    - 修改main函数中的dataPath，文件路径具体到shapenetcore_partanno_segmentation_benchmark_v0_normal下的一个类，例如shapenetcore_partanno_segmentation_benchmark_v0_normal\\\02691156\\\，记得路径末尾加"\\\\"
    - 修改main函数中的savetrainPath和savevalPath，将训练集数据和测试数据分别存储到两个文件路径下（数据分类）
    - 运行即可得到分类后的数据
  - pyfile: train.py
    - 修改文件中的trainPath变量为之前savetrainPath中的路径地址，valPath变量为之前savevalPath中的路径地址
    - 修改文件中的segClass变量为当前面向的类（例如飞机类），改变量涉及到模型的保存，为防止模型被刷新或者搞不清楚导致需要重新训练浪费时间，所以每次需要改一下，不需要一定改成对应的名称，但需要自己能够区分
  - pyfile：result_visiualization.py
    - 修改filePath，从savevalPath路径下随便copy一个txt文件到filePath指向的文件路径下
    - 修改需要导入的model的名字modelName，运行即可得到训练完的模型的分类效果图
