## PointNet
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

