## PointNet的复现

#### Data

- 所使用数据集：ShapeNetPart
- data下载地址：https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

#### 函数列表

- data_visualization
  
- CloudPointVisualization：用于绘制单个点云数据样本的图像，分别输出原图（单色点）和按label绘制的图像（多色点）
  
- data_predeal

  - CloudPointRandomSample：用于**随机**采样单个点云数据样本中的数据，输出采样后的数据点；如果选择归一化数据@normalization，则对于R*C的数组A内的所有数据x：

  $$
  x_{ij}'=\frac{x_{ij}-\sum_{i=1}^{R}x_{ij}/R}{max(norm(x_{j}))}
  $$

  - MultiCloudPointRandomSample：用于对一个文件夹内多点云样本数据进行批量处理；如果选择保存数据@savefile，则将数据采样后批量保存至指定文件夹

- DatasetPointCloud：用于数据的封装，以便向模型传输数据

#### 模型

- PointNet