import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def CloudPointVisualization(pointData, label, title=None):
    labelData = pd.DataFrame(
        data={
            "x": pointData[:, 0],
            "y": pointData[:, 1],
            "z": pointData[:, 2],
            "label": label,
        }
    )
    # 单纯绘制点云图像而不管其在物体中所属的部分
    fig1 = plt.figure(figsize=(15, 10))
    ax1 = plt.axes(projection="3d")
    ax1.scatter(labelData["x"], labelData["y"], labelData["z"])
    if title:
        plt.title(title + " with no classification")
    plt.show()
    # 按点云图像各点所属部分分别绘制
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = plt.axes(projection="3d")
    for i in range(label.min(), label.max() + 1):
        c_df = labelData[labelData['label'] == i]
        ax2.scatter(c_df["x"], c_df["y"], c_df["z"])
    # ax.legend()
    if title:
        plt.title(title + " with classification")
    plt.show()

if __name__ == "__main__":
    filePath = 'F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\data\shapenetcore_partanno_segmentation_benchmark_v0_normal\\02691156\\1a32f10b20170883663e90eaf6b4ca52.txt'
    # 读取点云文件
    point_cloud = np.loadtxt(filePath)[:, 0:3]
    label = np.loadtxt(filePath)[:, -1].astype('int')
    print("data load completed.")
    # 绘图
    CloudPointVisualization(point_cloud, label, 'plane')
    print('point cloud shape:{}'.format(point_cloud.shape))
    print('label shape:{}'.format(label.shape))
    print("point construct completed.")