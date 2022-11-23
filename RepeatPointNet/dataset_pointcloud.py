import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
import os
from data_predeal import MultiCloudPointRandomSample


class DatasetPointCloud(Dataset):
    def __init__(self, rootDir, fileDir, originPointData=None, originlabel=None, loadFromFile=True,
                 # predeal parameters
                 allFile=True, numFile=1000, numSample=1024, normalization=True):
        self.loadFromFile = loadFromFile
        self.allFile = allFile
        self.numFile = numFile
        self.numSample = numSample
        self.normalization = normalization

        if self.loadFromFile:
            # rootDir是数据存储的根目录（每个类数据对应文件夹的上一级）
            # fileDir是单类数据存储目录
            self.rootDir = rootDir
            self.fileDir = fileDir

            path = os.path.join(self.rootDir, self.fileDir)
            # self.txtFileList = np.array([filename.path for filename in os.scandir(path) if
            #                               filename.name.endswith(".txt")])
            dealedData = MultiCloudPointRandomSample(dataPath=path, allFile=self.allFile,
                                                     numFile=self.numFile, numSample=self.numSample,
                                                     normalization=normalization, savefile=False)
            self.pointData = [dealedData[i]['point'] for i in range(len(dealedData))]
            self.label = [dealedData[i]['label'] for i in range(len(dealedData))]
        else:
            self.pointData = [originPointData[i] for i in range(len(originPointData))]
            self.label = [originlabel[i] for i in range(len(originlabel))]

    def __getitem__(self, index):
        pointData = self.pointData[index]
        label = self.label[index]
        pointData = np.reshape(pointData, (1, pointData.shape[0], pointData.shape[1]))

        return pointData, label

    def __len__(self):
        return len(self.pointData)

if __name__ == "__main__":
    rootPath = 'F:\\ZJU课程\\机器学习（胡浩基）\\FinalAssignment\\data\\shapenetcore_partanno_segmentation_benchmark_v0_normal\\'
    filePath = "02691156\\"
    dataset = DatasetPointCloud(rootDir=rootPath, fileDir=filePath, allFile=False, numFile=10)
    for pointData, label in dataset:
        print('data shape:{} \nlabel shape:{}'.format(pointData.shape, label.shape))
        break
    batch_size = 64
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)