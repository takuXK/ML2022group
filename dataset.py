import os
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PointDataset(Dataset):
    def __init__(self, filePath):

        fileList = os.listdir(filePath)

        cloudPointDataList = []
        cloudLabelDataList = []

        for fileName in tqdm.tqdm(fileList):
            filePathName = os.path.join(filePath, fileName)
            cloudPointData = np.loadtxt(filePathName)[:, 0:3]
            cloudLabelData = np.loadtxt(filePathName)[:, -1].astype('int')

            cloudPointDataList.append(torch.Tensor(cloudPointData))
            cloudLabelDataList.append(torch.Tensor(cloudLabelData))

        clouds = torch.cat(cloudPointDataList)
        labels = torch.cat(cloudLabelDataList)

        self.x_data = clouds
        self.y_data = labels.long().squeeze()

        self.length = clouds.size(0)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

def GetDataLoader(filePath, shuffle=True, batch_size=64):
    point_data_set=PointDataset(filePath)
    data_loader=DataLoader(dataset=point_data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader


if __name__ == "__main__":
    filepath= "F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\RepeatPointNet\\testpath\\"
    dataSet = PointDataset(filePath=filepath)
    dataLoader = GetDataLoader(filePath=filepath)