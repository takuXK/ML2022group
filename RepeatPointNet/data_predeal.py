import numpy as np
import os
import random
import tqdm
from data_visualization import CloudPointVisualization


def CloudPointRandomSample(originPointData, originLabel, numSample=1024, normalization=True):
    numPoints = len(originLabel)

    if numPoints < numSample:
        raise ValueError("the number of points is less than the number you want to sample.")
    elif numPoints == numSample:
        print("the number of points is equals to the number you want to sample and the original data will output")
        samplePointData = originPointData
        sampleLabel = originLabel
    else:
        # 随机采样index
        sampleIndex = random.sample(list(range(numPoints)), numSample)
        # 将index映射到点云数据样本上
        samplePointData = np.array([originPointData[i] for i in sampleIndex])
        sampleLabel = np.array([originLabel[i] for i in sampleIndex])

    finalData = {"point": samplePointData, "label": sampleLabel}

    # 当需要归一化数据时
    if normalization:
        normPointData = samplePointData - np.mean(samplePointData, axis=0)
        normPointData = normPointData / np.max(np.linalg.norm(normPointData, axis=1))

        finalData = {"point": normPointData, "label": sampleLabel}

    return finalData

def MultiCloudPointRandomSample(dataPath, allFile=True, numFile=1024, numSample=1024,
                                normalization=True, savefile=False, savefilePath=None):
    finalDataList = []

    if allFile:
        fileList = os.listdir(dataPath)
    else:
        if numFile < len(os.listdir(dataPath)):
            fileList = os.listdir(dataPath)[:numFile]
        else:
            print("the number of file is less than the number in the data file and all data will be predealed.")
            fileList = os.listdir(dataPath)
    print("File sampling...")
    for fileName in tqdm.tqdm(fileList):
        filePathName = os.path.join(dataPath, fileName)
        cloudPointData = np.loadtxt(filePathName)[:, 0:3]
        cloudLabelData = np.loadtxt(filePathName)[:, -1].astype('int')

        dealedData = CloudPointRandomSample(cloudPointData, cloudLabelData, numSample, normalization)
        if savefile:
            saveLocation = os.path.join(savefilePath, fileName)
            savecontext = np.c_[dealedData['point'], dealedData['label']]
            np.savetxt(saveLocation, savecontext, delimiter=' ', fmt='%.6f')

        finalDataList.append(dealedData)

    return np.array(finalDataList)

if __name__ == "__main__":
    dataPath = 'F:\\ZJU课程\\机器学习（胡浩基）\\FinalAssignment\\data\\shapenetcore_partanno_segmentation_benchmark_v0_normal\\02691156\\'
    fileList = os.listdir(dataPath)
    fileName = os.path.join(dataPath, fileList[0])
    point_cloud = np.loadtxt(fileName)[:, 0:3]
    label = np.loadtxt(fileName)[:, -1].astype('int')
    # compare the point before sample & after sample
    CloudPointVisualization(point_cloud, label, 'plane1')
    resultData = CloudPointRandomSample(point_cloud, label, 1024)
    CloudPointVisualization(resultData["point"], resultData["label"], "plane2")
    # test MultiCloudPointRandomSample
    savefilePath = 'F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\RepeatPointNet\\testpath\\'
    resultData2 = MultiCloudPointRandomSample(dataPath, False, 10, savefile=True, savefilePath=savefilePath)