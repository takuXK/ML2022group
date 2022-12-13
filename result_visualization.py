import torch
import numpy as np
import tqdm
from data_visualization import CloudPointVisualization
from dataset import GetDataLoader

modelName = 'model'
model = torch.load(modelName + ".pkl")

filePath = 'F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\RepeatPointNet\\visualpath\\'
pointDataLoader = GetDataLoader(filePath=filePath, shuffle=False, batch_size=64)

prediction = []
cloud_data = []
for cloud, label in tqdm.tqdm(pointDataLoader):
    cloud1, label = cloud.reshape(64, 1, 3).cuda(), label.cuda()
    out = model(cloud1)
    _, pre = torch.max(out, 1)
    cloud_data.append(torch.Tensor.cpu(cloud).numpy())
    prediction.append(torch.Tensor.cpu(pre).numpy().transpose())


cloudPoint = cloud_data[0]
cloudPred = prediction[0]
for i in range(1, len(cloud_data)):
    cloudPoint = np.concatenate((cloudPoint, cloud_data[i]), axis=0)
    cloudPred = np.concatenate((cloudPred, prediction[i]))

CloudPointVisualization(cloudPoint, cloudPred, 'plane')