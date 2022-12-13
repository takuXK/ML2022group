import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import datetime
import tqdm
from dataset import GetDataLoader
import pointnet_model
from data_predeal import IOUcalculation

trainpath= "F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\RepeatPointNet\\trainpath\\"
valpath= "F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\RepeatPointNet\\valpath\\"

epoches = 100
pointnum = 1024
classnum = 4
epochAccuracy = []
epoch_mIOU = []

print("Train Data Loading...")
train_loader = GetDataLoader(filePath=trainpath, shuffle=True, batch_size=64)
print("Test Data Loading...")
test_loader = GetDataLoader(filePath=valpath, shuffle=False, batch_size=64)

net = pointnet_model.PointNet(pointnum).cuda()

optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()

for epoch in range(epoches):
    print("Running Epoch: " + str(epoch + 1))
    time_start = datetime.datetime.now()
    net.train()
    for cloud, label in tqdm.tqdm(train_loader):
        cloud, label = cloud.reshape(64, 1, 3).cuda(), label.cuda()
        out = net(cloud)
        loss = loss_function(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    totalCorrection = 0
    totalCmatrix = np.zeros((classnum, classnum))
    net.eval()
    for cloud, label in tqdm.tqdm(test_loader):
        cloud, label = cloud.reshape(64, 1, 3).cuda(), label.cuda()
        out = net(cloud)
        _, pre = torch.max(out, 1)
        correct = (pre == label).sum()
        totalCorrection += correct.item()
        CMatrix = confusion_matrix(torch.Tensor.cpu(label), torch.Tensor.cpu(pre))
        if CMatrix.shape[0] < classnum:
            CMatrix = np.pad(CMatrix, (0, classnum - CMatrix.shape[0]), 'constant', constant_values=0)
        totalCmatrix += CMatrix
    time_end = datetime.datetime.now()
    time_span_str = str((time_end - time_start).seconds)
    accuracy = totalCorrection / len(test_loader.dataset)
    print("\n" + str(epoch + 1) + " Epoch Accuracy: " + str(accuracy) + "\ntime consuming: " + time_span_str + "s")
    iou, miou = IOUcalculation(totalCmatrix)
    print("IOU: " + str(iou) + "\n")
    print("mIOU: " + str(miou))

    epochAccuracy.append(accuracy)
    epoch_mIOU.append(miou)

    if epoch == 0:
        optimized_model = net
        saveEpoch = epoch
    else:
        if accuracy > epochAccuracy[epoch - 1]:
            optimized_model = net
            saveEpoch = epoch
        elif accuracy == epochAccuracy[epoch - 1]:
            if miou > epoch_mIOU[epoch - 1]:
                optimized_model = net
                saveEpoch = epoch
            else:
                pass
        else:
            pass

segClass = "plane"
modelName = "model-" + segClass + ".pkl"
torch.save(optimized_model, modelName)
print("saved epoch: " + str(saveEpoch))
# model = torch.load('model.pkl')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(np.linspace(1, epoches, epoches), np.array(epochAccuracy), label="Accuracy", color='b')
ax2 = ax1.twinx()
ax2.plot(np.linspace(1, epoches, epoches), np.array(epoch_mIOU), label="mIOU", color='r')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax2.set_ylabel("mIOU")
figname = 'plane-' + str(pointnum) + '-' + str(epoches) + '-' + str(classnum)
# plt.savefig(figname)
plt.show()

# python的强大之处
# acc=sum([(torch.max(net(cloud.cuda()),1)[1]==label.cuda()).sum() for cloud,label in test_loader]).item()/len(test_loader.dataset)