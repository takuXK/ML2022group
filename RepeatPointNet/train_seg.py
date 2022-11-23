import os
import torch
import torchmetrics
from dataset_pointcloud import DatasetPointCloud
from pointnet import PointNet


rootPath = 'F:\\ZJU课程\\机器学习（胡浩基）\\FinalAssignment\\data\\shapenetcore_partanno_segmentation_benchmark_v0_normal\\'
filePath = "02691156\\"
dataset = DatasetPointCloud(rootDir=rootPath, fileDir=filePath, allFile=False, numFile=10,
                            numSample=1024, normalization=True)
allPointData = dataset.pointData
allLabel = dataset.label

valSplit = 0.2
splitIndex = int(len(allLabel) * (1-valSplit))
trainOriginPoint = allPointData[:splitIndex]
trainOriginLabel = allLabel[:splitIndex]
valOriginPoint = allPointData[splitIndex:]
valOriginLabel = allLabel[splitIndex:]

print("Num train point clouds:", len(trainOriginPoint))
print("Num train point cloud labels:", len(trainOriginLabel))
print("Num val point clouds:", len(valOriginPoint))
print("Num val point cloud labels:", len(valOriginLabel))

trainDataset = DatasetPointCloud(rootDir=None, fileDir=None, loadFromFile=False,
                                 originPointData=trainOriginPoint, originlabel=trainOriginLabel)
valDataset = DatasetPointCloud(rootDir=None, fileDir=None, loadFromFile=False,
                               originPointData=valOriginPoint, originlabel=valOriginLabel)

batchsize = 64

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchsize, shuffle=True)
valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batchsize, shuffle=True)


model = PointNet()
model.train()

# 优化器、损失函数和评价指标
optim = torch.optim.Adam(params=model.parameters(), weight_decay=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy()

# 训练轮数
epoch_num = 50
# 每多少个epoch保存
save_interval = 2
# 每多少个epoch验证
val_interval = 2
best_acc = 0
# 模型保存地址
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 训练过程
plot_acc = []
plot_loss = []
for epoch in range(epoch_num):
    total_loss = 0
    for batch_id, data in enumerate(trainLoader()):
        inputs = torch.as_tensor(data[0], dtype=torch.float32)
        labels = torch.as_tensor(data[1], dtype=torch.int64)
        predicts = model(inputs)

        # 计算损失和反向传播
        loss = loss_fn(predicts, labels)
        total_loss = total_loss + loss.numpy()[0]
        loss.backward()
        # 计算acc
        predicts = torch.reshape(predicts, (predicts.shape[0] * predicts.shape[1], -1))
        labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
        accuracy.update(predicts, labels)
        # 优化器更新
        optim.step()
        optim.zero_grad()
    avg_loss = total_loss / batch_id
    plot_loss.append(avg_loss)
    print("epoch: {}/{}, loss is: {}, acc is:{}".format(epoch, epoch_num, avg_loss, accuracy.accumulate()))
    accuracy.reset()

    # 保存
    if epoch % save_interval == 0:
        model_name = str(epoch)
        torch.save(model.state_dict(), './output/PointNet_{}.pdparams'.format(model_name))
        torch.save(optim.state_dict(), './output/PointNet_{}.pdopt'.format(model_name))
    # 训练中途验证
    if epoch % val_interval == 0:
        model.eval()
        for batch_id, data in enumerate(valLoader()):
            inputs = torch.as_tensor(data[0], dtype=torch.float32)
            labels = torch.as_tensor(data[1], dtype=torch.int64)
            predicts = model(inputs)
            predicts = torch.reshape(predicts, (predicts.shape[0]*predicts.shape[1], -1))
            labels = torch.reshape(labels, (labels.shape[0]*labels.shape[1], 1))
            accuracy.update(predicts, labels)
        val_acc = accuracy.accumulate()
        plot_acc.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            print("===================================val===========================================")
            print('val best epoch in:{}, best acc:{}'.format(epoch, best_acc))
            print("===================================train===========================================")
            torch.save(model.state_dict(), './output/best_model.pdparams')
            torch.save(optim.state_dict(), './output/best_model.pdopt')
        accuracy.reset()
        model.train()