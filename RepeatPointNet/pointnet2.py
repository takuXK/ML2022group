from torch import nn
import torch
from torchsummary import summary


class PointNet(nn.Module):
    def __init__(self, numClass=4, numPoint=1024):
        super(PointNet, self).__init__()
        self.numPoint = numPoint
        self.inputTransformNet = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool2d((numPoint, 1))
        )
        self.inputFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.featureTransformNet = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool2d((numPoint, 1))
        )
        self.featureFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.segNet = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, numClass, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batchsize = x.shape[0]

        output1 = self.inputTransformNet(x)
        output1 = torch.squeeze(output1, dim=1)
        output1 = self.inputFC(output1)
        output1 = torch.reshape(output1, [batchsize, 3, 3])

        output2 = torch.reshape(x, shape=(batchsize, 1024, 3))
        output2 = torch.matmul(output2, output1)
        output2 = torch.unsqueeze(output2, dim=1)
        output2 = self.mlp1(output2)

        output1 = self.featureTransformNet(output2)
        output1 = torch.squeeze(output1)
        output1 = self.featureFC(output1)
        output1 = torch.reshape(output1, [batchsize, 64, 64])

        output2 = torch.reshape(output2, shape=(batchsize, 64, 1024))
        output2 = torch.transpose(output2, (0, 2, 1))
        output2 = torch.matmul(output2, output1)
        output2 = torch.transpose(output2, (0, 2, 1))
        output2 = torch.unsqueeze(output2, dim=-1)
        tempout = output2
        output2 = self.mlp2(output2)
        output2 = torch.max(output2, dim=2)

        globalFeatExpand = torch.tile(torch.unsqueeze(output2, dim=1), [1, self.numPoint, 1, 1])
        output2 = torch.concat([tempout, globalFeatExpand], dim=1)
        output2 = self.segNet(output2)
        output2 = torch.squeeze(output2, dim=-1)
        output2 = torch.transpose(output2, (0, 2, 1))

        return output2

if __name__ == "__main__":
    pointnet = PointNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pointnet.to(device)
    summary(model, input_size=(1024, 3))