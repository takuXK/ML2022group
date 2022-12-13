import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self, point_num):
        super(PointNet, self).__init__()

        self.inputTransform = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d((point_num, 1)),
        )
        self.inputFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 9),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.featureTransform = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d((point_num, 1)),
        )
        self.featureFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 64),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.7,inplace=True),对于ShapeNet数据集来说,用dropout反而准确率会降低
            nn.Linear(256, 16),
            nn.Softmax(dim=1),
        )
        self.inputFC[4].weight.data = torch.zeros(3 * 3, 256)
        self.inputFC[4].bias.data = torch.eye(3).view(-1)

    def forward(self, x):  # [B, N, XYZ]
        '''
            B:batch_size
            N:point_num
            K:k_classes
            XYZ:input_features
        '''
        batch_size = x.size(0)  # batchsize大小
        x = x.unsqueeze(1)  # [B, 1, N, XYZ]

        t_net = self.inputTransform(x)  # [B, 1024, 1,1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.inputFC(t_net)  # [B, 3*3]
        t_net = t_net.view(batch_size, 3, 3)  # [B, 3, 3]

        x = x.squeeze(1)  # [B, N, XYZ]

        x = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(x, t_net)])  # [B, N, XYZ]# 因为mm只能二维矩阵之间，故逐个乘再拼起来

        x = x.unsqueeze(1)  # [B, 1, N, XYZ]

        x = self.mlp1(x)  # [B, 64, N, 1]

        t_net = self.featureTransform(x)  # [B, 1024, 1, 1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.featureFC(t_net)  # [B, 64*64]
        t_net = t_net.view(batch_size, 64, 64)  # [B, 64, 64]

        x = x.squeeze(3).permute(0, 2, 1)  # [B, N, 64]

        x = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(x, t_net)])  # [B, N, 64]

        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, N, 1]

        x = self.mlp2(x)  # [B, N, 64]

        x, _ = torch.max(x, 2)  # [B, 1024, 1]

        x = self.fc(x.squeeze(2))  # [B, K]
        return x