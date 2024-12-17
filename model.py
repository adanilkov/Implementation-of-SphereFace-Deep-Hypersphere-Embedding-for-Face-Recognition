import torch
from torch import nn
import torch.nn.functional as F

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-7, m=None):
        super(AngularPenaltySMLoss, self).__init__()

        self.m = 4. if not m else m

        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)

        numerator = torch.cos(self.m * torch.acos(
            torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)
        L = numerator - torch.log(denominator)

        return -torch.mean(L)

class SphereCNN(nn.Module):
    def __init__(self, class_num: int, feature=False):
        super(SphereCNN, self).__init__()
        self.class_num = class_num
        self.feature = feature

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc5 = nn.Linear(512 * 6 * 6, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.angular = AngularPenaltySMLoss(512, self.class_num)

    def forward(self, x, y):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn5(self.fc5(x)))

        if self.feature:
            return x
        else:
            x_angle = self.angular(x, y)
            return x, x_angle
