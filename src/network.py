import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 1, padding=1, stride=1)
        self.conv1_bn = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, padding=1, stride=1)
        self.conv2_bn = nn.BatchNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, in_channel * 4, 1, padding=1, stride=1)
        self.conv3_bn = nn.BatchNorm2d(in_channel * 4)

    # def forward(self, x):
    #     x = 


class Net(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, 7, padding=1, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.conv6_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(18 * 18 * 64, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 80)
        self.fc2_bn = nn.BatchNorm1d(80)
        self.fc3 = nn.Linear(80, 2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight)

    def forward(self, x):
        x_ = x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x))) + x_
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x_ = x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x))) + x_
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)

        return x
