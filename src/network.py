import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, downsample=False, dim_up=False):
        super().__init__()
        self.downsample = downsample
        self.dim_up = dim_up
        self.first_conv_stride = 2 if self.downsample else 1

        self.conv1 = nn.Conv2d(in_channel, mid_channel, 1, padding=0, stride=self.first_conv_stride)
        self.conv1_bn = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1, stride=1)
        self.conv2_bn = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, mid_channel * 4, 1, padding=0, stride=1)
        self.conv3_bn = nn.BatchNorm2d(mid_channel * 4)

        if self.dim_up:
            self.dim_up_conv = nn.Conv2d(in_channel, mid_channel * 4, 1, padding=0, stride=self.first_conv_stride)
            self.dim_up_bn = nn.BatchNorm2d(mid_channel * 4)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.conv3_bn(self.conv3(x))

        if self.dim_up:
            x = x + self.dim_up_bn(self.dim_up_conv(residual))
        else:
            x = x + residual

        return F.relu(x)


class Net(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, 7, padding=1, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.res1 = BottleneckBlock(64, 64, dim_up=True)
        self.res2 = BottleneckBlock(256, 64)
        self.res3 = BottleneckBlock(256, 64)

        self.res4 = BottleneckBlock(256, 128, downsample=True, dim_up=True)
        self.res5 = BottleneckBlock(512, 128)
        self.res6 = BottleneckBlock(512, 128)
        self.res7 = BottleneckBlock(512, 128)

        self.res8 = BottleneckBlock(512, 256, downsample=True, dim_up=True)
        self.res9 = BottleneckBlock(1024, 256)
        self.res10 = BottleneckBlock(1024, 256)
        self.res11 = BottleneckBlock(1024, 256)
        self.res12 = BottleneckBlock(1024, 256)
        self.res13 = BottleneckBlock(1024, 256)

        self.res14 = BottleneckBlock(1024, 512, downsample=True, dim_up=True)
        self.res15 = BottleneckBlock(2048, 512)
        self.res16 = BottleneckBlock(2048, 512)

        self.fc = nn.Linear(3 * 3 * 2048, 2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BottleneckBlock):
                m.initialize_weights()
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)
        x = F.avg_pool2d(x, 3, stride=1, padding=1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
