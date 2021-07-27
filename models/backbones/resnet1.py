import torch
import torch.nn as nn

class BottleBlock(nn.Module):
    extention = 4
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BottleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.extention)
        self.relu = nn.ReLU()
        self.downsample = downsample
        
    def forward(self, x):
        self.rediuals = x
        out = self.relu(self.bn1(self.conv1(x)))
        print("out1.shape", out.shape)
        out = self.relu(self.bn2(self.conv2(out)))
        print("out2.shape", out.shape)
        out = self.bn3(self.conv3(out))
        print("out3.shape", out.shape)
        if self.downsample:
            self.rediuals = self.downsample(x)
        out += self.rediuals
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.block = block
        self.layers = layers
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.make_layers(self.block, 64, self.layers[0], stride=1)
        self.stage2 = self.make_layers(self.block, 128, self.layers[1], stride=2)
        self.stage3 = self.make_layers(self.block, 256, self.layers[2], stride=2)
        self.stage4 = self.make_layers(self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        stem = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        print(stem.shape)
        stage1 = self.stage1(stem)
        print("stage1.shape", stage1.shape)
        stage2 = self.stage2(stage1)
        print("stage2.shape", stage2.shape)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stage1, stage2, stage3, stage4

    
    def make_layers(self, block, mid_channels, block_num, stride):
        block_list = []
        downsample = None
        if stride != 1 or self.inplanes != mid_channels * block.extention:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, mid_channels * block.extention, stride=1, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels * block.extention)
            )
        conv_block = block(self.inplanes, mid_channels, downsample)
        block_list.append(conv_block)
        self.inplane = mid_channels * block.extention
        for _ in range(1, block_num):
            conv_block = block(self.inplanes, mid_channels)
            block_list.append(conv_block)
        return nn.Sequential(*block_list)




resnet = ResNet(BottleBlock, [3, 4, 6, 3])
x=torch.randn(1,3,224,224)
x=resnet(x)