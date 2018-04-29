import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups=3):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x

class ShuffleUnit_2b(nn.Module):
    def __init__(self, inp, out):
        super(ShuffleUnit_2b, self).__init__()
        
        self.out = out
        self.mid_out = self.out // 2       
        
        self.group_conv1 = nn.Sequential(nn.Conv2d(inp, self.mid_out, kernel_size=1, stride=1, groups=3), nn.BatchNorm2d(self.mid_out))
        
        
        self.depthwise_conv = nn.Sequential(nn.Conv2d(self.mid_out, self.mid_out, kernel_size=3, stride=1, groups=3), nn.BatchNorm2d(self.mid_out))

        self.group_conv2 = nn.Sequential(nn.Conv2d(self.mid_out, self.out, kernel_size=1, stride=1, groups=3, padding=1), nn.BatchNorm2d(self.out), nn.ReLU(inplace = True))
        
    def forward(self, x):
        original = x
        x = self.group_conv1(x)
        x = channel_shuffle(x)
        x = self.depthwise_conv(x)
        x = self.group_conv2(x)
        x = original + x
        x = F.relu(x)
        return x

class ShuffleUnit_2c(nn.Module):
    def __init__(self, inp, out, groups=3):
        super(ShuffleUnit_2c, self).__init__()
        
        self.out = out
        self.mid_out = self.out // 2
        self.out -= inp
        
        
        self.group_conv1 = nn.Sequential(nn.Conv2d(inp, self.mid_out, kernel_size=1, stride=1, groups=groups), nn.BatchNorm2d(self.mid_out))
        
        
        self.depthwise_conv = nn.Sequential(nn.Conv2d(self.mid_out, self.mid_out, kernel_size=3, stride=2, groups=3), nn.BatchNorm2d(self.mid_out))

        self.group_conv2 = nn.Sequential(nn.Conv2d(self.mid_out, self.out, kernel_size=1, stride=1, groups=3), nn.BatchNorm2d(self.out), nn.ReLU(inplace = True))
        
    def forward(self, x):
        original = x
        original = F.avg_pool2d(original, kernel_size=3, stride=2)
        x = self.group_conv1(x)
        x = channel_shuffle(x)
        x = self.depthwise_conv(x)
        x = self.group_conv2(x)
        x = torch.cat((original, x), 1)
        x = F.relu(x)
        return x

class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = nn.Sequential(
                        ShuffleUnit_2c(24, 240, groups=1),
                        ShuffleUnit_2b(240,240),
                        ShuffleUnit_2b(240,240),
                        ShuffleUnit_2b(240,240))
                
        self.stage3 = nn.Sequential(
                        ShuffleUnit_2c(240, 480),
                        ShuffleUnit_2b(480, 480),
                        ShuffleUnit_2b(480, 480),
                        ShuffleUnit_2b(480, 480),
                        ShuffleUnit_2b(480, 480),
                        ShuffleUnit_2b(480, 480),
                        ShuffleUnit_2b(480, 480),
                        ShuffleUnit_2b(480, 480))
        
        self.stage4 = nn.Sequential(
                        ShuffleUnit_2c(480, 960),
                        ShuffleUnit_2b(960, 960),
                        ShuffleUnit_2b(960, 960),
                        ShuffleUnit_2b(960, 960))
        
        self.fc = nn.Linear(960, 9)
     
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.avg_pool2d(x, [7,7])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

 
