import torch
import torch.nn as nn
import torch.nn.functional as F


def ChannelShuffle(x, groups=3):
    b, n, h, w = x.data.size()

    group_channels = n // groups
    x = x.view(b, groups, group_channels, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(b, -1, h, w)

    return x

class ShuffleUnit(nn.Module):
    def __init__(self, input_channels, output_channels, groups=3, stride=1):
        super(ShuffleUnit, self).__init__()
        
        self.stride = stride
        self.groups = groups
        self.output_channels = output_channels
        self.mid_channels = self.output_channels // 4
        
        self.padding=1
        
        if stride == 2:
            self.output_channels -= input_channels
            self.padding = 0
            
        first_group = self.groups
            
        if input_channels == 24:
            first_group = 1
            
        
        self.group_conv1 = nn.Sequential(nn.Conv2d(input_channels, 
                                                   self.mid_channels, 
                                                   kernel_size=1, 
                                                   groups=first_group), 
                                         nn.BatchNorm2d(self.mid_channels))
        
        
        self.depthwise_conv = nn.Sequential(nn.Conv2d(self.mid_channels, 
                                                      self.mid_channels, 
                                                      kernel_size=3, 
                                                      stride=self.stride, 
                                                      groups=self.mid_channels, 
                                                      bias=True), 
                                            nn.BatchNorm2d(self.mid_channels))

        self.group_conv2 = nn.Sequential(nn.Conv2d(self.mid_channels, 
                                                   self.output_channels,
                                                   kernel_size=1, 
                                                   groups=self.groups,
                                                   padding=self.padding), 
                                         nn.BatchNorm2d(self.output_channels), 
                                         nn.ReLU())
        
    def forward(self, x):
        original = x
        
        if self.stride == 2 :
            original = F.avg_pool2d(original, kernel_size=3, stride=2)
            
        x = self.group_conv1(x)
        x = shuffle(x, self.groups)
        x = self.depthwise_conv(x)
        x = self.group_conv2(x)
        
        if self.stride == 2 :
            x = torch.cat((original, x), 1)
        else:
            x = original + x
            
        x = F.relu(x)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, groups=3):
        super(ShuffleNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.base = 240
        
        self.groups = groups
        
        if self.groups == 1 :
            self.base = 144
        elif self.groups == 2 :
            self.base = 200
        elif self.groups == 4 :
            self.base = 272
        elif self.groups == 8 :
            self.base = 384
            
            
        stage2 = ShuffleUnit(24, self.base, groups=self.groups, stride=2)
        stage2_repeat = ShuffleUnit(self.base, self.base, groups=self.groups)
        
        self.stage2 = nn.Sequential(stage2,
                                    stage2_repeat,
                                    stage2_repeat,
                                    stage2_repeat)
            
        
        stage3 = ShuffleUnit(self.base, self.base*2, groups=self.groups, stride=2)
        stage3_repeat = ShuffleUnit(self.base*2, self.base*2, groups=self.groups)
        
        self.stage3 = nn.Sequential(stage3,
                                    stage3_repeat,
                                    stage3_repeat,
                                    stage3_repeat,
                                    stage3_repeat,
                                    stage3_repeat,
                                    stage3_repeat,
                                    stage3_repeat)
        
        stage4 = ShuffleUnit(self.base*2, self.base*4, groups=self.groups, stride=2)
        stage4_repeat = ShuffleUnit(self.base*4, self.base*4, groups=self.groups)
        
        self.stage4 = nn.Sequential(stage4,
                                    stage4_repeat,
                                    stage4_repeat,
                                    stage4_repeat)
                        
        
        self.fc = nn.Linear(self.base*4, 9)
     
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

 
