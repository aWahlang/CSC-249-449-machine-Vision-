import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        
        def conv_bn(inp, out, kernel, stride):
            return nn.Sequential(
                    nn.Conv2d(inp, out, kernel, stride, 1),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace = True)
                    )
    
        def conv_dw(inp, out, stride):
            return nn.Sequential(
                    #depthwise convolution
                    nn.Conv2d(inp, inp, 3, stride, 1, groups = inp),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace = True),
                    
                    #pointwise convolution
                    nn.Conv2d(inp, out, 1, 1, 0, bias = False),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace = True)
                    )
            
        self.model = nn.Sequential(
                conv_bn(3, 32, 3, 2),#224 -> 112
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),#122 -> 56
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),#56 -> 28
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),#28 -> 14
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),#14 -> 7
                conv_dw(1024, 1024, 1),
                nn.AvgPool2d(7),

                )
        
        self.fc1 = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100, 9)
     
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x

