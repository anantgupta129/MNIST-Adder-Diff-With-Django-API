import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

        self.dense1 = nn.Linear(1, 8)
        self.dense2 = nn.Linear(8, 32)
        # sum
        self.sum1 = nn.Linear(10+32, 64)
        self.sum2 = nn.Linear(64, 128)
        self.sum3 = nn.Linear(128, 1)
        # diff
        self.dif1 = nn.Linear(10+32, 64)
        self.dif2 = nn.Linear(64, 128)
        self.dif3 = nn.Linear(128, 1)

    def forward(self, x, num):# calling defined objects through forward function.
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))) # 1 convolutional block
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)# Flattening of the 1*1*10 matrix 
        out_lab = F.log_softmax(x, dim=1)

        num = F.relu(self.dense2(F.relu(self.dense1(num))))
        num = torch.cat((F.relu(x),num), dim=1)

        # sum processor
        out_sum = F.relu(self.sum1(num))
        out_sum = F.relu(self.sum2(out_sum))
        out_sum = self.sum3(out_sum)

        # diff processor
        out_dif = F.relu(self.dif1(num))
        out_dif = F.relu(self.dif2(out_dif))
        out_dif = self.dif3(out_dif)

        return out_lab, out_sum, out_dif
        
