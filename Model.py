'''
Author: ReaganWu wuzhuoyu11@gmail.com
Date: 2023-10-07 14:14:48
LastEditors: ReaganWu wuzhuoyu11@gmail.com
LastEditTime: 2023-10-09 16:13:50
FilePath: /MonteCarlo_Reagan/Model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

'''
The basic model from PQENet, but I wanna use directly training method instead of Q-Leanring
'''
class CustomNet(nn.Module):
    def __init__(self, dim, width, beta):
        super(CustomNet, self).__init__()
        self.dim = dim
        self.width = width
        self.innerwidth = int(width / 2)
        self.beta = beta
        self.fc1 = nn.Linear(self.dim, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, self.width)
        self.fc4 = nn.Linear(self.width, self.width)
        self.outlayer = nn.Linear(self.width, 1, bias=True)

    def forward(self, x):
        s = torch.nn.functional.pad(x, (0, self.width - self.dim))
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        y = y + s
        s = y
        y = self.fc3(y)
        y = torch.sigmoid(y)
        y = self.fc4(y)
        y = torch.sigmoid(y)
        y = y + s
        output = self.outlayer(y)
        return output
    
    def assign_value(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.normal_(m.bias)