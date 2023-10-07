'''
Author: ReaganWu wuzhuoyu11@gmail.com
Date: 2023-10-07 15:14:01
LastEditors: ReaganWu wuzhuoyu11@gmail.com
LastEditTime: 2023-10-07 18:27:54
FilePath: /MonteCarlo_Reagan/Loader_Monte.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import numpy as np
import pandas as pd
from torch.nn.functional import normalize
import random

#===========Hyper Parameters===========#
Nmc_max = 5000
Nmc_initial = 500
Nbasepoints = int(1e6)
dim = 10
device = torch.device("cuda")

# Auxiliary functions
def eta(grid):
    res = 1.0
    for i in range(dim):
        res = res -  grid[:,i]*grid[:,i]
    return res

def eta_diff(grid, count):
    return -2*grid[:,count]

def eta_diff_2(grid, count):
    return -4

def RHS_pde(x):
    dim = x.size(1)
    RHS = 1
    for i in range(dim):
        RHS = RHS*torch.sin(4*np.pi*x[:, i:(i+1)])
        RHS = -160*np.pi**2*RHS
    return RHS.to(device)

def load_data(filename):
    # 从CSV文件加载数据
    file = pd.read_csv(filename, header=None)
    data = file.values
    test_x = data[:, 0:dim]  # 提取输入特征
    test_y = data[:, dim].reshape(-1, 1)  # 提取目标标签
    test_y[np.abs(test_y) < 1.0e-32] = 0.0  # 将非常小的目标标签值设为0
    test_x = torch.tensor(test_x, dtype=torch.float32)  # 转换为PyTorch张量
    return test_x, test_y

def generate_samples(Nmc_max, Nmc_initial, Nbasepoints, dim):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")

    source = torch.randn(size=(Nbasepoints, dim))  # 随机生成Nbasepoints个样本
    source = normalize(source, p=2.0)  # 归一化样本
    radius = torch.pow(torch.rand(size=(Nbasepoints, 1)), 1 / dim)  # 随机生成半径
    source = radius * source  # 将半径应用于样本
    max_value = torch.max(source)  # 计算最大值
    min_value = torch.min(source)  # 计算最小值

    if max_value > 1 or min_value < -1:
        raise ValueError  # 检查最大值和最小值是否在合理范围内

    return Nmc_max, Nmc_initial, source
