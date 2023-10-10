'''
Author: ReaganWu wuzhuoyu11@gmail.com
Date: 2023-10-07 18:11:13
LastEditors: ReaganWu wuzhuoyu11@gmail.com
LastEditTime: 2023-10-08 10:16:23
FilePath: /MonteCarlo_Reagan/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from Model import CustomNet
from Loader_Monte import load_data, generate_samples
from train import train_model

# 设置超参数和初始化
dim = 10
Num_epo = 10000
device = torch.device("cuda")

Nmc_max, Nmc_initial, source = generate_samples(5000, 500, int(1e7), dim)
qnet = CustomNet(dim, 64, 0.51).to(device)
qnet.assign_value()

test_x, test_y = load_data("Dataset/poisson10d.csv")
initial_lr = 0.01

# epoch_sample 函数
def epoch_sample(Nmc, source, dim):
    index = np.random.randint(0, source.shape[0], size=Nmc)
    x = source[index, :]
    return torch.tensor(x, dtype=torch.float32)

# 创建优化器和学习率调度器
optimizer = torch.optim.Adam(qnet.parameters(),
                                lr=initial_lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=1000, 
                                             gamma=0.8)


# 训练模型
loss_list, test_loss, max_value_loss, min_value_loss = train_model(qnet, source, epoch_sample, optimizer, scheduler, dim)

# 可视化损失和结果
# 添加可视化代码
