'''
Author: ReaganWu wuzhuoyu11@gmail.com
Date: 2023-10-07 18:09:15
LastEditors: ReaganWu wuzhuoyu11@gmail.com
LastEditTime: 2023-10-07 18:37:04
FilePath: /MonteCarlo_Reagan/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import trange
from Loader_Monte import eta, eta_diff, eta_diff_2, RHS_pde


def train_model(qnet, grid_list, epoch_sample, optimizer, scheduler, dim):


    loss_list = np.zeros(Num_epo)
    test_loss = np.zeros(Num_epo)
    max_value_loss = np.zeros(Num_epo)
    min_value_loss = np.zeros(Num_epo)

    for count in trange(Num_epo):
    Nmc = int(Nmc_initial + (Nmc_max - Nmc_initial) * (1 + count) / (1 + Num_epo))
    epoch_sample = torch.randint(0, len(source), (Nmc, 1))
    grid_list = []
    for i in range(dim):
        ent = [[u] for u in source[epoch_sample, i]]  # grab sample from reservoir
        ent = torch.tensor(ent, requires_grad=True).to(device)
        grid_list.append(ent)
    grid_tuple = tuple(grid_list)
    grid = torch.cat(grid_tuple, 1).to(device)

    # Net output
    out = qnet(grid)

    # Tensor reshape
    out_r = torch.reshape(out, (-1,))
    # Compute partial derivatives and the Laplacian
    l2 = 0.0
    for k, entry in enumerate(grid_list):
        df_dx = torch.autograd.grad(out, entry, grad_outputs=torch.ones(entry.size(),
                    device=device), create_graph=True)[0]
        d2f_dx2 = torch.autograd.grad(df_dx, entry, grad_outputs=torch.ones(entry.size(),
                    device=device), create_graph=True)[0]
        df_dx = torch.reshape(df_dx, (-1,))
        d2f_dx2 = torch.reshape(d2f_dx2, (-1,))
        l2 += d2f_dx2 * eta(grid) + 2 * df_dx * eta_diff(grid, k) + out_r * eta_diff_2(grid, k)
    # evaluate function
    l = out_r * eta(grid)
    # evaluate PDE operator
    lq = RHS_pde(grid).reshape(-1) + l2.to(device)
    LQ = lq.clone().detach().to(device)


    # Q-learning
    loss_to_min = -1 * torch.dot(LQ, l.to(device))
    # with torch.cuda.stream(s):
    optimizer.zero_grad()
    loss_to_min.backward()
    optimizer.step()
    scheduler.step()

    loss = float(torch.dot(LQ, LQ))
    loss /= Nmc
    loss_list[count] = loss

    return loss_list, test_loss, max_value_loss, min_value_loss
