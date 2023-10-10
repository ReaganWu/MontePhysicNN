'''
Author: ReaganWu wuzhuoyu11@gmail.com
Date: 2023-10-07 18:09:15
LastEditors: ReaganWu wuzhuoyu11@gmail.com
LastEditTime: 2023-10-09 17:00:36
FilePath: /MonteCarlo_Reagan/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import trange
from Loader_Monte import eta, eta_diff, eta_diff_2, RHS_pde, generate_samples
from Lossfunc import pde_loss, data_driven_loss


def train_model(qnet, Num_epo, pde_weight, data_driven_weight, Qoptimizer, Qscheduler):


    for count in trange(Num_epo):
        # Generate Monte Carlo data
        num_samples = 500  # Adjust the number of samples
        data, target = generate_samples(num_samples)

        # Net output
        out = qnet(data)
        # Compute the PINN loss
        pde_term = pde_loss(out, data)
        data_driven_term = data_driven_loss(out, data, target)
        total_loss = pde_weight * pde_term + data_driven_weight * data_driven_term

        # Optimize the network to minimize the total loss
        Qoptimizer.zero_grad()
        total_loss.backward()
        Qoptimizer.step()
        Qscheduler.step()

        # Monitor your training progress and losses
        if count % 10 == 0:  # Print every 10 epochs or adjust as needed
            print(f"Epoch {count}/{Num_epo}, Loss: {total_loss.item()}")

    return total_loss
