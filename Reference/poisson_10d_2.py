import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange
from math import exp, sqrt, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import i0, i1, iv
from numpy import random
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")


class PDEQnet(nn.Module):
    def __init__(self, dim, width, beta):
        super(PDEQnet, self).__init__()
        self.dim = dim
        self.width = width
        self.innerwidth = int(width/2)
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
        # self.c.weight.data = torch.as_tensor(np.random.uniform(-1, 1, size=self.c.weight.shape), dtype=torch.float32)
        # self.wb.weight.data = torch.as_tensor(np.random.normal(0, 1, size=self.wb.weight.shape),  dtype=torch.float32)
        # self.wb.bias.data = torch.as_tensor(np.random.normal(0, 1, size=self.wb.bias.shape) ,dtype=torch.float32)
        # self.wb2.weight.data = torch.as_tensor(np.random.normal(0, 1, size=self.wb2.weight.shape),  dtype=torch.float32)
        # self.wb2.bias.data = torch.as_tensor(np.random.normal(0, 1, size=self.wb2.bias.shape) ,dtype=torch.float32)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.normal_(m.bias)


# Parameters
dim = 10
file = pd.read_csv("/home/reaganwu/Projects/MonteCarlo_Reagan/Dataset/poisson10d.csv", header=None)
data = file.values
test_x = data[:, 0:dim]
test_y = data[:, dim].reshape(-1,1)
print(test_y, test_x.shape)
test_y[np.abs(test_y) < 1.0e-32] = 0.0
print(test_y)
test_x = torch.tensor(test_x, dtype=torch.float32)
#d = [1 for i in range(dim)]

# Hyper parameters
N = 2**6
beta = 0.5+0.01
# Learning rate
initial_lr = 0.01

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

# Monte Carlo
Nmc_max = 5000
Nmc_initial = 500
Nbasepoints = int(1e7)

# Q fit, fixed grid
qnet = PDEQnet(dim, N, beta).to(device)
# initialization of PDEQnet paramters
qnet.assign_value()

source = torch.randn(size=(Nbasepoints, dim))
source = normalize(source, p=2.0)
radius = torch.rand(size = (Nbasepoints,1))
radius = torch.pow(torch.rand(size = (Nbasepoints,1)), 1/dim)
source = radius*source
max_value = torch.max(source)
min_value = torch.min(source)
if max_value > 1 or min_value < -1:
    raise ValueError

# Training algorithm, main
# Num of epoch
Num_epo = 10000

# Loss level
loss_list = np.zeros(Num_epo)
test_loss = np.zeros(Num_epo)
max_value_loss = np.zeros(Num_epo)
min_value_loss = np.zeros(Num_epo)

# Optmizer, scheduler
# Qoptimizer = optim.RMSprop(qnet.parameters(), lr=initial_lr, alpha=0.99, eps=1e-08)
# Qscheduler = LambdaLR(Qoptimizer, lr_lambda=lambda epoch: initial_lr / (1 + (epoch // 1000)))

Qoptimizer = optim.Adam(qnet.parameters(), lr=initial_lr)
Qscheduler = torch.optim.lr_scheduler.StepLR(Qoptimizer, step_size = 1000, gamma=0.8)

print(sum(p.numel() for p in qnet.parameters() if p.requires_grad))

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
    Qoptimizer.zero_grad()
    loss_to_min.backward()
    Qoptimizer.step()
    Qscheduler.step()

    loss = float(torch.dot(LQ, LQ))
    loss /= Nmc
    loss_list[count] = loss

    #=====================test loss=====================
    test_out = qnet(test_x.to(device)).cpu()
    test_out = torch.reshape(test_out, (-1,))
    test_prediction = test_out * eta(test_x)
    test_loss[count] = np.mean(np.abs(test_prediction.detach().numpy().reshape(-1, 1) - test_y))
    max_value_loss[count] = np.max(np.abs(test_prediction.detach().numpy().reshape(-1, 1) - test_y))
    min_value_loss[count] = np.min(np.abs(test_prediction.detach().numpy().reshape(-1, 1) - test_y))
    if count % 10 == 0:
        print('step: ', count, 'lr: ', Qscheduler.get_lr())
        print('step: ', count, 'mean loss: ', loss_list[count],
                'test mean loss: ', test_loss[count], 'max test loss: ', max_value_loss[count], 'min test loss: ', min_value_loss[count])

error = np.abs(test_prediction.detach().numpy().reshape(-1, 1) - test_y).reshape(-1)
model_save_name = 'poisson_10d.pkl'
path = f"./{model_save_name}"
torch.save(qnet.state_dict(), path)

# Loss level
plt.figure(figsize=(7,4.5))
#ax = fig.add_subplot(1, 2, 1)
axis=[i for i in range(Num_epo)]
axis = np.cumsum([1/(1+(i//500)) for i in range(Num_epo)])
plt.xlabel('Cumulative training time')
plt.ylabel('Loss level')
plt.yscale('log')
fig1 = plt.plot(axis,loss_list,'blue')
plt.savefig("poission_10d_1.jpg")

# Loss level
plt.figure(figsize=(7,4.5))
#ax = fig.add_subplot(1, 2, 1)
axis=[i for i in range(len(loss_list))]
plt.xlabel('Training epoch')
plt.ylabel('Loss level')
plt.yscale('log')
fig1 = plt.plot(axis,loss_list,'blue')
plt.savefig("poission_10d_2.jpg")


log_loss_list = [log(x) for x in loss_list]
plt.figure(figsize=(7,4.5))
#ax = fig.add_subplot(1, 2, 1)
axis=[i for i in range(Num_epo)]
plt.xlabel('Number of epochs')
plt.ylabel('Loss level')
fig1 = plt.plot(axis[400:600], log_loss_list[400:600],'blue')
plt.savefig("poission_10d_3.jpg")


plt.figure(4)
plt.plot(range(len(test_loss)), test_loss, 'ro', label = 'test mean loss')
plt.legend(loc='best')
plt.savefig("poission_10d_4.jpg")
plt.figure(5)
plt.plot(range(len(max_value_loss)), max_value_loss, 'ro', label = 'max loss value')
plt.legend(loc='best')
plt.savefig("poission_10d_5.jpg")
plt.figure(6)
plt.plot(range(len(min_value_loss)), min_value_loss, 'ro', label = 'min loss value')
plt.legend(loc='best')
plt.savefig("poission_10d_6.jpg")
plt.figure(7)
sns.displot(error, bins = 20)
plt.savefig("poission_10d_7.jpg")
plt.show()
