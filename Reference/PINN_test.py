import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

# Define your PDEQnet class as in your original code
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


def compute_pde_residual(output, grid):
    # Compute the PDE residual for your specific problem
    # Example: Laplace equation in 2D
    u_xx = torch.autograd.grad(output, grid, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_xx, grid, grad_outputs=torch.ones_like(u_xx), create_graph=True, retain_graph=True)[0]

    # The PDE residual is the difference between the LHS and RHS
    # LHS for the Laplace equation is Laplacian(u) = u_xx + u_yy
    LHS = u_xx + u_yy

    # Example: RHS of Laplace equation is 0
    RHS = torch.zeros_like(output)

    # Compute the residual
    residual = LHS - RHS

    return residual


# Define loss functions for the PDE and data-driven loss
def pde_loss(output, grid):
    # Compute the PDE residual using the output and grid
    pde_residual = compute_pde_residual(output, grid)
    return torch.mean(pde_residual**2)

def data_driven_loss(output, data, target):
    # Compute the loss for the Monte Carlo data generator
    data_residual = output - data
    return torch.mean(data_residual**2)

# Create your data generator function (Monte Carlo in your case) and provide sample data
# Create your data generator function (Monte Carlo in your case) and provide sample data
def data_generator(num_samples):
    # Generate random data for training (as tensors with requires_grad set to True)
    data = torch.randn((num_samples, input_dim), requires_grad=True)  # Set requires_grad to True
    target = torch.randn((num_samples, 1), requires_grad=True)  # Set requires_grad to True
    return data, target

# Hyperparameters
input_dim = 10
N = 64
initial_lr = 0.01
Num_epo = 10000
pde_weight = 1.0
data_driven_weight = 1.0
beta = 0.5+0.01

# Initialize your PDEQnet
qnet = PDEQnet(input_dim, N, beta)

# Define an optimizer and learning rate scheduler
Qoptimizer = optim.Adam(qnet.parameters(), lr=initial_lr)
Qscheduler = StepLR(Qoptimizer, step_size=1000, gamma=0.8)

# Training loop
for count in trange(Num_epo):
    # Generate Monte Carlo data
    num_samples = 500  # Adjust the number of samples
    data, target = data_generator(num_samples)

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


# Save the trained model if needed
torch.save(qnet.state_dict(), "pinn_model.pth")
