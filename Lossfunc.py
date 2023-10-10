'''
Author: ReaganWu wuzhuoyu11@gmail.com
Date: 2023-10-09 16:57:24
LastEditors: ReaganWu wuzhuoyu11@gmail.com
LastEditTime: 2023-10-09 17:15:44
FilePath: /MonteCarlo_Reagan/Lossfunc.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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