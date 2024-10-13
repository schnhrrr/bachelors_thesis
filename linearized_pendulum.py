#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Code inspired/taken from: https://github.com/hubertbaty/PINNS-EDO/blob/main/pinn-harmo-figs8%269.ipynb

# Analytical Solution 
# https://de.wikipedia.org/wiki/Mathematisches_Pendel

PHI_MAX = np.deg2rad(10)
l = 0.1
g = 9.81

def pendulum(t, phi_0=PHI_MAX, l=l, g=g, alpha=0):         
    w0 = np.sqrt(g/l)                                      
    sin = torch.sin(w0*t + alpha)                         
    phi = phi_0*sin
    return phi

# Define PINN class
class PINN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fci = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation()) # fully connected input layer
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation()]) for _ in range(N_LAYERS-1)])
        self.fco = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, t):
        phi = self.fci(t)
        phi = self.fch(phi)
        phi = self.fco(phi)
        return phi

# Visualize results    
def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    plt.figure()
    plt.plot(x,yh, color="tab:red", linewidth=2, alpha=0.8, label="NN prediction")
    plt.plot(x,y, color="blue", linewidth=2, alpha=0.8,linestyle='--', label="Exact solution")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=30, color="tab:green", alpha=0.4, 
                    label='Colloc. points')
    l = plt.legend()
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.3, 0.3)
    plt.title("Step: %i"%(i+1))
    plt.ylabel(r'$\phi$ / rad')
    plt.xlabel(r'$t$ / s')
    plt.axis("on")

# Compute analytical solution
t = torch.linspace(0,1,1000).view(-1,1)
phi = pendulum(t).view(-1,1)

# Compute two training points
t_data = torch.linspace(0,0.1,2).view(-1,1)
phi_data = pendulum(t_data).view(-1,1)

# Collocation points
t_physics = torch.linspace(0,1,40).view(-1,1).requires_grad_(True)

plt.figure()
plt.plot(t, phi, label="Exact solution")
plt.scatter(t_data, phi_data, label="Training data", color='r')
plt.legend()
plt.show()

torch.manual_seed(123)
model = PINN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
loss_data_history = []
loss_physics_history = []
loss_energy_history = []
loss_history = []
mse_history = []

# Set hyperparameters
w_data = 1.
w_physics = 1.e-4
w_energy = 1.e-4

# Training loop
for i in range(30000):
    if i % 1000 == 0: 
        print('Epoche: ',i)

    optimizer.zero_grad() # set gradient to zero
    
    # Compute "Data Loss" 
    phi_data_eval = model(t_data)
    loss_data = w_data*torch.mean((phi_data_eval-phi_data)**2)

    # Compute "Physics Loss" 
    phi_physics_eval = model(t_physics)
    dphi = torch.autograd.grad(phi_physics_eval, t_physics, torch.ones_like(phi_physics_eval), create_graph=True)[0] # dphi/dt
    dphi2 = torch.autograd.grad(dphi, t_physics, torch.ones_like(dphi), create_graph=True)[0] # d^2phi/dt^2
    physics_residual = dphi2 + g/l * phi_physics_eval
    loss_physics = w_physics*torch.mean(physics_residual**2)

    # Compute "Energy Loss" 
    energy_residual = g*l*phi_physics_eval*dphi + l**2 * dphi *dphi2
    loss_energy = w_energy*torch.mean(energy_residual**2)

    phi_eval = model(t)
    mse = torch.mean((phi_eval-phi)**2)

    # Total loss backpropagation
    loss = loss_data + loss_physics + loss_energy
    loss.backward()
    optimizer.step()

    # Plot solution 
    if (i+1) % 100 == 0:
        loss_data_history.append(loss_data.detach())
        loss_physics_history.append(loss_physics.detach())
        loss_energy_history.append(loss_energy.detach())
        loss_history.append(loss.detach())
        mse_history.append(mse.detach())

        phi_predict = model(t).detach()
        tp = t_physics.detach()

        plot_result(t, phi, t_data, phi_data, phi_predict, tp)

        if (i+1) % 6000 == 0: plt.show()
        else: plt.close("all")

# Plot loss and MSE history after training loop
plt.plot(loss_history, label="loss")
plt.yscale('log')
plt.legend()

plt.plot(loss_physics_history, label="loss_physics")
plt.yscale('log')
plt.legend()

plt.plot(loss_data_history, label="loss_data")
plt.yscale('log')
plt.legend()

plt.plot(loss_energy_history, label="loss_energy")
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")
plt.yscale('log')
plt.legend()

fig33 = plt.figure(33)
plt.plot(loss_history, label="Loss")
plt.plot(mse_history, label="MSE")
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")
plt.yscale('log')
plt.legend()  
