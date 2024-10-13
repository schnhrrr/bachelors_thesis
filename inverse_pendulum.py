#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from RK_Solver import RK_Solver

g = 9.81            # m/s^2
l = 3               # m
w0 = np.sqrt(g/l)   # s^-1
m = 1               # kg

def pendulum(y, t):
    """Function returns the derivative of the vector which consists 
    of the angular displacement (y[0]) and velocity (y[1])"""
    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = -g/l * np.sin(y[0])
    return dydt

def computeE0(y0, l):
    E_pot = m*g*l*(1-np.cos(y0[0]))
    E_kin = 0.5*m*(l*y0[1])**2
    return E_pot + E_kin

# Calculate solution with Runge-Kutta 4
t0 = 0          # s
tmax = 20       # s
t_step = 0.1   # s     
y0 = np.array([np.pi*0.75, np.pi/3])  # initial conditions
E0 = computeE0(y0, l)              # initial Energy of the system
t, y = RK_Solver(pendulum, y0, t0, tmax, t_step)
t_tensor = torch.Tensor(t).view(-1,1)
y1_tensor = torch.Tensor(y[:,0]).view(-1,1)
y2_tensor = torch.Tensor(y[:,1]).view(-1,1)

# plot angular displacement over time (RK4 solution)
fig1 = plt.figure(1)
plt.title('Non linear pendulum')
plt.plot(t,y[:,0])
plt.xlabel(r'$t$')
plt.ylabel(r'$\phi (t)$')
plt.show()

# plot phase space (RK4 solution)
fig2 = plt.figure(2)
plt.title('Phase space')
plt.plot(y[:,0], y[:,1])
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\omega$')
plt.show()

# Define neural network
class PINN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fci = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation()) # fully connected input layer
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN),activation()]) for _ in range(N_LAYERS-1)])
        self.fco = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, t):
        firstlayer = self.fci(t)
        hiddenlayer = self.fch(firstlayer)
        phi = self.fco(hiddenlayer)
        return phi

# visualize training progress   
def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    plt.figure()
    plt.plot(x,yh, color="tab:red", linewidth=2, alpha=0.8, label="NN prediction")
    plt.plot(x,y, color="blue", linewidth=2, alpha=0.8,linestyle='--', label="Exact solution")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=30, color="tab:green", alpha=0.4, 
                    label='Coloc. points')
    l = plt.legend()
    plt.setp(l.get_texts(), color="k")
    #plt.xlim(-0.05, 10)
    #plt.ylim(-np.pi, np.pi)
    plt.title("Step: %i"%(i+1))
    plt.ylabel(r'$\phi$ / rad')
    plt.xlabel(r'$t$ / s')
    plt.axis("on")

# Generate collocation points (sampled over problem domain)
t_colloc = torch.linspace(0,tmax,50).view(-1,1).requires_grad_(True)

# Defining tensors of data points
t_data = t_tensor[::20]
phi_data = y1_tensor[::20]

# Hyperparameters
w_data = 1
w_physics = 4.5e-3
w_energy = 5.e-3
lr = 7e-4

torch.manual_seed(123)
model = PINN(1,1,32,4)

# Adding trainable parameter to network (length of pendulum)
L = torch.tensor(1.0, requires_grad=True)
parameters = list(model.parameters())
parameters.append(L)

optimizer = torch.optim.Adam(parameters,lr=lr)
files = []
loss_data_history = []
loss_physics_history = []
loss_energy_history = []
loss_history = []
L_history = []
mse_history = []

# Beginn Training
for i in range(50000):
    if i % 5000 == 0: 
        print('Epoche: ',i)

    optimizer.zero_grad() # set gradient to zero
    
    # compute "data loss"
    phi_data_eval = model(t_data)
    loss_data = w_data*torch.mean((phi_data_eval-phi_data)**2)

    # compute "Physics Loss"
    phi_physics_eval    = model(t_colloc)
    dphi                = torch.autograd.grad(phi_physics_eval, t_colloc, torch.ones_like(phi_physics_eval), create_graph=True)[0] # dphi/dt
    dphi2               = torch.autograd.grad(dphi, t_colloc, torch.ones_like(dphi), create_graph=True)[0] # d^2phi/dt^2
    physics_residual    = dphi2 + g/L * torch.sin(phi_physics_eval)
    loss_physics        = w_physics*torch.mean(physics_residual**2)

    # compute "Energy Loss"
    energy_residual     = m*g*L*(1-torch.cos(phi_physics_eval)) + 0.5*m*(L*dphi)**2 - computeE0(y0, L)
    loss_energy         = w_energy*torch.mean(energy_residual**2)

    phi_eval = model(t_tensor)
    mse = torch.mean((phi_eval-y1_tensor)**2)

    # Total loss backpropagation
    loss = loss_data + loss_physics + loss_energy
    loss.backward()
    optimizer.step()

    # plot the result
    if (i+1) % 100 == 0:
        loss_data_history.append(loss_data.detach())
        loss_physics_history.append(loss_physics.detach())
        loss_energy_history.append(loss_energy.detach())
        loss_history.append(loss.detach())
        L_history.append(L.clone().detach().numpy())
        mse_history.append(mse.detach())

        phi_predict = model(t_tensor).detach()
        tp = t_colloc.detach()

        plot_result(t, y[:,0], t_data, phi_data, phi_predict, tp)

        if (i+1) % 10000 == 0: 
            plt.show()
            print(f'The real length of the pendulum is {l}, and the predicted length is {L}')
        else: plt.close("all")



# Plot loss history and MSE
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
plt.yscale('log')
plt.legend()
plt.show()

# L over iterations plot
fig44 = plt.figure(44)
plt.plot([1]+L_history, label=r'Trained parameter $L$')
plt.plot([0, 600],[3,3], label=r'True length $l=3$ m', linestyle='--', color='k', linewidth=1.2)
plt.text(300,2,'L(50000) = 2.9915 m')
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")
plt.ylabel('$L$ / m')
plt.xlim(-2,500)
plt.legend()
