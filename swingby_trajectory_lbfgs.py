#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Architecture of NN
n_input     = 1 # normalized time
n_output    = 2 # x,y
n_layers    = 3
n_neurons   = 64
lr          = 1.e-4

# Define NN class
class NN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_NEURONS, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fci = nn.Sequential(nn.Linear(N_INPUT, N_NEURONS), activation()) # fully connected input layer
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_NEURONS, N_NEURONS),activation()]) for _ in range(N_LAYERS-1)])
        self.fco = nn.Linear(N_NEURONS, N_OUTPUT)

    def forward(self, t):
        firstlayer = self.fci(t)
        hiddenlayer = self.fch(firstlayer)
        x = self.fco(hiddenlayer)
        return x

def plotTrajectory(r):
    x, y = r[:,0].detach().numpy(), r[:,1].detach().numpy()
    plt.figure()
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.plot(x,y)
    for xtemp, ytemp, gmtemp in ao_xygm:
        plt.scatter(xtemp, ytemp, s=gmtemp*500, c='k', marker='o')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()    

def plotGravityAndThrust(r, Thrust):
    x, y = r[:,0].detach().numpy(), r[:,1].detach().numpy()
    Thrust = Thrust.detach().numpy()
    Gx = 0
    Gy = 0
    for xtemp, ytemp, gmtemp in ao_xygm:
        Gx += gmtemp * m0 * (x - xtemp) / ((x - xtemp)**2 + (y - ytemp)**2)**1.5
        Gy += gmtemp * m0 * (y - ytemp) / ((x - xtemp)**2 + (y - ytemp)**2)**1.5
    
    t = np.linspace(0, 1, len(x))
    G = np.sqrt((Gx**2 + Gy**2)) 
    
    plt.figure()
    plt.plot(t, G, label="Total gravity" , color='k')
    plt.plot(t, Thrust, label="Thrust")
    plt.fill_between(t,np.zeros_like(Thrust[:,0]),Thrust[:,0])
    plt.plot(t, G.reshape(-1,1) + Thrust, label="Required force for the trajectory", linestyle="--")
    plt.xlabel('Normalized time')
    plt.ylabel('Exerted force')
    plt.legend()
    plt.show()

# Thrust vector
def ode(t, r):
    x, y    = r[:,0], r[:,1]
    dxdt    = torch.autograd.grad(x, t, create_graph=True, grad_outputs=torch.ones_like(x))[0] / T
    dxdt2   = torch.autograd.grad(dxdt, t, create_graph=True, grad_outputs=torch.ones_like(dxdt))[0] / T
    dydt    = torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones_like(y))[0] / T
    dydt2   = torch.autograd.grad(dydt, t, create_graph=True, grad_outputs=torch.ones_like(dydt))[0] / T

    ode_x = (m0 * dxdt2).view(1,-1)
    ode_y = (m0 * dydt2).view(1,-1)

    for xtemp, ytemp, gmtemp in ao_xygm:
        ode_x += gmtemp * m0 * (x - xtemp) / ((x - xtemp)**2 + (y - ytemp)**2)**1.5
        ode_y += gmtemp * m0 * (y - ytemp) / ((x - ytemp)**2 + (y - ytemp)**2)**1.5
    
    return ode_x.view(-1,1), ode_y.view(-1,1)

# Set random seed
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

# Physical Parameters
tmin, tmax  = torch.tensor([[0.0]]), torch.tensor([[1.0]]) # normalized time
t_colloc    = torch.linspace(0,1,200).view(-1,1).requires_grad_(True)
r0          = torch.tensor([[-1.,-1.]]) # start point
r1          = torch.tensor([[1.,1.]]) # end point
m0          = 1.
T           = torch.tensor(1.0, requires_grad=True) # end time

ao_xygm     =  [[-0.5,  -1., 0.5],  # astronomic objects: x, y, gravitational mass
                [-0.2,  0.4, 1.0],
                [ 0.8,  0.3, 0.5]]


# Initialize model
model = NN(n_input, n_output, n_neurons, n_layers)
parameters = list(model.parameters())
parameters.append(T)
optimizer_adam = torch.optim.Adam(parameters, lr=lr)
optimizer_lbfgs = torch.optim.LBFGS(parameters, lr=.1, max_iter=10)

# Initialize
loss_BC_history = []
loss_physics_history = []
loss_history = []

# Hyperparameters
w_BC = 160
w_physics = .9

# Begin training
for i in range(3100):
    if i % 100 == 0: print('Epoche: ',i)

    def closure():
        optimizer.zero_grad()

        # Compute loss from boundary conditions
        BC0 = model(tmin)
        BC1 = model(tmax)
        loss_BC = (torch.mean((BC0 - r0)**2) + torch.mean((BC1 - r1)**2)) / 2

        # Compute loss from physics
        r_model = model(t_colloc)
        Fx, Fy = ode(t_colloc, r_model)
        Thrust = torch.sqrt(Fx**2 + Fy**2)
        loss_physics = torch.mean(Thrust**2)

        loss = w_BC * loss_BC + w_physics * loss_physics
        if (i+1) % 100 == 0:
            loss_BC_history.append(loss_BC.detach())
            loss_physics_history.append(loss_physics.detach())
            loss_history.append(loss.detach())
        loss.backward()
        return loss

    # 3000 iterations ADAM, 100 iterations LBFGS
    if i<3000:
        optimizer = optimizer_adam
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()
    else:
        optimizer = optimizer_lbfgs
        optimizer.step(closure)    

    # Visualize training progress
    if (i+1) % 3100 == 0:
        r_model = model(t_colloc)
        Fx, Fy = ode(t_colloc, r_model)
        Thrust = torch.sqrt(Fx**2 + Fy**2)
        plotTrajectory(r_model)
        plotGravityAndThrust(r_model, Thrust)

# Plotting loss history
plt.figure()
plt.plot(loss_BC_history, label="BC Loss")
plt.yscale('log')
plt.legend()
plt.plot(loss_physics_history, label="Physics Loss")
plt.yscale('log')
plt.legend()
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")

plt.figure()
plt.plot(loss_history, label="Total Loss")
plt.yscale('log')
plt.legend()
plt.xlabel('Training step ($10^2$)',fontsize="xx-large")
