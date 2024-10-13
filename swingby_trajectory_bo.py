#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

ao_xygm     =  [[-0.5, -1.0, 0.5],  # astronomic objects: x, y, gravitational mass
                [-0.2,  0.4, 1.0],
                [ 0.8,  0.3, 0.5]]

def trainModelWithHyperparameters(w_BC, w_physics):
    # Initialize model
    model = NN(n_input, n_output, n_neurons, n_layers)
    parameters = list(model.parameters())
    parameters.append(T)
    optimizer = torch.optim.Adam(parameters, lr=lr)

    loss_BC_history = []
    loss_physics_history = []
    loss_history = []

    # Begin training
    for i in range(5000):
        if (i+1) % 5000 == 0: print('Epoche: ',i+1,f'\nw1: {w_BC}\nw2: {w_physics}\n\n')

        optimizer.zero_grad()    

        # Compute loss from boundary conditions
        BC0 = model(tmin)
        BC1 = model(tmax)
        loss_BC = (torch.mean((BC0-r0)**2) + torch.mean((BC1-r1)**2)) / 2

        # Compute loss from physics
        r_model = model(t_colloc)
        Fx, Fy = ode(t_colloc, r_model) 
        Thrust = torch.sqrt(Fx**2 + Fy**2)
        loss_physics = torch.mean(Thrust**2)

        loss = w_BC * loss_BC + w_physics * loss_physics
        loss.backward()
        optimizer.step()

        # Visualize training progress
        if (i+1) % 100 == 0:
            loss_BC_history.append(loss_BC.detach())
            loss_physics_history.append(loss_physics.detach())
            loss_history.append(loss.detach())

            if (i+1) % 1000 == 0: 
                plotTrajectory(r_model)
                plotGravityAndThrust(r_model, Thrust)
            else: plt.close("all")

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

    return -loss.clone().detach().numpy()

# %%
# Bayesian optimization of the hyperparameters w_BC and w_physics        
#############################################################################

# Gaussian-Process-Modelling - Initial Sampling *****************************

# searching in the rectangle with upper and lower bounds:
lowerw1 = 1e2
upperw1 = 1e4

lowerw2 = 0.1
upperw2 = 1e1

interval_w1 = [lowerw1, upperw1]
interval_w2 = [lowerw2, upperw2]

bounds = torch.tensor([[lowerw1, lowerw2], [upperw1, upperw2]])

# initializing list of points in the 3d loss landscape (w1,w2,z)
points = []

# Computing loss values on the corners of the domain 
for w2 in interval_w2:
    for w1 in interval_w1:
        z = trainModelWithHyperparameters(w1, w2)
        new_point = torch.tensor([w1,w2,z])
        points.append(new_point)

# Computing loss values on random points inside of the domain
n_random_points = 50
random_points = np.exp(np.random.uniform(np.log(lowerw1), np.log(upperw1), size=(n_random_points,2)))
random_points[:,1] = np.exp(np.random.uniform(np.log(lowerw2), np.log(upperw2), size=(n_random_points)))

for w1, w2 in random_points:
    z = trainModelWithHyperparameters(w1, w2)
    new_point = torch.tensor([w1,w2,z])
    points.append(new_point)

# making tensor out of list of points
loss_landscape = torch.stack(points)

# getting lowest initial loss (max. of -Loss)
best_init_loss = loss_landscape[:,2].max().item() 

# %% 
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf


def getNextPoints(loss_landscape, best_init_loss, bounds, n_points=1):
    single_model = SingleTaskGP(loss_landscape[:,:2], loss_landscape[:,2].unsqueeze(1))
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)

    EI = qExpectedImprovement(model=single_model, best_f=best_init_loss)

    candidates, _ = optimize_acqf(
                    acq_function=EI,
                    bounds=bounds,
                    q=n_points, # Number of candidates returned
                    num_restarts=100,
                    raw_samples=512) 
    return candidates


# Optimization loop **********************************************************************
n_runs = 50
for i in range(n_runs):
    print(f"Nr. of optimization run: {i+1}")

    new_candidate = getNextPoints(loss_landscape, best_init_loss, bounds) # w1 and w2
    values = new_candidate.tolist()[0]

    new_loss_value = trainModelWithHyperparameters(*values) # -loss value
    values.append(new_loss_value)

    new_point = torch.tensor(values)
    points.append(new_point)
    loss_landscape = torch.stack(points)


    best_init_loss = loss_landscape[:,2].max().item() 

# Visualize 3d loss landscape ********************************************************
w1 = [point[0].item() for point in loss_landscape]
w2 = [point[1].item() for point in loss_landscape]
z  = [point[2].item() for point in loss_landscape]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w1,w2,z)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('Loss Landscape')
plt.show()

max_index = torch.argmax(loss_landscape[:,2])
best_w1 = loss_landscape[max_index, 0].item()
best_w2 = loss_landscape[max_index, 1].item()

print(f'Best values are w1: {best_w1} and w2: {best_w2}')

# %%