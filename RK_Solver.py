import numpy as np

A_RK4 = np.array([[0, 0, 0, 0],[0.5, 0, 0, 0],[0, 0.5, 0, 0],[0, 0, 1, 0]]) 
B_RK4 = np.array([1/6, 1/3, 1/3, 1/6])
C_RK4 = np.array([0, 0.5, 0.5, 1])

''' Runge-Kutte-Solver:
    inputs:
    -Fun(y, t)     Function which gives you derivative of given y and t
    -y0            Initial-Values
    -t0            Starting time
    -tmax          Ending time
    -epsilon       Step size
    -A, B, C       Butcher tableau
    
    outputs:
    -time_vec      Vector reaching from t0 to tmax with stepsize epsilon
    -solution      Matrix with the solution of the system'''

def RK_Solver(Fun, y0, t0, tmax, epsilon, A=A_RK4, B=B_RK4, C=C_RK4): # Fun(y, t), A, B, C and y0 must be np-arrays

    NUM_EQUATIONS = len(y0)
    NUM_INTERMEDIATE_STEPS = len(A)

    time_vec = np.arange(t0, tmax + epsilon, epsilon)
    solution = np.zeros((len(time_vec),NUM_EQUATIONS))
    solution[0,:] = y0
 
    for n, t_n in enumerate(time_vec[:-1]):
        k = np.zeros((NUM_EQUATIONS, NUM_INTERMEDIATE_STEPS))
        
        # calculating k-Matrix, if there are more than 1 stages
        if NUM_INTERMEDIATE_STEPS > 1:
            for i in range(NUM_INTERMEDIATE_STEPS):
                k[:,i] = solution[n,:] + epsilon * np.sum(A[i,:] * Fun(k, t_n + epsilon * C), axis=1)   
            solution[n+1,:] = solution[n,:] + epsilon * np.sum(B * Fun(k, t_n + epsilon*C), axis=1)

        # calculating k-Vector, if there is only 1 stage
        else:
            k = solution[n,:] + epsilon * A * Fun(k, t_n + epsilon * C).T
            solution[n+1,:] = solution[n,:] + epsilon * np.sum(B * Fun(k.T, t_n + epsilon * C), axis=1)


    return time_vec, solution


