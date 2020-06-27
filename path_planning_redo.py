# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:42:02 2020

@author: JiNzo
"""

#%% Import packages
from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from casadi import *


#%% Setup the problem

L = 1.5 #Define what the length of the car is, as this will affect the turning circle.

Nsim    = 50           # how much samples to simulate
nx = 4                  # x, y, v, theta (angle bicycle)
nu = 2                  # a, delta (angle wheel)
Tf = 2                  # Control horizon [s]
Nhor = 20              #number of control intervals
dt = Tf/Nhor            #sample time

#Initialise the matrices for logging variables
time_hist      = np.zeros((Nsim+1, Nhor+1))
x_hist         = np.zeros((Nsim+1, Nhor+1))
y_hist         = np.zeros((Nsim+1, Nhor+1))
v_hist         = np.zeros((Nsim+1, Nhor+1))
theta_hist     = np.zeros((Nsim+1, Nhor+1))
a_hist         = np.zeros((Nsim+1, Nhor+1))
delta_hist     = np.zeros((Nsim+1, Nhor+1))

p_hist = np.zeros((Nsim+1, 2))

#Initialise a matrix for plotting the position of the moving object

# Define the type of optimal control problem
ocp = Ocp(T=FreeTime(10.0)) #Freetime problem because otherwise it will reach the destination in the solution time

# Define states, represented as CasADi matrix expressions
x     = ocp.state()
y     = ocp.state()
v     = ocp.state()
theta = ocp.state()

delta = ocp.control()
a     = ocp.control()

# Specify the ODE's that define the behaviour of the system (bicycle model)
ocp.set_der(x,      v*cos(theta))
ocp.set_der(y,      v*sin(theta))
ocp.set_der(theta,  v/L*tan(delta))
ocp.set_der(v,      a)

# Define parameter
X_0 = ocp.parameter(nx)

# Initial constraints
X = vertcat(x, y, v, theta)
U = vertcat(a, delta)
ocp.subject_to(ocp.at_t0(X) == X_0)

# Initial guesses
ocp.set_initial(x,      0)
ocp.set_initial(y,      0)
ocp.set_initial(theta,  0)
ocp.set_initial(v,    0.5)

ocp.set_initial(a,      0.5)
ocp.set_initial(delta,  0)

# Path constraints
ocp.subject_to(0 <= (v <= 5))
ocp.subject_to(-2 <= (x <= 12))
ocp.subject_to(-2 <= (y <= 12))

ocp.subject_to(-5 <= (a <= 5))
ocp.subject_to(-pi/6 <= (delta <= pi/6))

# Add a stationary objects along the path
# TODO: clearly the current solution isn't taking into account the obstacle, so fix this
p0 = vertcat(6,6)
r0 = 1

p = vertcat(x,y)
ocp.subject_to(sumsqr(p-p0)>=r0**2)

# Now input a moving obstacle
p_move = ocp.parameter(2)
r_move = 1
# Add a constraint that the car cannot come close to the obstacle
ocp.subject_to(sumsqr(p-p_move)>=r_move**2)

final_X_x = 10
final_X_y = 10

# Objective functions
# TODO: tune the weightings of the different objectives
# Currently the problem is that the car wont reach the destination because the control horizon makes it hit target in the last control step
ocp.add_objective(sumsqr(ocp.T))
ocp.add_objective(ocp.sum(0.5*sumsqr(a),grid='control'))
ocp.add_objective(ocp.sum(5*sumsqr(delta),grid='control'))
ocp.add_objective(ocp.sum(sumsqr(final_X_x-p[0])+sumsqr(final_X_y-p[1])))
#ocp.add_objective(-ocp.sum(sumsqr(v),grid='control'))

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt', options)

ocp.method(MultipleShooting(N=Nhor, M=1, intg='rk'))

# Set initial value for states
current_X = vertcat(0,0,0,pi/4)
ocp.set_value(X_0,current_X)

#Specify the final value for the states and set
final_X = vertcat(final_X_x,final_X_y,0,pi/4)
ocp.subject_to(ocp.at_tf(X)==final_X)



# Set the initial value for the moving obstacle
current_move = vertcat(7,8)
ocp.set_value(p_move,current_move)

p_hist[0,:]=(current_move[0],current_move[1])

#%% Solve the problem and extract the solution, then plot the outputs over time
sol = ocp.solve()

# Extract solutions from the optimisation problem
t_sol, x_sol = sol.sample(x, grid='control')
t_sol, y_sol = sol.sample(y, grid='control')
t_sol, v_sol = sol.sample(v, grid='control')
t_sol, theta_sol = sol.sample(theta, grid='control')
t_sol, delta_sol = sol.sample(delta, grid='control')
t_sol, a_sol = sol.sample(a, grid='control')

obj_function = sol.value(ocp.objective)

#%% Set up the subplots

fig1,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6)

ax1.plot(t_sol,x_sol)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('x [m]')

ax2.plot(t_sol,y_sol)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y [m]')

ax3.plot(t_sol, v_sol)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('v [m/s]')

ax4.plot(t_sol,theta_sol)
ax4.set_xlabel('t [s]')
ax4.set_ylabel('theta [rad]')

ax5.plot(t_sol,a_sol)
ax5.set_xlabel('t [s]')
ax5.set_ylabel('a [m/s^2]')

ax6.plot(t_sol,delta_sol)
ax6.set_xlabel('t [s]')
ax6.set_ylabel('delta [rad]')
ax6.set_ylim(-0.5,0.5)

plt.subplots_adjust(hspace=1)

#%% Create an animated plot of the

fig2 = plt.figure()
ax7 = plt.subplot(1,1,1)

size_array = np.size(x_sol)
ts = np.linspace(0,2*pi,1000)

ax7.plot(p0[0]+r0*cos(ts),p0[1]+r0*sin(ts),'b')
ax7.plot(current_move[0]+r_move*cos(ts),current_move[1]+r_move*sin(ts),'r')

for k in range(size_array):

    ax7.plot(x_sol,y_sol,'ro', markersize = 10)

ax7.set_aspect('equal',adjustable='box')

#%% Implement the MPC portion of the control

# Get discretised dynamics as CasADi function to simulate the system
sim_system_dyn = ocp._method.discrete_system(ocp)

# Log the outputs from the first solve, in the first row
time_hist[0,:]    = t_sol
x_hist[0,:]       = x_sol
y_hist[0,:]       = y_sol
v_hist[0,:]       = v_sol
theta_hist[0,:]   = theta_sol
a_hist[0,:]       = a_sol
delta_hist[0,:]   = delta_sol

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)

    # Combine first control inputs
    current_U = vertcat(delta_sol[0], a_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]

    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    # Calculate the new position of the moving obstacle
    current_move[0]=current_move[0]+0.25
    current_move[1]=current_move[1]-0.25

    # Set the parameter p_move to the new current_move
    ocp.set_value(p_move, current_move)

    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing
    t_sol, x_sol     = sol.sample(x,     grid='control')
    t_sol, y_sol     = sol.sample(y,     grid='control')
    t_sol, v_sol     = sol.sample(v,     grid='control')
    t_sol, theta_sol = sol.sample(theta, grid='control')
    t_sol, delta_sol = sol.sample(delta, grid='control')
    t_sol, a_sol     = sol.sample(a,     grid='control')

    # TODO: The isn't upgrading to the absolute time at the moment, just the relative time
    time_hist[i+1,:]    = t_sol
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    v_hist[i+1,:]       = v_sol
    theta_hist[i+1,:]   = theta_sol
    delta_hist[i+1,:]   = delta_sol
    a_hist[i+1,:]       = a_sol

    #Save the position of the moving obstacle for plotting later
    p_hist[i+1,:] = [current_move[0],current_move[1]]

    #The previous solution makes a good initial guess for the next iteration, so put that here
    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(v, v_sol)
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(delta, delta_sol)
    ocp.set_initial(a, a_sol)

#%% Plot the output using the values from the history logs

# TODO: Plot the data using the _hist logs of data




