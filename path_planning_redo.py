# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:42:02 2020

@author: JiNzo
"""


from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

# Bicycle model
L = 1.5 #Define what the length of the car is, as this will affect the turning circle.


Nsim    = 30            # how much samples to simulate
nx = 4                  # x, y, v, theta (angle bicycle)
nu = 2                  # a, delta (angle wheel)
Tf = 5                  # Control horizon [s]
Nhor = 10              #number of control intervals
dt = Tf/Nhor            #sample time

time_hist      = np.zeros((Nsim+1, Nhor+1))
x_hist         = np.zeros((Nsim+1, Nhor+1))
y_hist         = np.zeros((Nsim+1, Nhor+1))
theta_hist     = np.zeros((Nsim+1, Nhor+1))
delta_hist     = np.zeros((Nsim+1, Nhor+1))
V_hist         = np.zeros((Nsim+1, Nhor+1))

ocp = Ocp(T=FreeTime(10.0))

# Define states
x     = ocp.state()
y     = ocp.state()
v     = ocp.state()
theta = ocp.state()

# Define controls
delta = ocp.control()
a     = ocp.control()

# Specify ODE
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
ocp.subject_to(0 <= (v <= 10))
ocp.subject_to(-2 <= (x <= 12))
ocp.subject_to(-2 <= (y <= 12))

ocp.subject_to(-10 <= (a <= 10))
ocp.subject_to(-pi/6 <= (delta <= pi/6))

# Objective functions
ocp.add_objective(sumsqr(ocp.T))
ocp.add_objective(ocp.sum(sumsqr(a),grid='control'))
ocp.add_objective(ocp.sum(sumsqr(delta),grid='control'))
ocp.add_objective(-ocp.sum(sumsqr(v),grid='control'))

# Pick a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt', options)

ocp.method(MultipleShooting(N=Nhor, M=1, intg='rk'))

# Set initial value for states
current_X = vertcat(0,0,0,pi/4)
ocp.set_value(X_0,current_X)

#Specify the end
final_X = vertcat(10,10,0,pi/4)
ocp.subject_to(ocp.at_tf(X)==final_X)

sol = ocp.solve()
































