from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

#%% Definition of Car Length and Initial Conditions

L = 1.5 #Define what the length of the car is, as this will affect the turning circle.

# Define the vector that will describe the current state of the system in the simulation
nx = 5 #Define number of states of the system to go into the vector
nu = 2 #Define the number of control parameters for the system
Tf = 5 #Control horizon [s]
Nhor = 40 #number of control intervals
dt = Tf/Nhor #sample time

#%% Define inital conditions

current_X = vertcat(0,0,0,0,0) # x,y,theta,v,delta
final_X = vertcat(10,10,pi/2,0,0) # x,y,theta,v,delta

path_length_new = vertcat(0) #Initialise the path length parameter to be zero


#%% Define the simulation length and define empty arrays
Nsim = int(5*Tf/dt) #How many simulation steps to run, keep in mind that this should be determined by dt.
add_noise = False
add_disturbance = False

# Create a log of the variables
x_history = np.zeros(Nsim+1)
y_history = np.zeros(Nsim+1)
theta_history = np.zeros(Nsim+1)
v_history = np.zeros(Nsim+1)
delta_history = np.zeros(Nsim+1)

a_history = np.zeros(Nsim)
ddelta_history = np.zeros(Nsim)

p0_xhistory = np.zeros(Nsim+1)
p0_yhistory = np.zeros(Nsim+1)

#%% Define format of the optimal control problem

ocp = Ocp(T = FreeTime(3))

#%% Define the states in a 2-dimensional plane
x = ocp.state()
y = ocp.state()
theta = ocp.state()
v = ocp.state()
delta = ocp.state()


#%% Define the two control variables.
ddelta = ocp.control(order = 0)
a = ocp.control(order = 0)

# Define parameter
X_0 = ocp.parameter(nx)

#%% Path length parameter
## Define a parameter for the total path length
#path_length = ocp.parameter(1)


#%% Specify ODE's
ocp.set_der(x, v*cos(theta))
ocp.set_der(y, v*sin(theta))
ocp.set_der(theta, v*(tan(delta)/L))
ocp.set_der(v,a)
ocp.set_der(delta,ddelta)

#%% Minimal time objective, with a path constraint objective (for now)
ocp.add_objective(ocp.T) #objective purely focused on time

#%% Specify constraints
ocp.subject_to(-3 <= v)
ocp.subject_to(v <= 3)
ocp.subject_to(-(pi/4) <= delta)
ocp.subject_to(delta <= (pi/4))
ocp.subject_to(-2 <= a)
ocp.subject_to(a <= 2)
ocp.subject_to(-(pi/6) <= ddelta)
ocp.subject_to(ddelta <= (pi/6))
ocp.subject_to(0 <= ocp.T)

#%% Specify state bounds
ocp.subject_to(-2 <= x)
ocp.subject_to(x <= 15)
ocp.subject_to(-2 <= y)
ocp.subject_to(y <= 15)

#%% Initial constraints
X = vertcat(x,y,theta,v,delta)
ocp.subject_to(ocp.at_t0(X)==X_0) #Starting x position

#%% Final constraint
ocp.subject_to(ocp.at_tf(X)==final_X)

#%% Add an object and an object constraint
p0 = vertcat(6,6) #This marks the centerpoint of the obstacle.
r0 = 0.5 #This marks the radius of the obstacle that you are trying to avoid.
angle = -(pi/6) #This marks the straight line the obstacle will move along

p1 = vertcat(8,1)
r1 = 0.5

p2 = vertcat(3,6)
r2 = 0.5

p = vertcat(x,y)
ocp.subject_to(sumsqr(p-p0)>=1.2*(r0**2)) #sumsqr calculates the sum of squares sum_ij (x_ij**2)
ocp.subject_to(sumsqr(p-p1)>=1.2*(r1**2))
ocp.subject_to(sumsqr(p-p2)>=1.2*(r2**2))

#%% Specify a solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True #using expand makes the solver faster
options["print_time"] = False
ocp.solver('ipopt', options)

#Concretely specify the method
ocp.method(MultipleShooting(N=Nhor, M=1, intg='rk'))

#%% ocp.set_initial sets initial guesses for the solver to utilise
ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, pi/2)
ocp.set_initial(v, 1)
ocp.set_initial(a, 1)
ocp.set_initial(delta, 0)
ocp.set_initial(ddelta, 0)

# -----------------------------------------------------------------------------------------------
## Solve the ocp for the first time
# -----------------------------------------------------------------------------------------

# This section sets the values of the parameters to either the initial conditions prescribed
ocp.set_value(X_0,current_X)
#ocp.set_value(path_length,path_length_new)
# Solve
try:
    sol = ocp.solve()
except:
    ocp.show_infeasibilities(1e-6)

#Get the discretised dynamics as CasADi function
sim_car_dyn = ocp._method.discrete_system(ocp)

x_history[0] = current_X[0]
y_history[0] = current_X[1]
theta_history[0] = current_X[2]
v_history[0] = current_X[3]
delta_history[0] = current_X[4]

p0_xhistory[0] = p0[0]
p0_yhistory[0] = p0[0]

# ------------------------------------------------------------------------------------------------
## Simulate solving with the updated state several times
index = 0
for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)

    if fabs(final_X.full()[0,0]-current_X.full()[0,0]) <= 2 and fabs(final_X.full()[1,0]-current_X.full()[1,0]) <= 2:
        print('Within 2m in each direction, switch to path planning')
        break

    #Get the solution from sol
    tsa, x_sol = sol.sample(x, grid='control')
    tsa, y_sol = sol.sample(y, grid='control')
    tsa, theta_sol = sol.sample(theta, grid='control')
    tsa, v_sol = sol.sample(v, grid='control')
    tsa, delta_sol = sol.sample(delta, grid='control')

    tsa, ddelta_sol = sol.sample(ddelta, grid='control')
    tsa, a_sol = sol.sample(a, grid='control')
    #Simulate dynamics
    Usol = vertcat(ddelta_sol[0], a_sol[0])
    current_X = sim_car_dyn(x0=current_X, u=Usol, T = .99*dt)["xf"] #only using delta_sol


    #Set the parameter X0 to the new current_X
    ocp.set_value(X_0,current_X[:5])

    #update the position of the moving obstacle

    p0[0] = p0[0] + cos(angle)/25
    p0[1] = p0[1] + sin(angle)/25

    ocp.set_value(p0,p0)

    #Solve the optimisation problem
    try:
        sol = ocp.solve()
    except:
        infeasibilities = ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution

    #Log data for post-processing
    x_history[i+1] =current_X[0].full()
    y_history[i+1] =current_X[1].full()
    theta_history[i+1] =current_X[2].full()
    v_history[i+1] =current_X[3].full()
    delta_history[i+1] =current_X[4].full()
    ddelta_history[i] =Usol[0]
    a_history[i] =Usol[1]

    p0_xhistory[i+1] = p0[0]
    p0_yhistory[i+1] = p0[1]

    index += 1
#------------------------------------------------------------------------
# Now that the car is close to the end run a short path planning for the last part
#------------------------------------------------------------------------
try:
    sol = ocp.solve() # Solve it one last time as path planning for a smooth finish.
except:
    infeasibilities = ocp.show_infeasibilities(1e-6)

tsa, x_sol = sol.sample(x, grid='control')
tsa, y_sol = sol.sample(y, grid='control')
tsa, theta_sol = sol.sample(theta, grid='control')
tsa, v_sol = sol.sample(v, grid='control')
tsa, delta_sol = sol.sample(delta, grid='control')

tsa, ddelta_sol = sol.sample(ddelta, grid='control')
tsa, a_sol = sol.sample(a, grid='control')


## Log the data
counter = 0
while counter < Nhor+1:
    x_history[index+1] = x_sol[counter]
    y_history[index+1] = y_sol[counter]
    theta_history[index + 1] = theta_sol[counter]
    v_history[index + 1] = v_sol[counter]
    delta_history[index + 1] = delta_sol[counter]
    ddelta_history[index + 1] = ddelta_sol[counter]
    a_history[index + 1] = a_sol[counter]

    p0[0] = p0[0]# + cos(angle) / 25
    p0[1] = p0[1]# + sin(angle) / 25
    p0_xhistory[index+1] = p0[0]
    p0_yhistory[index+1] = p0[1]

    counter+=1
    index+=1
#------------------------------------------------------------------------
#Plot the outputs and states
#------------------------------------------------------------------------

time_sim = np.linspace(0, dt*index, index+1)

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1)
ax1.plot(time_sim, x_history[0:index+1],label='x_history')
ax1.plot(time_sim, y_history[0:index+1], label='y_history')
ax1.plot(time_sim, v_history[0:index+1], label='v_history')
ax1.plot(time_sim[:-1],a_history[0:index],'b.', label='a_history')
ax1.set_xlabel('Time [s]')
ax1.set_title('Position, Speed, Velocity')
ax1.legend()

ax2.plot(time_sim, delta_history[0:index+1], label='delta_history')
ax2.plot(time_sim, theta_history[0:index+1], label='theta_history')
ax2.plot(time_sim[:-1],ddelta_history[0:index],'b.', label='ddelta_history')
ax2.set_title('Delta, Theta, dDelta')
ax2.set_xlabel('Time [s]')
ax2.legend()

fig.tight_layout()

#------------------------------------------------------------------------
#Plot the outputs and states
#------------------------------------------------------------------------

plt.ion() #turn the interactive plot on; comment out this line if you don't want an interactive plot
if plt.isinteractive():
  fig2, ax3 = plt.subplots(1, 1)
  plt.ion()
  ax3.set_xlabel("X [m]")
  ax3.set_ylabel("Y [m]")
  ax3.set_xlim([-1,12])
  ax3.set_ylim([-1,12])
  ts = np.linspace(0, 2 * pi, 1000)
  ax3.plot(p1[0] + r1 * cos(ts), p1[1] + r1 * sin(ts), 'r-')
  ax3.plot(p2[0] + r2 * cos(ts), p2[1] + r2 * sin(ts), 'r-')
  for k in range(index+1):
      cart_x_pos_k    = x_history[k]
      cart_y_pos_k    = y_history[k]
      theta_k         = theta_history[k]
      p0_x_k = p0_xhistory[k]
      p0_y_k = p0_yhistory[k]

      #Set the color scale
      color_point_k= 3*[0.95*(1-float(k)/Nsim)] #Color scale for the car
      scale = [0.5,0.5,0.5] #Set how much lighter you want the arrows to be
      color_car_k = np.multiply(scale,color_point_k)

      # Plot the square to represent the car
      ax3.plot(cart_x_pos_k, cart_y_pos_k, marker = (4,0,theta_k*(180/pi)), markersize = 10, color = color_car_k)
      # Plot the triangle to show the orientation of the car
      orientation_x_k = cart_x_pos_k+cos(theta_k)
      orientation_y_k = cart_y_pos_k+sin(theta_k)
      ax3.plot(orientation_x_k, orientation_y_k, marker = (3,0,-90+theta_k*(180/pi)), markersize = 5, color = color_point_k)
      # Plot the line connecting the car and its orientation.
      ax3.plot([cart_x_pos_k,orientation_x_k],[cart_y_pos_k,orientation_y_k],"-", linewidth = 1.5, color = color_point_k)
      # Plot the obstacle
      ax3.plot(p0_xhistory[k-1] + r0 * cos(ts), p0_yhistory[k-1] + r0 * sin(ts), 'w-')
      ax3.plot(p0_x_k + r0 * cos(ts), p0_y_k + r0 * sin(ts), 'r-')

      plt.pause(dt/8)
plt.show(block=True)

