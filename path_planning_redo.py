#%% Import packages
from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
import pandas as pd


#%% Predefine functions
def path_predict(x_minus, x_now, y_minus, y_now, t_minus, t_now, dt):
    # Calculate the velocity from previous points
    v_x = (x_now - x_minus)/(t_now-t_minus)
    v_y = (y_now - y_minus) /(t_now-t_minus)
    #Use the velocity to predict the position at an instance in the future
    x_fut = x_now + v_x*(dt)
    y_fut = y_now + v_y*(dt)
    return x_fut, y_fut

#%% Setup the problem

L = 1.5 #Define what the length of the car is, as this will affect the turning circle.

Nsim    = 60           # how much samples to simulate
nx = 4                  # x, y, v, theta (angle bicycle)
nu = 2                  # a, delta (angle wheel)
Nhor = 30              #number of control intervals

#Initialise the matrices for logging variables
time_hist      = np.zeros((Nsim+1, Nhor+1))
x_hist         = np.zeros((Nsim+1, Nhor+1))
y_hist         = np.zeros((Nsim+1, Nhor+1))
v_hist         = np.zeros((Nsim+1, Nhor+1))
theta_hist     = np.zeros((Nsim+1, Nhor+1))
a_hist         = np.zeros((Nsim+1, Nhor+1))
delta_hist     = np.zeros((Nsim+1, Nhor+1))

p_hist = np.zeros((Nsim+1, 2))
p1_hist = np.zeros((Nsim+1, 2))
p2_hist = np.zeros((Nsim+1, 2))

#Initialise a matrix for plotting the position of the moving object

# Define the type of optimal control problem
ocp = Ocp(T=FreeTime(10.0)) #Freetime problem because otherwise it will reach the destination in the solution time

# Define states, represented as CasADi matrix expressions
x     = ocp.state()
y     = ocp.state()
v     = ocp.state()
theta = ocp.state()

slack1 = ocp.state()

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
ocp.subject_to(0 <= (v <= 3))
ocp.subject_to(-2 <= (x <= 12))
ocp.subject_to(-2 <= (y <= 12))

ocp.subject_to(0 <= slack1)

ocp.subject_to(-2 <= (a <= 2))
ocp.subject_to(-pi/6 <= (delta <= pi/6))

# Add a stationary objects along the path
p0 = vertcat(4,3.5)
r0 = 1

p = vertcat(x,y)
ocp.subject_to(sumsqr(p-p0)>=r0**2)

# Now input a moving obstacle
p_move = ocp.parameter(2)
r_move = 1 #Radius of the buffer around the vehicle.
v_move = 1 #Velocity of the moving obstacle [m/s]
theta_move = -pi/4 #angle of the the moving obstacle

# Add the moving obstacle in the future as a soft constraint.
p_move_p1 = ocp.parameter(2)
p_move_p2 = ocp.parameter(2)

# Add a constraint that the car cannot come close to the obstacle
ocp.subject_to(sumsqr(p-p_move)>=r_move**2)

final_X_x = 10
final_X_y = 10

# Objective functions
#ocp.add_objective(ocp.T)
ocp.add_objective(ocp.integral(0.5*sumsqr(a),grid='control'))
ocp.add_objective(ocp.integral(5*sumsqr(delta),grid='control'))
ocp.add_objective(ocp.integral(10*(sumsqr(sqrt((final_X_x-p[0])**2+(final_X_y-p[1])**2)))))

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
current_move = vertcat(5,8)
ocp.set_value(p_move,current_move)

#Set the initial "future" values of the moving obstacle
current_p1 = vertcat(5,8)
ocp.set_value(p_move_p1,current_p1)
current_p2 = vertcat(5,8)
ocp.set_value(p_move_p2,current_p2)

p_hist[0,:]=(current_move[0],current_move[1])
p1_hist[0,:]=(current_p1[0],current_p1[1])
p2_hist[0,:]=(current_p2[0],current_p2[1])

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
    current_move[0]=current_move[0]+(v_move*(t_sol[1]-t_sol[0]))
    current_move[1]=current_move[1]-(v_move*(t_sol[1]-t_sol[0]))

    # Set the parameter p_move to the new current_move
    ocp.set_value(p_move, current_move)

    if i > 1:
        # TODO: Implement the path prediction function and update the parameter
        [current_p1[0],current_p1[1]] = path_predict(p_hist[i-1,0], p_hist[i,0], p_hist[i-1,1], p_hist[i,1],\
                                                 time_hist[i-1,0], time_hist[i,0], t_sol[2])
        [current_p2[0],current_p2[1]] = path_predict(p_hist[i-1,0], p_hist[i,0], p_hist[i-1,1], p_hist[i,1],\
                                                 time_hist[i-1,0], time_hist[i,0], t_sol[3])

        # Set the parameter values for current p1 and current p2
        ocp.set_value(p_move_p1, current_p1)
        ocp.set_value(p_move_p2, current_p2)

    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing
    t_sol, x_sol     = sol.sample(x,     grid='control')
    t_sol, y_sol     = sol.sample(y,     grid='control')
    t_sol, v_sol     = sol.sample(v,     grid='control')
    t_sol, theta_sol = sol.sample(theta, grid='control')
    t_sol, delta_sol = sol.sample(delta, grid='control')
    t_sol, a_sol     = sol.sample(a,     grid='control')

    t_begin = time_hist[i,1]*np.ones(Nhor+1)

    time_hist[i+1,:]    = t_sol + t_begin
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    v_hist[i+1,:]       = v_sol
    theta_hist[i+1,:]   = theta_sol
    delta_hist[i+1,:]   = delta_sol
    a_hist[i+1,:]       = a_sol

    #Save the position of the moving obstacle for plotting later
    p_hist[i+1,:] = [current_move[0],current_move[1]]
    if i <= 1:
        p1_hist[i+1,:] = [current_move[0],current_move[1]]
        p2_hist[i+1,:] = [current_move[0],current_move[1]]
    else:
        p1_hist[i+1, :] = [current_p1[0], current_p1[1]]
        p2_hist[i+1, :] = [current_p2[0], current_p2[1]]

    #The previous solution makes a good initial guess for the next iteration, so put that here
    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(v, v_sol)
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(delta, delta_sol)
    ocp.set_initial(a, a_sol)

#%% Plot the output as an animation

plt.ion() #turn the interactive plot on; comment out this line if you don't want an interactive plot
if plt.isinteractive():
  fig3, ax8 = plt.subplots(1, 1)
  plt.ion()
  ax8.set_xlabel("X [m]")
  ax8.set_ylabel("Y [m]")
  ax8.set_xlim([-1,12])
  ax8.set_ylim([-1,12])
  ax8.set_aspect('equal', adjustable='box')
  ts = np.linspace(0, 2 * pi, 1000)
  ax8.plot(p0[0] + r0 * cos(ts), p0[1] + r0 * sin(ts), 'r-')
  for k in range(len(x_hist)):
      cart_x_pos_k    = x_hist[k,0]
      cart_y_pos_k    = y_hist[k,0]
      theta_k         = theta_hist[k,0]

      p0_x_k = p_hist[k,0]
      p0_y_k = p_hist[k,1]

      #Set the color scale
      color_point_k= 3*[0.95*(1-float(k)/Nsim)] #Color scale for the car
      scale = [0.5,0.5,0.5] #Set how much lighter you want the arrows to be
      color_car_k = np.multiply(scale,color_point_k)

      # Plot the square to represent the car
      ax8.plot(cart_x_pos_k, cart_y_pos_k, marker = (4,0,theta_k*(180/pi)), markersize = 10, color = color_car_k)
      # Plot the triangle to show the orientation of the car
      orientation_x_k = cart_x_pos_k+cos(theta_k)
      orientation_y_k = cart_y_pos_k+sin(theta_k)
      ax8.plot(orientation_x_k, orientation_y_k, marker = (3,0,-90+theta_k*(180/pi)), markersize = 5, color = color_point_k)
      # Plot the line connecting the car and its orientation.
      ax8.plot([cart_x_pos_k,orientation_x_k],[cart_y_pos_k,orientation_y_k],"-", linewidth = 1.5, color = color_point_k)
      # Plot the obstacle
      ax8.plot(p_hist[k-1,0] + r0 * cos(ts), p_hist[k-1,1] + r0 * sin(ts), 'w-')
      ax8.plot(p0_x_k + r0 * cos(ts), p0_y_k + r0 * sin(ts), 'r-')

      plt.pause(0.1/10)
plt.show(block=False)
plt.ioff()

#%% Pandas dataframe

#Add the first row of time_hist, x_hist, and y_hist into their own pd.DataFrame
t_data = pd.DataFrame(time_hist)
x_data = pd.DataFrame(x_hist)
y_data = pd.DataFrame(y_hist)
p_x_data = pd.DataFrame(p_hist[:,0])
p_y_data = pd.DataFrame(p_hist[:,1])
p1_x_data = pd.DataFrame(p1_hist[:,0])
p1_y_data = pd.DataFrame(p1_hist[:,1])
p2_x_data = pd.DataFrame(p2_hist[:,0])
p2_y_data = pd.DataFrame(p2_hist[:,1])

t_x_y = pd.DataFrame(zip(t_data.loc[:,0],x_data.loc[:,0],y_data.loc[:,0],p_x_data.loc[:,0],p_y_data.loc[:,0]))
t_x_y = t_x_y.rename(columns={0: "Time",1: "X_Pos",2: "Y_Pos",3: "P_X_Pos",4: "P_Y_Pos" })
print(t_x_y[0:Nsim+1])

p_x_y = pd.DataFrame(zip(t_data.loc[:,0],p_x_data.loc[:,0],p_y_data.loc[:,0],p1_x_data.loc[:,0],\
                         p1_y_data.loc[:,0],p2_x_data.loc[:,0],p2_y_data.loc[:,0]))
p_x_y = p_x_y.rename(columns={0: "Time",1: "PX_Pos",2: "PY_Pos",3: "PX1_Pos",4: "PY1_Pos", 5: "PX2_Pos", 6: "PY2_Pos"})
print(p_x_y[0:Nsim+1])