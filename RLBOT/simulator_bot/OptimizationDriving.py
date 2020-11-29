import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
import gekko
from gekko import GEKKO
import csv
from mpl_toolkits.mplot3d import Axes3D

# This optimizer will minimize time to get to desired end points

# Next I will need to add to the cost function the error between the rocket and the desired set point

class Optimizer():
    def __init__(self):
#################GROUND DRIVING OPTIMIZER SETTTINGS##############
        self.d = GEKKO(remote=False) # Driving on ground optimizer

        ntd = 21

        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        self.d.options.NODES = 2
        self.d.options.SOLVER = 3
        self.d.options.IMODE = 6# MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.d.options.MAX_ITER = 800
        self.d.options.MV_TYPE = 0
        self.d.options.DIAGLEVEL = 0
        # self.d.options.OTOL = 1

        # final time for driving optimizer
        self.tf = self.d.FV(value=1.0,lb=1,ub=100.0)

        # allow gekko to change the tf value
        self.tf.STATUS = 1

        # Scaled time for Rocket league to get proper time

        # Acceleration variable
        self.a = self.d.MV(fixed_initial=False, lb = -1, ub = 1)
        self.a.STATUS = 1
        # self.a.DCOST = 1e-1

        # # Boost variable, its integer type since it can only be on or off
        # self.u_thrust_d = self.d.MV(fixed_initial=False,lb=0,ub=1) #Manipulated variable integer type
        # self.u_thrust_d.STATUS = 1
        # self.u_thrust_d.DCOST = 1e-5
        #
        # # Throttle value, this can vary smoothly between 0-1
        # self.u_throttle_d = self.d.MV(value = 1, lb = 0.02, ub = 1)
        # self.u_throttle_d.STATUS = 1
        # self.u_throttle_d.DCOST = 1e-5

        # Turning input value also smooth
        self.u_turning_d = self.d.MV(lb = -1, ub = 1)
        self.u_turning_d.STATUS = 1
        # self.u_turning_d.DCOST = 1e-1

        # end time variables to multiply u2 by to get total value of integral
        # Time vector length is nt1 and nt2
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final = self.d.Param(value = self.p_d)

        # integral over time for u_pitch^2
        # self.u2_pitch = self.d.Var(value=0)
        # self.d.Equation(self.u2.dt() == 0.5*self.u_pitch**2)

    def optimize2D(self, si, sf, vi, vf, ri, omegai): #these are 1x2 vectors s or v [x, z]

        # variables intial conditions are placed here
            # Position and Velocity in 2d
        self.sx = self.d.Var(value=si[0], lb=-4096, ub=4096) #x position
        self.sy = self.d.Var(value=si[1], lb=-5120, ub=5120) #y position

            # Pitch rotation and angular velocity
        self.yaw = self.d.Var(value = ri) #orientation yaw angle
        self.omega = self.d.Var(fixed_initial=False, value=omegai, lb=-5.5, ub=5.5) #angular velocity
        # self.v_mag = self.d.Intermediate(self.d.sqrt((self.vx**2) + (self.vy**2)))

        self.v_mag = self.d.Var(value = vi, ub = 2300, lb =0)
        # self.v_inter = self.d.Var(value = vi)
        self.vx = self.d.Var(value=(np.cos(ri) * vi)) #x velocity
        self.vy = self.d.Var(value=(np.sin(ri) * vi)) #y velocity

        # curvature as a polynomical approximation
        self.curvature = self.d.Intermediate(0.0069 - ((7.67e-6) * self.v_mag) + ((4.35e-9)*(self.v_mag**2)) - ((1.48e-12) * (self.v_mag**3)) + ((2.37e-16) * (self.v_mag**4)))

        # velocity dependent acceleration modeled as a cspline
        ba=991.666 #acceleration due to boost
        kv = np.array([0, 1410, 2300]) #velocity input
        ka = np.array([1600+ba, 0+ba, 0+ba]) #acceleration ouput
        kv_fine = np.linspace(0, 2300, 10) # Finer detail so cspline can fit
        ka_fine = np.interp(kv_fine, kv, ka) # finder detail acceleration
        self.kv_fine = kv_fine
        self.ka_fine = ka_fine

        self.throttle_acceleration = self.d.Var(fixed_initial=False)
        self.d.cspline(self.v_mag, self.throttle_acceleration, kv_fine, ka_fine)

# Differental equations
        # self.d.Equation(self.v_inter == self.v_mag)
        self.d.Equation(self.v_mag.dt()/self.tf == (self.a * self.throttle_acceleration))# + (self.u_thrust_d * 991.666))

        self.d.Equation(self.vx == (self.v_mag * self.d.cos(self.yaw)))
        self.d.Equation(self.vy == (self.v_mag * self.d.sin(self.yaw)))

        self.d.Equation(self.sx.dt()/self.tf == self.vx)
        self.d.Equation(self.sy.dt()/self.tf == self.vy)

        self.d.Equation(self.omega/self.tf == ((self.u_turning_d) * (1/self.curvature) * self.v_mag))
        self.d.Equation(self.yaw.dt()/self.tf == self.omega)
        # self.d.fix(self.sz, pos = len(self.d.time) - 1, val = 1000)


        #Soft constraints for the end point
        # Uncomment these 4 objective functions to get a simlple end point optimization
        #sf[1] is y position @ final time etc...
        self.d.Obj(self.final*1e4*((self.sy-sf[1])**2)) # Soft constraints
        # self.d.Obj(self.final*1e3*(self.vz-vf[1])**2)
        self.d.Obj(self.final*1e4*((self.sx-sf[0])**2)) # Soft constraints
        # self.d.Obj(self.final*1e3*(self.vx-vf[0])**2)


        #Objective function to minimize time
        self.d.Obj(1e1*self.tf)

        #Objective functions to follow trajectory
        # self.d.Obj(self.final * (self.errorx **2) * 1e3)

        # self.d.Obj(self.final*1e3*(self.sx-traj_sx)**2) # Soft constraints
        # self.d.Obj(self.errorz)
        # self.d.Obj(( self.all * (self.sx - trajectory_sx) **2) * 1e3)
        # self.d.Obj(((self.sz - trajectory_sz)**2) * 1e3)

        # minimize thrust used
        # self.d.Obj(self.u2*self.final*1e3)

        # minimize torque used
        # self.d.Obj(self.u2_pitch*self.final)

        #solve
        # self.d.solve('http://127.0.0.1') # Solve with local apmonitor server
        self.d.solve(disp=True)

        # NOTE: another data structure type or class here for optimal control vectors
        # Maybe it should have some methods to also make it easier to parse through the control vector etc...
        # print('time', np.multiply(self.d.time, self.tf.value[0]))
        # time.sleep(3)

        self.ts = np.multiply(self.d.time, self.tf.value[0])

        return self.a, self.u_turning_d, self.ts

    def getTrajectoryData(self):
        return [self.ts, self.sx, self.sy, self.vx, self.vy, self.yaw, self.omega]

    def getInputData(self):
        return [self.ts, self.a]




# Main Code

opt = Optimizer()

s_ti = [0,0]
v_ti = 2000
s_tf = [-2000,-1000]
v_tf = [00.00, 00.0]
r_ti = np.pi/2 # inital orientation of the car
omega_ti = 0.0 # initial angular velocity of car

acceleration, turning, t_star = opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti)

# print('u', acceleration.value)
# print('tf', opt.tf.value)
# print('tf', opt.tf.value[0])
print('vx', opt.vx.value)
print('vy', opt.vy.value)
print('a', opt.a.value)

ts = opt.d.time * opt.tf.value[0]
# plot results
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
# plt.subplot(2, 1, 1)
Axes3D.plot(ax, opt.sx.value, opt.sy.value, ts, c='r', marker ='o')
plt.ylim(-2500, 2500)
plt.xlim(-2500, 2500)
plt.ylabel('Position y')
plt.xlabel('Position x')
ax.set_zlabel('time')

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
# plt.subplot(2, 1, 1)
Axes3D.plot(ax, opt.vx.value, opt.vy.value, ts, c='r', marker ='o')
plt.ylim(-2500, 2500)
plt.xlim(-2500, 2500)
plt.ylabel('velocity y')
plt.xlabel('Velocity x')
ax.set_zlabel('time')

plt.figure(1)
num_plots = 4
plt.subplot(num_plots,1,1)
plt.plot(ts, opt.a, 'r-')
plt.ylabel('acceleration')

plt.subplot(num_plots,1,2)
plt.plot(ts, np.multiply(opt.yaw, 1/math.pi), 'r-')
plt.ylabel('yaw orientation')

plt.subplot(num_plots, 1, 3)
plt.plot(ts, opt.v_mag, 'b-')
plt.ylabel('vmag')

plt.subplot(num_plots, 1, 4)
plt.plot(opt.kv_fine, opt.ka_fine, 'r-')
plt.ylabel('kv vs ka')
# plt.figure(1)

plt.show()
