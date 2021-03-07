import numpy as np
import matplotlib.pyplot as plt
import math
import gekko
from gekko import GEKKO
import csv
from mpl_toolkits.mplot3d import Axes3D

class Optimizer():
    def __init__(self):
#################GROUND DRIVING OPTIMIZER SETTTINGS##############
        self.d = GEKKO(remote=False) # Driving on ground optimizer

        ntd = 21

        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        self.d.options.NODES = 3
        self.d.options.SOLVER = 3
        self.d.options.IMODE = 6# MPC mode
        self.d.options.MAX_ITER = 800
        self.d.options.MV_TYPE = 0
        self.d.options.DIAGLEVEL = 0
        self.d.options.OTOL = 0.1
        self.d.options.RTOL = 0.0001

        # final time for driving optimizer
        self.tf = self.d.FV(value=1.0,lb=0.1,ub=10.0, name='tf')

        # allow gekko to change the tf value
        self.tf.STATUS = 1
        
        # time variable
        self.t = self.d.Var(value=0)
        self.d.Equation(self.t.dt()/self.tf == 1)

        # Acceleration variable
        self.a = self.d.MV(fixed_initial=False, lb = 0, ub = 1, name='a')
        self.a.STATUS = 1
        
        # Jumping integer varaibles and equations
        self.u_jump = self.d.MV(fixed_initial=False, lb=0, ub=1)
        self.u_jump.STATUS = 1
        self.jump_hist = self.d.Var(value=0, name='jump_hist', lb=0, ub=1)
        self.d.Equation(self.jump_hist.dt() == self.u_jump*(ntd-1))
        # self.d.Equation(1.0 >= self.jump_hist)

        # pitch input throttle (rotation of system)
        self.u_p = self.d.MV(fixed_initial=False, lb = -1, ub=1)
        self.u_p.STATUS = 1

        # Final variable that allows you to set an objective function considering only final state
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final = self.d.Param(value = self.p_d, name='final')

        # Model constants and parameters
        self.Dp = self.d.Const(value = 2.7982, name='D_pitch')
        self.Tp = self.d.Const(value = 12.146, name='T_pitch')
        self.pi = self.d.Const(value = 3.14159, name='pi')
        self.g = self.d.Const(value = 650, name='Fg')
        self.Db = self.d.Const(value = -0.0305)
        self.jump_magnitude = self.d.Param(value = 3000, name = 'jump_mag')

    def optimize2D(self, si, sf, vi, vf, ri, omegai, bsi, bvi): #these are 1x2 vectors s or v [x, z]

        # variables and intial conditions 
        # Position in 2d
        self.sx = self.d.Var(value=si[0], lb=-4096, ub=4096, name='x') #x position
        # self.sy = self.d.Var(value=si[1], lb=-5120, ub=5120, name='y') #y position
        self.sz = self.d.Var(value = si[1])
        # Ball position
        self.bsx = self.d.Var(value=bsi[0], name='bsx')
        self.bsz = self.d.Var(value=bsi[1], name = 'bsz')

        # Pitch rotation and angular velocity
        self.pitch = self.d.Var(value = ri, name='pitch', lb=-1*self.pi, ub=self.pi)
        self.pitch_dot = self.d.Var(fixed_initial=False, name='pitch_dot')

        # Velocity in 2D
        self.v_mag = self.d.Var(value=(vi), name='v_mag')
        self.vx = self.d.Var(value=np.cos(ri) * vi, name='vx') #x velocity
        # self.vy = self.d.Var(value=(np.sin(ri) * vi), name='vy') #y velocity
        self.vz = self.d.Var(value = (np.sin(ri) * vi), name='vz')
        # Ball velocity
        self.bvx = self.d.Var(value = bvi[0])
        self.bvz = self.d.Var(value = bvi[1])

## Non-linear state dependent dynamics descired as csplines.
        #curvature vs vel as a cubic spline for driving state
        cur = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_cur = np.array([0,500,1000,1500,1750,2300])
        v_cur_fine = np.linspace(0,2300,100)
        cur_fine = np.interp(v_cur_fine, v_cur, cur)
        self.curvature = self.d.Var(name='curvature')
        self.d.cspline(self.v_mag, self.curvature, v_cur_fine, cur_fine)

        # throttle vs vel as cubic spline for driving state
        ba=991.666 #Boost acceleration magnitude
        kv = np.array([0, 1410, 2300]) #velocity input
        ka = np.array([1600+ba, 0+ba, 0+ba]) #acceleration ouput
        kv_fine = np.linspace(0, 2300, 100) # Higher resolution
        ka_fine = np.interp(kv_fine, kv, ka) # Piecewise linear high resolution of ka
        self.throttle_acceleration = self.d.Var(fixed_initial=False, name='throttle_accel')
        self.d.cspline(self.v_mag, self.throttle_acceleration, kv_fine, ka_fine)

# Differental equations
    # Velocity diff eqs
        self.d.Equation(self.vx.dt()/self.tf == (self.a*ba * self.d.cos(self.pitch)*self.jump_hist) + (self.a * self.throttle_acceleration * (1-self.jump_hist)) + (self.u_jump * self.jump_magnitude * self.d.cos(self.pitch + np.pi/2)))
        self.d.Equation(self.vz.dt()/self.tf == (self.a*ba * self.d.sin(self.pitch)*self.jump_hist) - (self.g * (1-self.jump_hist)) + (self.u_jump * self.jump_magnitude * self.d.sin(self.pitch + np.pi/2)))
        self.d.Equation(self.v_mag == self.d.sqrt((self.vx*self.vx) + (self.vz*self.vz)))
        self.d.Equation(2300 >= self.v_mag)
    # Ball Diff eqs
        self.d.Equation(self.bvx.dt()/self.tf == self.Db*self.bvx)
        self.d.Equation(self.bvz.dt()/self.tf == -1*self.g + self.Db*self.bvz)

    # Position diff eqs
        self.d.Equation(self.sx.dt()/self.tf == self.vx)
        # self.d.Equation(self.sy.dt()/self.tf == self.vy)
        self.d.Equation(self.sz.dt()/self.tf == self.vz)
        self.d.Equation(self.bsx.dt()/self.tf == self.bvx)
        self.d.Equation(self.bsz.dt()/self.tf == self.bvz)

    # Orientation diff eqs
        self.d.Equation(self.pitch_dot.dt()/self.tf == ((self.Tp * self.u_p) + (self.Dp * self.pitch_dot * (1 - self.d.abs2(self.u_p)))) * self.jump_hist)
        self.d.Equation(self.pitch.dt()/self.tf == self.pitch_dot)


# Objective functions
        # Final Position Objectives
        # self.d.Minimize(self.final*1e2*((self.sz-sf[1])**2)) # z final position objective
        # self.d.Minimize(self.final*1e2*((self.sx-sf[0])**2)) # x final position objective
        # Final Velocity Objectives
        # self.d.Obj(self.final*1e3*(self.vz-vf[1])**2)
        # self.d.Obj(self.final*1e3*(self.vx-vf[0])**2)
        # Final position to ball objectives
        self.d.Minimize(self.final*1e4*(self.sx - self.bsx)**2)
        self.d.Minimize(self.final*1e4*(self.sz - self.bsz)**2)

        # Minimum Time Objective
        self.d.Minimize(1e2*self.tf)

        #solve
        # self.d.solve('http://127.0.0.1') # Solve with local apmonitor server
        self.d.open_folder()


    def solve_continuous(self):
        self.d.solve(disp=True)
    
    def solve_integer(self):
        # change variables to integers
        # self.u_jump = 
        self.d.solve(disp=True)
        

    def getTrajectoryData(self):
        return [self.ts, self.sx, self.sz, self.vx, self.vz, self.pitch, self.pitch_dot]

    def getInputData(self):
        return [self.ts, self.a]

# Main Code

opt = Optimizer()

s_ti = [-2000,0]
v_ti = 200
s_tf = [0,4000]
v_tf = [00.00, 00.0]
r_ti = 0 # inital orientation of the car
omega_ti = 0.0 # initial angular velocity of car

bsi = [2000,3000]
bvi = [-400, 0]

opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti, bsi, bvi)
opt.solve_continuous()

# Printing stuff
# print('u', acceleration.value)
# print('tf', opt.tf.value)
# print('tf', opt.tf.value[0])
# print('u jump', opt.jump)
# for i in opt.u_jump: print(i.value)
print('u_jump', opt.u_jump.value)
print('jump his', opt.jump_hist.value)
print('v_mag', opt.v_mag.value)
print('a', opt.a.value)

# Plotting stuff

ts = opt.d.time * opt.tf.value[0]
t_max = opt.tf.value[0]
x_max = np.max(opt.sx.value)
vx_max = np.max(opt.vx.value)
z_max = np.max(opt.sz.value)
vz_max = np.max(opt.vz.value)
# plot results
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
# plt.subplot(2, 1, 1)
Axes3D.plot(ax, opt.sx.value, ts, opt.sz.value, c='r', marker ='o')
Axes3D.plot(ax, opt.bsx.value, ts, opt.bsz.value, c='b', marker = '*')
plt.ylim(0, t_max)
plt.xlim(0, x_max)
plt.ylabel('time')
plt.xlabel('Position x')
ax.set_zlabel('position z')

n=5 #num plots
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
# plt.subplot(2, 1, 1)
Axes3D.plot(ax, opt.vx.value, ts, opt.vz.value,  c='r', marker ='o')
Axes3D.plot(ax, opt.bvx.value, ts, opt.bvz.value, c='b', marker = '*')
plt.ylim(0, t_max)
plt.xlim(-1*vx_max, vx_max)
# plt.zlim(0, 2000)
plt.ylabel('time')
plt.xlabel('Velocity x')
ax.set_zlabel('vz')

plt.figure(1)
plt.subplot(n,1,1)
plt.plot(ts, opt.a, 'r-')
plt.ylabel('acceleration')

plt.subplot(n,1,2)
plt.plot(ts, np.multiply(opt.pitch, 1/math.pi), 'r-')
plt.ylabel('pitch orientation')

plt.subplot(n, 1, 3)
plt.plot(ts, opt.v_mag, 'b-')
plt.ylabel('vmag')

plt.subplot(n, 1, 4)
plt.plot(ts, opt.u_p, 'b-')
plt.ylabel('u_p')

plt.subplot(n, 1, 5)
plt.plot(ts, opt.u_jump, 'b-')
plt.plot(ts, opt.jump_hist, 'r-')
plt.ylabel('jump(b), jump hist(r)')

plt.show()

print('asdf')