#!/usr/bin/env python3
import numpy as np
from stl import mesh
from pyquaternion import Quaternion
# from scipy.optimize import minimize, Bounds
# from scipy import integrate as int
# import scipy.linalg
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from gekko import GEKKO
import csv
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mplot3d
import sys
import linecache
import time
from Ball import Ball
from Car import Car
# from simulator_utilities import SimulationParameters

from DeepToot.src.dynamics_util.BouncingBallCalculator import BouncingBallCalculator

class BouncingOptimizer():
    position = []
    velocity = []
    orientation= []
    ang_vel = []

    @staticmethod
    def PrintException():
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


    def __init__(self):
#################GROUND DRIVING OPTIMIZER SETTTINGS##############
        self.d = GEKKO(remote=False) # Driving on ground optimizer

        ntd = 9

        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        self.d.options.NODES = 2
        self.d.options.SOLVER = 3
        self.d.options.IMODE = 6# MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.d.options.MAX_ITER = 800
        self.d.options.MV_TYPE = 0
        self.d.options.DIAGLEVEL = 2

        # final time for driving optimizer
        self.tf = self.d.FV(value=9.9,lb=3,ub=9.9, name='tf')

        # allow gekko to change the tfd value
        self.tf.STATUS = 1

        # Solver time for time based c splines
        self.solver_time = self.d.Var(value=0)
        self.d.Equation(self.solver_time.dt()/self.tf == 1)

        # Acceleration variable
        self.a = self.d.MV(lb = 0, ub = 1)
        self.a.STATUS = 1
        self.d.free(self.a)
        # self.a.DCOST = 1e-8

        # Torque input <Tx, Ty, Tz>, will change the q_omega[1:3] values since q_omega[0] is always zero in pure quaternion form
        self.alpha = self.d.Array(self.d.MV, 3)
        for i in self.alpha:
            # i.value = 0
            i.STATUS = 1
            i.upper = 1
            i.lower = -1
            i.DCOST = 0
            self.d.free(i)
        # Angular Velocity input (testing for now, maybe is quicker than a torque input)
        self.q_omega = self.d.Array(self.d.Var, 4) # w, x, y, z for quaternion
        for qi in self.q_omega:
            # qi.value = 0
            # qi.STATUS = 1
            qi.upper = 1
            qi.lower = -1
            # qi.DCOST = 1e-5
        # self.q_omega[0].status = 0# Shut off scalar part of omega quaternion
        self.omega_mag = self.d.Var() # Magnitude of angular velocity

        # end time variables to multiply u2 by to get total value of integral
        # Time vector length is nt1 and nt2
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final = self.d.Param(value = self.p_d)

        # gravity
        self.g = self.d.Param(value = -650)
        self.D_b = self.d.Param(value = -0.0305) # Air drag parameter on ball

        # Drag and torque coefficient
        self.T_r = self.d.Param(value = -36.07956616966136) # torque coefficient for roll
        self.T_p = self.d.Param(value = -12.14599781908070) # torque coefficient for pitch
        self.T_y = self.d.Param(value = 8.91962804287785) # torque coefficient for yaw
        self.D_r = self.d.Param(value = -4.47166302201591) # drag coefficient for roll
        self.D_p = self.d.Param(value = -2.798194258050845) # drag coefficient for pitch
        self.D_y = self.d.Param(value = -1.886491900437232) # drag coefficient for yaw
        self.amax = self.d.Param(value = 991.666+60)
        self.vmax = self.d.Param(value = 2300)
        # integral over time for u_pitch^2
        # self.u2_pitch = self.d.Var(value=0)
        # self.d.Equation(self.u2.dt() == 0.5*self.u_pitch**2)


    def optimizeAerial(self, sim_params, bc: BouncingBallCalculator): #these are 1x2 vectors s or v [x, z]
        try:
            ball = bc.trajectory
            car = sim_params.car_state

            #state variables
            self.bs = self.d.Array(self.d.Var, (3)) # ball position
            for ind, b in enumerate(self.bs):
                b.value = ball.pos[ind,0]
            self.bv = self.d.Array(self.d.Var, (3)) # ball velocity
            for ind, b in enumerate(self.bv):
                b.value = ball.vel[ind,0]

            # c-spline of ball trajectory
            for index, i in enumerate(self.bs):
                self.d.cspline(self.solver_time, i, ball.t, ball.pos[index])
            for index, i in enumerate(self.bv):
                self.d.cspline(self.solver_time, i, ball.t, ball.vel[index])

            # variables intial conditions are placed here
                # Position and Velocity in 2d

            self.sx = self.d.Var(value=car.position[0], lb=-4096, ub=4096) #x position
            self.sy = self.d.Var(value=car.position[1], lb=-5120, ub=5120) #y position
            self.sz = self.d.Var(value = car.position[2], lb = 0, ub = 2000)

            # NOTE: Important note, in gui i have entrys as ijkw, the math uses the wijk order
            # That is why q[0] = car.orientation[3]
            self.q = self.d.Array(self.d.Var, 4) #orientation quaternion
            self.q[0].value = car.orientation[0]
            self.q[1].value = car.orientation[1]
            self.q[2].value = car.orientation[2]
            self.q[3].value = car.orientation[3]

            self.q_norm = [None, None, None, None] # Initialize q_norm
            self.q_norm[0] = self.d.Var(value = car.orientation[0]) #Intermediate(equation = self.q[0]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
            self.q_norm[1] = self.d.Var(value = car.orientation[1]) #Intermediate(equation = self.q[1]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
            self.q_norm[2] = self.d.Var(value = car.orientation[2]) #Intermediate(equation = self.q[2]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
            self.q_norm[3] = self.d.Var(value = car.orientation[3]) #Intermediate(equation = self.q[3]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))


            # self.q_omega[0].value = 0
            # self.q_omega[1].value = car.ang_vel[1]
            # self.q_omega[2].value = car.ang_vel[2]
            # self.q_omega[3].value = car.ang_vel[3]

            # Intermediate value since rotating vector by quateiron requires q*v*q^-1 (this is q*v)
            ux = [0,1,0,0]
            self.hi = [None, None, None, None]
            self.hi[0] = self.d.Intermediate(equation = -1* (self.q_norm[1]*ux[1]))
            self.hi[1] = self.d.Intermediate(equation = (self.q_norm[0]*ux[1]))
            self.hi[2] = self.d.Intermediate(equation = (self.q_norm[3]*ux[1]))
            self.hi[3] = self.d.Intermediate(equation =  -1* (self.q_norm[2]*ux[1]))

            # This is the unit vector that points in the direction of the nose of the car (used to find the v.dt() from thruster)
            self.heading = self.d.Array(self.d.Var, 3)
            # Set intiial condition of heading
            # q = Quaternion([car.orientation[3], car.orientation[0], car.orientation[1], car.orientation[2]])
            # ri = Quaternion(car.orientation) # Get orientation into quaternion form to use the rotate method
            h = car.orientation.rotate(ux[1:])
            self.heading[0].value = h[0]
            self.heading[1].value = h[1]
            self.heading[2].value = h[2]
            print('h:', h)


            self.v_mag = self.d.Var()
            self.vx = self.d.Var(value=car.velocity[0]) #x velocity
            self.vy = self.d.Var(value=car.velocity[1]) #y velocity
            self.vz = self.d.Var(value=car.velocity[2])

    # Differental equations

            # Heading equations assuming original heading vector is <1,0,0>

            # Quaternion Derivative, q/dt = w*q
            self.d.Equation(self.q[0].dt() == 0.5 * self.tf * ((self.q_omega[0]*self.q[0]) - (self.q_omega[1]*self.q[1]) - (self.q_omega[2]*self.q[2]) -  (self.q_omega[3]*self.q[3])))
            self.d.Equation(self.q[1].dt() == 0.5 * self.tf * ((self.q_omega[0]*self.q[1]) + (self.q_omega[1]*self.q[0]) + (self.q_omega[2]*self.q[3]) - (self.q_omega[3]*self.q[2])))
            self.d.Equation(self.q[2].dt() == 0.5 * self.tf * ((self.q_omega[0]*self.q[2]) - (self.q_omega[1]*self.q[3]) + (self.q_omega[2]*self.q[0]) + (self.q_omega[3]*self.q[1])))
            self.d.Equation(self.q[3].dt() == 0.5 * self.tf * ((self.q_omega[0]*self.q[3]) + (self.q_omega[1]*self.q[2]) - (self.q_omega[2]*self.q[1]) + (self.q_omega[3]*self.q[0])))

            # Normalize quaternion after it has been affected by possible noise from quaternion derivative floating point math
            self.d.Equation(self.q_norm[0] == self.q[0]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
            self.d.Equation(self.q_norm[1] == self.q[1]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
            self.d.Equation(self.q_norm[2] == self.q[2]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))
            self.d.Equation(self.q_norm[3] == self.q[3]/self.d.sqrt((self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2)))

            # Omega/dt, angular acceleration, alpha are the controller inputs, and their related Torque magnitude (roll, pitch, yaw | T_r, T_p, T_y)
            # There is also a damper component to the rotataion dynamics, for pitch and yaw, the dampening effect is altered by the input value of that
            # rotational input, and requres an absoulte value but to make calculation converge better I tried sqrt(var^2) instead of the absoulte value
            # Since absoulte value would require me to move to integer type soltuion
            self.d.Equation(self.q_omega[1].dt() == self.tf * ((self.alpha[0] * self.T_r) + (self.q_omega[1]*self.D_r)))
            self.d.Equation(self.q_omega[2].dt() == self.tf * ((self.alpha[1] * self.T_p) + (self.q_omega[2]*self.D_p * (1-self.d.sqrt(self.alpha[1]*self.alpha[1])))))
            self.d.Equation(self.q_omega[3].dt() == self.tf * ((self.alpha[2] * self.T_y) + (self.q_omega[3]*self.D_y * (1-self.d.sqrt(self.alpha[2]*self.alpha[2])))))
            # self.d.Equation(self.q_omega[1].dt() == self.tf * (0.5 * self.alpha[0] * self.T_r)) # I'm multiplying the acceleration output by 0.5 to keep paths reachable by feedback controller only
            # self.d.Equation(self.q_omega[2].dt() == self.tf * (0.5 * self.alpha[1] * self.T_p))
            # self.d.Equation(self.q_omega[3].dt() == self.tf * (0.5 * self.alpha[2] * self.T_y))


            # Get unit vector pointing in heading direction (hi is q*v, this is (q*v)*q^-1) Thats why it has negatives on q_norm[1,2,3]
            # hi is the intermediate q*v, which is the lefthand side of the vector equation (q*v)*q^-1 to rotate a vector by the quaternion
            # This equation is the right handsize of the equation (q*v)*q^-1
            # hi is always rotating the quaternion amount from the unit x vector [0, 1, 0, 0] [w, i, j, k]
            # Unit x vector
            ux = [0, 1, 0, 0]
            self.d.Equation(self.heading[0] == (self.hi[0]*-1*self.q_norm[1]) + (self.hi[1]*self.q_norm[0]) + (self.hi[2]*-1*self.q_norm[3]) - (self.hi[3]*-1*self.q_norm[2]))
            self.d.Equation(self.heading[1] == (self.hi[0]*-1*self.q_norm[2]) - (self.hi[1]*-1*self.q_norm[3]) + (self.hi[2]*self.q_norm[0]) + (self.hi[3]*-1*self.q_norm[1]))
            self.d.Equation(self.heading[2] == (self.hi[0]*-1*self.q_norm[3]) + (self.hi[1]*-1*self.q_norm[2]) - (self.hi[2]*-1*self.q_norm[1]) + (self.hi[3]*self.q_norm[0]))

            self.d.Equation(self.v_mag == self.d.sqrt((self.vx)**2 + (self.vy)**2 + (self.vz)**2))
            self.d.Equation(self.v_mag <= 2300)

            self.d.Equation(self.vx.dt() == self.tf * self.amax * self.a * self.heading[0])
            self.d.Equation(self.vy.dt()== self.tf * self.amax * self.a * self.heading[1])
            self.d.Equation(self.vz.dt() == self.tf * ((self.amax * self.a * self.heading[2]) + self.g))
            # Car velocity with drag componnent
            # self.d.Equation(self.vx.dt() == self.tf * ((self.amax * self.a * self.heading[0]) + (self.vx * self.D_b)))
            # self.d.Equation(self.vy.dt() == self.tf * ((self.amax * self.a * self.heading[1]) + (self.vy * self.D_b)))
            # self.d.Equation(self.vz.dt() == self.tf * ((self.amax * self.a * self.heading[2]) + self.g + (self.vz * self.D_b)))

            self.d.Equation(self.sx.dt() == self.tf * self.vx)
            self.d.Equation(self.sy.dt() == self.tf * self.vy)
            self.d.Equation(self.sz.dt() == self.tf * self.vz)

            # Limit maximum total omega
            self.d.Equation(self.omega_mag == self.d.sqrt((self.q_omega[1]**2) + (self.q_omega[2]**2) + self.q_omega[3]**2))
            self.d.Equation(self.omega_mag <= 5.5)

    # Objective Functions
            # Final Position Objectives
            # self.d.Obj(self.final*1e5*((self.sx-sf[0])**2)) # Soft constraints
            # self.d.Obj(self.final*1e5*((self.sy-sf[1])**2)) # Soft constraints
            # self.d.Obj(self.final*1e5*((self.sz-sf[2])**2))

            # Final Velocity Objectives
            # self.d.Obj(self.final * 1e5 * (self.vx - vf[0])**2)
            # self.d.Obj(self.final * 1e5 * (self.vy - vf[1])**2)
            # self.d.Obj(self.final * 1e5 * (self.vz - vf[2])**2)

            # Maximize z velocity
            # self.d.Obj(-1*self.final*self.vy)

            # Move with balls velocity at end
            # self.d.Obj(self.final*1e2*((self.vx - self.bvx)**2))
            # self.d.Obj(self.final*1e2*((self.vy - self.bvy)**2))

            # End orientation Objective
            # self.d.Obj(self.final * 1e9 * ((self.q_norm[0]*rf[1]) + (self.q_norm[1]*rf[0]) + (self.q_norm[2]*rf[3]) - (self.q_norm[3]*rf[2]))**2)
            # self.d.Obj(self.final * 1e9 * ((self.q_norm[0]*rf[2]) - (self.q_norm[1]*rf[3]) + (self.q_norm[2]*rf[0]) + (self.q_norm[3]*rf[1]))**2)
            # self.d.Obj(self.final * 1e9 * ((self.q_norm[0]*rf[3]) + (self.q_norm[1]*rf[2]) - (self.q_norm[2]*rf[1]) + (self.q_norm[3]*rf[0]))**2)

            # Collide with ball objective
            self.d.Minimize(self.final*1e6*((self.sx - self.bs[0])**2))
            self.d.Minimize(self.final*1e6*((self.sy - self.bs[1])**2))
            self.d.Minimize(self.final*1e6*((self.sz - self.bs[2])**2))
            # Limit the x axis rotation
            # self.d.Obj(1e10 * (self.q_omega[1]-0)**2)
            # self.d.Obj(1e10 * (self.q_omega[2]-0)**2)
            
            #Objective function to minimize time
            self.d.Minimize(self.tf)

        except Exception as e:
            BouncingOptimizer.PrintException()
    
    def organize_gekko_variables(self):
        self.position = [self.sx.value, self.sy.value, self.sz.value]
        self.velocity = [self.vx.value, self.vy.value, self.vz.value]
        # Orientation needs to be a list os Quaternions
        self.orientation = []
        for i, q in enumerate(self.q_norm[0].value):
            quat = Quaternion(w=self.q_norm[0].value[i], x=self.q_norm[1].value[i], y=self.q_norm[2].value[i], z=self.q_norm[3].value[i])
            self.orientation.append(quat)
        self.ang_vel = [self.q_omega[1], self.q_omega[2], self.q_omega[3]]

    def solve(self):
        try:


            #solve
            # self.d.solve('http://127.0.0.1') # Solve with local apmonitor server
            self.d.solve(disp=True)

            self.ts = np.multiply(self.d.time, self.tf.value[0])
            self.organize_gekko_variables() # Organize gekko variables into easily iterable arrays
            return self.a, self.ts

        except Exception as e:
            BouncingOptimizer.PrintException()

    def getTrajectoryData(self):
        return [self.ts, self.sx, self.sy, self.vx, self.vy, self.yaw, self.omega]

    def getInputData(self):
        return [self.ts, self.a]

# TODO: Clean up plotData function
# Make neater, more usable, maybe give it options?
# put titles on figures
def plotData(figNum, opt):
    # plot results
    fig = plt.figure(figNum)
    ax = fig.add_subplot(111, projection='3d')
    # plt.subplot(2, 1, 1)
    Axes3D.plot(ax, opt.sx, opt.sy, opt.sz, c='r', marker ='o')
    Axes3D.plot(ax, opt.bs[0], opt.bs[1], opt.bs[2], c='b', marker = '*')
    plt.ylabel('Position/Velocity y')
    plt.xlabel('Position/Velocity x')
    ax.set_zlabel('z')
    ax.set_xlim3d(-3000, 3000)
    ax.set_ylim3d(-3000, 3000)
    ax.set_zlim3d(-0, 2000)

    # fig = plt.figure(3)
    # ax = fig.add_subplot(111, projection='3d')
    # plt.subplot(2, 1, 1)
    # Axes3D.plot(ax, vx, vy, vz, c='b', marker ='.')

    # Orientation Plotting
    fig = plt.figure(figNum + 5)
    ax2 = fig.add_subplot(111, projection='3d')
    # Use optimizer quaternion to rotate 1,0,0 vector.
    ux = np.array([1, 0, 0])
    print('qnorm from opt')
    for i in range(len(opt.q_norm)):
        print(opt.q_norm[i].value)

    hx = None
    hy = None
    hz = None
    for i in range(len(opt.q_norm[0].value)):
        if(i==0):
            q_temp = [opt.q_norm[0].value[i], opt.q_norm[1].value[i], opt.q_norm[2].value[i], opt.q_norm[3].value[i]]
            q = Quaternion(q_temp)
            q=q.unit
            v = q.rotate(ux)
            hx = np.array(v[0])
            hy = np.array(v[1])
            hz = np.array(v[2])
            px = np.array([opt.sx.value[i]])
            py = np.array([opt.sy.value[i]])
            pz = np.array([opt.sz.value[i]])
        q_temp = [opt.q_norm[0].value[i], opt.q_norm[1].value[i], opt.q_norm[2].value[i], opt.q_norm[3].value[i]]
        q = Quaternion(q_temp)
        q=q.unit
        v= q.rotate(ux)
        hx = np.append(hx, v[0])
        hy = np.append(hy, v[1])
        hz = np.append(hz, v[2])
        px = np.append(px, opt.sx.value[i])
        py = np.append(py, opt.sy.value[i])
        pz = np.append(pz, opt.sz.value[i])
    for i in range(np.size(hx)):
        # color stuff
        cmax = np.size(hx)
        origin = np.array([0])
        x = np.array([0, hx[i]])
        y = np.array([0, hy[i]])
        z = np.array([0, hz[i]])
        qx = [px[i], px[i]+hx[i]*400]
        qy = [py[i], py[i]+hy[i]*400]
        qz = [pz[i], pz[i]+hz[i]*400]

        Axes3D.plot(ax2, x, y, z, c=[i/cmax,i/cmax,0,1], marker ='o')
        # Axes3D.plot(ax, qx, qy, qz, c = 'b', marker = '^')
        plt.draw()
    from util.mpl3dObjects.sphere import Sphere
    s = Sphere(ax2, x=0, y=0, z=0, radius = 1, detail_level=10)
    

    Axes3D.plot(ax2, [0],[0],[1], c='b', marker = '*')
    # Axes3D.plot(ax, hx, hy, hz, c='r', marker ='o')
    Axes3D.plot(ax2, [0],[0],[0], c='y', marker = '*')
    Axes3D.plot(ax2, [1],[0],[0], c='r', marker = '*')
    Axes3D.plot(ax2, [0],[1],[0], c='g', marker = '*')
    ax2.set_xlim3d(-1, 1)
    ax2.set_ylim3d(-1, 1)
    ax2.set_zlim3d(-1, 1)
    ax2.auto_scale_xyz([-1, 1], [-1, 1], [-.2, .2])
    plt.xlabel('x')
    plt.ylabel('y')
    # # ax2.set_aspect('equal')
    # X = np.random.rand(100)*10+5
    # Y = np.random.rand(100)*5+2.5
    # Z = np.random.rand(100)*50+25
    # # Create cubic bounding box to simulate equal aspect ratio
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #    ax2.plot([xb], [yb], [zb], 'w')
    plt.draw()

    plt.figure(figNum+1)
    plt.subplot(3,1,1)
    plt.plot(opt.ts, opt.vx, 'r-')
    plt.plot(opt.ts, opt.vy, 'g-')
    plt.plot(opt.ts, opt.vz, 'b-')
    plt.ylabel('velocity componenet')

    plt.subplot(3,1,2)
    plt.plot(opt.ts, opt.q_omega[1], 'r-')
    plt.plot(opt.ts, opt.q_omega[2], 'g-')
    plt.plot(opt.ts, opt.q_omega[3], 'b-')
    plt.ylabel('angular velocities')
    plt.ylim(-2, 2)

    plt.subplot(3,1,3)
    plt.plot(opt.ts, opt.alpha[0], 'r-')
    plt.plot(opt.ts, opt.alpha[1], 'b-')
    plt.plot(opt.ts, opt.alpha[2], 'g-')
    plt.ylim(0, 5)
    # plt.plot(ts, a[1], 'g-')
    # plt.plot(ts, a[2], 'b-')
    plt.ylabel('alpha')

    # plt.figure(figNum+2)
    # plt.plot(sx, sy, 'r-')
    # plt.xlim(-3000, 3000)
    # plt.ylim(-3000, 3000)
    # plt.title('xy pos')
    # plt.plot(sx[0], sy[0], 'go') # Plot starting point of trajectory
    # plt.plot(0, 0, 'g*')


if __name__ == "__main__":

    try:
        # Main Code

        opt = BouncingOptimizer()
        opt2 = BouncingOptimizer()
        opt3 = BouncingOptimizer()

    # First Trajectory
        s_ti = [3000, 3000, 300]
        v_ti = [-500, -1400, 300]
        s_tf = [0.0, 0.0, 1000]
        v_tf = [0.00, 0.0, 500]
        # Get starting orientation quaternion
        ux = np.array([1, 0, 0]) # Unit x vector
        di = np.array([1,0,0]) # Starting vector to point towards
        di = di/(np.linalg.norm(di)) # Make di a unit vector
        q_xyz = np.cross(ux, di)
        q_w = np.sqrt((np.linalg.norm(ux) ** 2) * (np.linalg.norm(di) ** 2)) + np.dot(ux, di)
        r_ti = Quaternion([1,0,0,1])
        # r_ti = Quaternion(scalar = q_w, vector = q_xyz)
        r_ti = r_ti.normalised

        # Get final orientatiaon quaternion
        df = np.array([0,-1,0])# final orientation vector to point towards
        df = df/(np.linalg.norm(df)) # Normalize df vector
        q_xyz = np.cross(ux, df) # If i do the order (ux, df) the y and z axes are flipped for some reason
        q_w = np.sqrt((np.linalg.norm(ux) ** 2) * (np.linalg.norm(df) ** 2)) + np.dot(ux, df)
        r_tf = Quaternion(scalar = q_w, vector = q_xyz)
        r_tf = r_tf.normalised
        r_tf = r_tf.conjugate

        omega_ti = [1, 0, 0, 0] # initial angular velocity of car

        # Initialize car, and ball states
        from simulator_utilities import State, SimulationParameters
        car = State()
        ball = State()
        quat = [r_ti.real, r_ti.imaginary[0], r_ti.imaginary[1], r_ti.imaginary[2]]
        car.init_from_raw_data(s_ti, v_ti, r_ti, omega_ti)  
        ball.init_from_raw_data([-3000, -3000, 100], [500, 500, 1500], [0,0,0,0], [0,0,0,0])
        sim_params = SimulationParameters()
        sim_params.initialize_from_raw(car, ball)

        # Ball calculator
        state0 = np.array([-3000,-3000,100,100,100,-2000])
        tf = 9.9
        dt = 0.01
        bc = BouncingBallCalculator(state0, tf, dt)
        bc.calculate()


        # opt.COLDSTART = 1
        # opt.LINEAR = 0
        opt.optimizeAerial(sim_params, bc)
        a, t_star = opt.solve()

        print('heading:','\n', opt.heading[0].value,'\n', opt.heading[1].value,'\n', opt.heading[2].value)
        hmag = np.sqrt(np.power(opt.heading[0].value, 2) + np.power(opt.heading[1].value, 2) + np.power(opt.heading[2].value, 2))
        print('hmag:', hmag)
        plotData(5, opt)

        # data = [opt.sx, opt.sy, opt.sz]
        # graph = FixedGraph(opt.sx[-1], opt.sy[-1], opt.sz[-1], 10)
        # ani = animation.FuncAnimation(graph.fig, graph.run, frames=data, interval=200, repeat=True)

        # Comment out this line if you don't want see the animation.
        plt.show()
        print('debug')
    except BaseException as e:
        BouncingOptimizer.PrintException()
        # a, t_star = opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, r_tf, omega_ti)
        # plotData(5, opt.ts, opt.sx, opt.sy, opt.sz, opt.vx, opt.vy, opt.vz, opt.q_omega)
        # plotAttitude(1, opt)
    plt.ioff()
    plt.show()
    print('debug')