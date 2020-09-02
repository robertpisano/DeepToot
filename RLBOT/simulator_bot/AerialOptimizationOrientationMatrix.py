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
from mathematical_conversions import convert_from_euler_angles

class AerialOptimizer():
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
        self.d = GEKKO(remote=True) # Driving on ground optimizer

##################################################
        # PHYSCIS PARAMETERS
        # gravity
        self.g = self.d.Param(value = -650)
        self.D_b = self.d.Param(value = -0.0305) # Air drag parameter on ball

        # torque coefficient in array form
        self.T = self.d.Array(self.d.Param, (3,3))
        self.T[0][0].value = -36.07956616966136
        self.T[0][1].value = 0.0
        self.T[0][2].value = 0.0
        self.T[0][2].value = 0.0
        self.T[0][2].value = -12.14599781908070
        self.T[0][2].value = 0.0
        self.T[0][2].value = 0.0
        self.T[0][2].value = 0.0
        self.T[0][2].value = 8.91962804287785

        # Drag coefficient in array form
        self.Drag = self.d.Array(self.d.Param, (3,3))
        self.Drag[0][0].value = -4.47166302201591
        self.Drag[0][1].value = 0.0
        self.Drag[0][2].value = 0.0
        self.Drag[1][0].value = 0.0
        self.Drag[1][1].value = -2.798194258050845
        self.Drag[1][2].value = 0.0
        self.Drag[2][0].value = 0.0
        self.Drag[2][1].value = 0.0
        self.Drag[2][2].value = -1.886491900437232

        self.amax = self.d.Param(value = 991.666+60)
################################################################

# nt1 nt2 are how many the first half and second half of time will be split
# t1 t2 are then concatenated to make the time vector which has variable deltaT

        # nt1 = 7
        # nt2 = 5
        # t1 = np.linspace(0,0.5, nt1)
        # t2 = np.linspace(0.55, 1.0, nt2)
        # self.d.time = np.concatenate((t1,t2), axis=0)

        ntd = 9

        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        self.d.options.NODES = 3
        self.d.options.SOLVER = 3
        self.d.options.IMODE = 6# MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.d.options.MAX_ITER = 800
        self.d.options.MV_TYPE = 1
        self.d.options.DIAGLEVEL = 0

        # final time for driving optimizer
        self.tf = self.d.FV(value=1.0,lb=0.1,ub=100.0)

        # allow gekko to change the tfd value
        self.tf.STATUS = 1

        # Scaled time for Rocket league to get proper time

        # Acceleration variable
        self.a = self.d.MV(lb = 0, ub = 1)
        self.a.STATUS = 1
        # self.a.DCOST = 1e-8

        # Torque input <Tx, Ty, Tz>, will change the q_omega[1:3] values since q_omega[0] is always zero in pure quaternion form
        self.alpha = self.d.Array(self.d.MV, (3, 3))
        for i in range(3):
            for j in range(3):
                if i == j: # if i == j we're along diagonal, so enable these variables for torque input
                    self.alpha[i][j].value = 0
                    self.alpha[i][j].STATUS = 1
                    self.alpha[i][j].upper = 1
                    self.alpha[i][j].lower = -1
                    # self.alpha[i][j].DCOST = 0
                else:
                    self.alpha[i][j].STATUS = 0
                    self.alpha[i][j].value = 0
              
        #Initialize orientation matrix to all zeroes
        self.orientation = self.d.Array(self.d.Var, (3,3))
        
        #Initialize ang vel matrix to all zeroes
        self.omega = self.d.Array(self.d.Var, (3,3))
        
        # Intermediate variable for A = [orientation][OMEGA]
        self.A1 = [[self.d.Intermediate(equation = (self.orientation.T[0][i] * self.omega[0][j]) + (self.orientation.T[1][i] * self.omega[1][j]) + (self.orientation.T[2][i] * self.omega[2][j])) for i in range(3)] for j in range(3)]
        # for i in range(3):
        #     for j in range(3):
        #         self.A1[i][j] = self.d.Intermediate(equation = (self.orientation.T[i][0] * self.omega[0][j]) + (self.orientation.T[i][1] * self.omega[1][j]) + (self.orientation.T[i][2] * self.omega[2][j]))

        # Intermediate variable for B = [D][orientation][OMEGA] = [D][A]
        self.B = [[self.d.Intermediate(equation = (self.Drag[i][0] * self.A1[0][j]) + (self.Drag[i][1] * self.A1[1][j]) + (self.Drag[i][2] * self.A1[2][j])) for i in range(3)] for j in range(3)]
        # for i in range(3):
        #     for j in range(3):
        #         self.B[i][j] = self.d.Intermediate(equation = (self.D[i][0] * self.A1[0][j]) + (self.D[i][1] * self.A1[1][j]) + (self.D[i][2] * self.A1[2][j]))

        # Intermediate variable for C = [T(torque)][alpha (u)]
        self.C = self.d.Array(self.d.Var, (3,3))
        for i in range(3):
            for j in range(3):
                self.C[i][j] = self.d.Intermediate(equation = (self.T[i][0] * self.alpha[0][j]) + (self.T[i][1] * self.alpha[1][j]) + (self.T[i][2] * self.alpha[2][j]))

        self.omega_mag = self.d.Var(ub = 5.5, lb = 0.0)

        # end time variables to multiply u2 by to get total value of integral
        # Time vector length is nt1 and nt2
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final = self.d.Param(value = self.p_d)

        # integral over time for u_pitch^2
        # self.u2_pitch = self.d.Var(value=0)
        # self.d.Equation(self.u2.dt() == 0.5*self.u_pitch**2)


    def optimizeAerial(self, sim_params): #these are 1x2 vectors s or v [x, z]
        try:
            ball = sim_params.ball_state
            car = sim_params.car_state

            # Ball variables
            self.bx = self.d.Var(value=ball.position[0], lb=-4096, ub=4096) #x position
            self.by = self.d.Var(value=ball.position[1], lb=-5120, ub=5120) #y position
            self.bz = self.d.Var(value = ball.position[2], lb = 0, ub = 2000)
            self.bvx = self.d.Var(value = ball.velocity[0])
            self.bvy = self.d.Var(value = ball.velocity[1])
            self.bvz = self.d.Var(value = ball.velocity[2])

            # variables intial conditions are placed here
                # Position and Velocity in 2d

            self.sx = self.d.Var(value=car.position[0], lb=-4096, ub=4096) #x position
            self.sy = self.d.Var(value=car.position[1], lb=-5120, ub=5120) #y position
            self.sz = self.d.Var(value = car.position[2], lb = 0, ub = 2000)

            # Set initial values for orientation matrix
            theta = convert_from_euler_angles(car.euler[0], car.euler[1], car.euler[2])
            for i in range(3):
                for j in range(3):
                    self.orientation[i][j].value = theta[i][j]

            # Set initial values for angular velocity matrix
            omega = car.ang_vel
            self.omega[0][0].value = 0
            self.omega[0][1].value = -1*omega[2]
            self.omega[0][2].value = omega[1]
            self.omega[1][0].value = omega[2]
            self.omega[1][1].value = 0
            self.omega[1][2].value = -1*omega[0]
            self.omega[2][0].value = -1*omega[1]
            self.omega[2][1].value = omega[1]
            self.omega[2][2].value = 0

            # # self.v_mag = self.d.Var(value = vi, ub = 2300, lb =0)
            self.vx = self.d.Var(value=car.velocity[0], ub=2300, lb=-2300) #x velocity
            self.vy = self.d.Var(value=car.velocity[1], ub=2300, lb=-2300) #y velocity
            self.vz = self.d.Var(value=car.velocity[2], ub=2300, lb=-2300)

    # Differental equations

            # Heading equations assuming original heading vector is <1,0,0>

            # Orientation Matrix Derivative, Theta/dt = [Omega][Theta]
            self.add_angular_velocity_equations(self.omega, self.orientation)

            # Omega/dt, angular acceleration, alpha are the controller inputs, and their related Torque magnitude (roll, pitch, yaw | T_r, T_p, T_y)
            # There is also a damper component to the rotataion dynamics, for pitch and yaw, the dampening effect is altered by the input value of that
            # rotational input, and requres an absoulte value but to make calculation converge better I tried sqrt(var^2) instead of the absoulte value
            # Since absoulte value would require me to move to integer type soltuion
            self.add_angular_acceleration_equations()

            # Linear velocity equations, boost comes from the front vector which is THETA[:, 0], ORIENTATION[:, 0]
            self.d.Equation(self.vx.dt() == self.tf * self.amax * self.a * self.orientation[0][0])
            self.d.Equation(self.vy.dt()== self.tf * self.amax * self.a * self.orientation[1][0])
            self.d.Equation(self.vz.dt() == self.tf * ((self.amax * self.a * self.orientation[2][0]) + self.g))
            # Car velocity with drag componnent
            # self.d.Equation(self.vx.dt() == self.tf * ((self.amax * self.a * self.heading[0]) + (self.vx * self.D_b)))
            # self.d.Equation(self.vy.dt() == self.tf * ((self.amax * self.a * self.heading[1]) + (self.vy * self.D_b)))
            # self.d.Equation(self.vz.dt() == self.tf * ((self.amax * self.a * self.heading[2]) + self.g + (self.vz * self.D_b)))

            self.d.Equation(self.sx.dt() == self.tf * self.vx)
            self.d.Equation(self.sy.dt() == self.tf * self.vy)
            self.d.Equation(self.sz.dt() == self.tf * self.vz)

            # Limit maximum total omega
            self.d.Equation(self.omega_mag == self.d.sqrt((self.omega[2][1]**2) + (self.omega[0][2]**2) + self.omega[1][0]**2))
            self.d.Equation(self.omega_mag <= 5.5)

            # Ball diff eq
            self.d.Equation(self.bx.dt() == self.tf * self.bvx)
            self.d.Equation(self.by.dt() == self.tf * self.bvy)
            self.d.Equation(self.bz.dt() == self.tf * self.bvz)
            self.d.Equation(self.bvx.dt() == self.tf * self.bvx * self.D_b)
            self.d.Equation(self.bvy.dt() == self.tf * self.bvy * self.D_b)
            self.d.Equation(self.bvz.dt() == self.tf * (self.g + (self.D_b * self.bvz)))

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
            self.d.Obj(self.final*1e2*((self.sx - self.bx)**2))
            self.d.Obj(self.final*1e2*((self.sy - self.by)**2))
            self.d.Obj(self.final*1e2*((self.sz - self.bz)**2))

            # Limit the x axis rotation
            # self.d.Obj(1e10 * (self.q_omega[1]-0)**2)
            # self.d.Obj(1e10 * (self.q_omega[2]-0)**2)
            
            #Objective function to minimize time
            self.d.Obj(self.tf)


        except Exception as e:
            AerialOptimizer.PrintException()
    
    # Will do regular matrix multiplication (currently only 3x3) and will set the elements of out
    def add_angular_velocity_equations(self, x=[], y=[]):
        for i in range(len(x[:,0])):
            for j in range(len(x[0,:])):
                self.d.Equation(self.orientation[i][j].dt() == self.tf * ((x[i][0]*y[0][j]) + (x[i][1]*y[1][j]) + (x[i][2]*y[2][j])))

    def add_angular_acceleration_equations(self):
        for i in range(3):
            for j in range(3):
                self.d.Equation(self.omega[i][j].dt() == self.tf * ((self.orientation[i][0] * (self.B[0][j] + self.C[0][j])) + (self.orientation[i][1] * (self.B[1][j] + self.C[1][j])) + (self.orientation[i][2] * (self.B[2][j] + self.C[2][j]))))

    def organize_gekko_variables(self):
        self.position = [self.sx.value, self.sy.value, self.sz.value]
        self.velocity = [self.vx.value, self.vy.value, self.vz.value]
        # Orientation needs to be a list os Quaternions
        self.orientation = []
        self.ang_vel = [self.omega[0][0], self.omega[1][0], self.omega[2][0]]

    def solve(self):
        try:


            #solve
            # self.d.solve('http://127.0.0.1') # Solve with local apmonitor server
            import os
            path = os.path.realpath(self.d._path)
            os.startfile(path)
            self.d.solve(disp=True, GUI=True)

            # WILL PROBABLY BREAK AFTER SOLVING
            self.ts = np.multiply(self.d.time, self.tf.value[0])
            self.organize_gekko_variables() # Organize gekko variables into easily iterable arrays
            return self.a, self.ts

        except Exception as e:

            AerialOptimizer.PrintException()

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
    Axes3D.plot(ax, opt.bx, opt.by, opt.bz, c='b', marker = '*')
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

    # # Orientation Plotting
    # fig = plt.figure(figNum + 5)
    # ax2 = fig.add_subplot(111, projection='3d')
    # # Use optimizer quaternion to rotate 1,0,0 vector.
    # ux = np.array([1, 0, 0])
    # print('qnorm from opt')
    # for i in range(len(opt.q_norm)):
    #     print(opt.q_norm[i].value)

    # hx = None
    # hy = None
    # hz = None
    # for i in range(len(opt.q_norm[0].value)):
    #     if(i==0):
    #         q_temp = [opt.q_norm[0].value[i], opt.q_norm[1].value[i], opt.q_norm[2].value[i], opt.q_norm[3].value[i]]
    #         q = Quaternion(q_temp)
    #         q=q.unit
    #         v = q.rotate(ux)
    #         hx = np.array(v[0])
    #         hy = np.array(v[1])
    #         hz = np.array(v[2])
    #         px = np.array([opt.sx.value[i]])
    #         py = np.array([opt.sy.value[i]])
    #         pz = np.array([opt.sz.value[i]])
    #     q_temp = [opt.q_norm[0].value[i], opt.q_norm[1].value[i], opt.q_norm[2].value[i], opt.q_norm[3].value[i]]
    #     q = Quaternion(q_temp)
    #     q=q.unit
    #     v= q.rotate(ux)
    #     hx = np.append(hx, v[0])
    #     hy = np.append(hy, v[1])
    #     hz = np.append(hz, v[2])
    #     px = np.append(px, opt.sx.value[i])
    #     py = np.append(py, opt.sy.value[i])
    #     pz = np.append(pz, opt.sz.value[i])
    # for i in range(np.size(hx)):
    #     # color stuff
    #     cmax = np.size(hx)
    #     origin = np.array([0])
    #     x = np.array([0, hx[i]])
    #     y = np.array([0, hy[i]])
    #     z = np.array([0, hz[i]])
    #     qx = [px[i], px[i]+hx[i]*400]
    #     qy = [py[i], py[i]+hy[i]*400]
    #     qz = [pz[i], pz[i]+hz[i]*400]

    #     Axes3D.plot(ax2, x, y, z, c=[i/cmax,i/cmax,0,1], marker ='o')
    #     # Axes3D.plot(ax, qx, qy, qz, c = 'b', marker = '^')
    #     plt.draw()
    # from util.mpl3dObjects.sphere import Sphere
    # s = Sphere(ax2, x=0, y=0, z=0, radius = 1, detail_level=10)
    

    # Axes3D.plot(ax2, [0],[0],[1], c='b', marker = '*')
    # # Axes3D.plot(ax, hx, hy, hz, c='r', marker ='o')
    # Axes3D.plot(ax2, [0],[0],[0], c='y', marker = '*')
    # Axes3D.plot(ax2, [1],[0],[0], c='r', marker = '*')
    # Axes3D.plot(ax2, [0],[1],[0], c='g', marker = '*')
    # ax2.set_xlim3d(-1, 1)
    # ax2.set_ylim3d(-1, 1)
    # ax2.set_zlim3d(-1, 1)
    # ax2.auto_scale_xyz([-1, 1], [-1, 1], [-.2, .2])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # # # ax2.set_aspect('equal')
    # # X = np.random.rand(100)*10+5
    # # Y = np.random.rand(100)*5+2.5
    # # Z = np.random.rand(100)*50+25
    # # # Create cubic bounding box to simulate equal aspect ratio
    # # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    # # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    # # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    # # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # # # Comment or uncomment following both lines to test the fake bounding box:
    # # for xb, yb, zb in zip(Xb, Yb, Zb):
    # #    ax2.plot([xb], [yb], [zb], 'w')
    plt.draw()

    plt.figure(figNum+1)
    plt.subplot(3,1,1)
    plt.plot(opt.ts, opt.vx, 'r-')
    plt.plot(opt.ts, opt.vy, 'g-')
    plt.plot(opt.ts, opt.vz, 'b-')
    plt.ylabel('velocity componenet')

    plt.subplot(3,1,2)
    plt.plot(opt.ts, opt.omega[0][0], 'r-')
    plt.plot(opt.ts, opt.omega[0][1], 'g-')
    plt.plot(opt.ts, opt.omega[0][2], 'b-')
    plt.ylabel('angular velocities')
    plt.ylim(-2, 2)

    plt.subplot(3,1,3)
    plt.plot(opt.ts, opt.alpha[0][0], 'r-')
    plt.plot(opt.ts, opt.alpha[1][0], 'b-')
    plt.plot(opt.ts, opt.alpha[2][0], 'g-')
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


if __name__ == "__main2__":
    opt = AerialOptimizer()
    opt.optimizeAerial()

if __name__ == "__main__":

    try:
        # Main Code


        opt = AerialOptimizer()
        opt2 = AerialOptimizer()
        opt3 = AerialOptimizer()

    # First Trajectory
        s_ti = [-2500.0, 3000.0, 150]
        v_ti = [500, -500, 200]
        s_tf = [0.0, 0.0, 1000]
        v_tf = [0.00, 0.0, 500]
        r_ti = np.array([00,90,0]) # Euler Angles [R,P,Y]
        r_tf = np.array([0,0,0])
        omega_ti = [1, 0, 0, 0] # initial angular velocity of car

        # Initialize car, and ball states
        from simulator_utilities import State, SimulationParameters
        car = State()
        ball = State()
        car.init_from_raw_data_euler(s_ti, v_ti, r_ti, omega_ti)  
        ball.init_from_raw_data([3000, -3000, 100], [-1500, 1500, 1500], [0,0,0,0], [0,0,0,0])
        sim_params = SimulationParameters()
        sim_params.initialize_from_raw(car, ball)


        # opt.COLDSTART = 1
        opt.LINEAR = 0
        opt.optimizeAerial(sim_params)
        a, t_star = opt.solve()

        plotData(5, opt)

        # data = [opt.sx, opt.sy, opt.sz]
        # graph = FixedGraph(opt.sx[-1], opt.sy[-1], opt.sz[-1], 10)
        # ani = animation.FuncAnimation(graph.fig, graph.run, frames=data, interval=200, repeat=True)

        # Comment out this line if you don't want see the animation.
        plt.show()
        print('debug')
    except BaseException as e:
        AerialOptimizer.PrintException()
        # a, t_star = opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, r_tf, omega_ti)
        # plotData(5, opt.ts, opt.sx, opt.sy, opt.sz, opt.vx, opt.vy, opt.vz, opt.q_omega)
        # plotAttitude(1, opt)
    plt.ioff()
    plt.show()
