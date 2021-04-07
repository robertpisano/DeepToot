from rlbot.utils.game_state_util import CarState, BallState, Vector3, Physics, Rotator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DeepToot.src.gekko_util.gekko_util import *




fig1 = plt.figure(1)
ax3d1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure(2)
ax3d2 = fig2.add_subplot(111, projection='3d')
fig3 = plt.figure(3)
ax3d3 = fig3.add_subplot(111, projection='3d')
fig4 = plt.figure(4)

def plot_car(o: Optimizer):
    # Plot 3d position of ball and car on fig1
    Axes3D.plot(ax3d1, o.car.pos.x, o.car.pos.y, o.time*o.tf, c = 'r', marker = 'o')
    # Limits
    ax3d1.set_xlabel('x')
    ax3d1.set_ylabel('y')
    ax3d1.set_zlabel('z')
    ax3d1.set_xlim3d(-2000, 2000)
    ax3d1.set_ylim3d(-2000, 2000)
    ax3d1.set_zlim3d(0, o.tf.value[0])

    # Plot velocity of plot
    Axes3D.plot(ax3d2, o.car.vel.x, o.car.vel.y, o.time*o.tf, c = 'r', marker = 'o')
    # Limits
    ax3d2.set_xlabel('x')
    ax3d2.set_ylabel('y')
    ax3d2.set_zlabel('z')
    ax3d2.set_xlim3d(-1000, 1000)
    ax3d2.set_ylim3d(-1000, 1000)
    ax3d2.set_zlim3d(0, o.tf.value[0])

    ts = o.time * o.tf
    fig4
    num_plots = 4
    plt.subplot(num_plots,1,1)
    plt.plot(ts, o.car.a, 'r-')
    plt.ylabel('acceleration')

    plt.subplot(num_plots,1,2)
    plt.plot(ts, o.car.vel.x, 'r-')
    # plt.plot(ts, o.car.ang_vel.z, 'b-')
    plt.ylabel('yaw orientation (red), yaw_dot (blue)')

    plt.subplot(num_plots, 1, 3)
    plt.plot(ts, o.car.v_mag, 'b-')
    plt.ylabel('vmag')

    plt.subplot(num_plots, 1, 4)
    plt.plot(o.car.kv_fine, o.car.ka_fine, 'r-')
    plt.ylabel('kv vs ka')
    
def plot_cars(o: Optimizer):
    # Plot 3d position of ball and car on fig1
    Axes3D.plot(ax3d1, o.c1.pos.x, o.c1.pos.y, o.time*o.tf1, c = 'r', marker = '^')
    plt.draw()
    Axes3D.plot(ax3d1, o.c2.pos.x, o.c2.pos.y, o.time*o.tf2, c = 'g', marker = '^')
    # Limits
    ax3d1.set_xlabel('x')
    ax3d1.set_ylabel('y')
    ax3d1.set_zlabel('t')
    ax3d1.set_xlim3d(-2000, 2000)
    ax3d1.set_ylim3d(-2000, 2000)
    ax3d1.set_zlim3d(0, max(o.tf1.value[0], o.tf2.value[0]))

    # Plot velocity of plot
    Axes3D.plot(ax3d2, o.c1.vel.x, o.c1.vel.y, o.time*o.tf1, c = 'r', marker = '^')
    plt.draw()
    Axes3D.plot(ax3d2, o.c2.vel.x, o.c2.vel.y, o.time*o.tf2, c = 'g', marker = '^')
    # Limits
    ax3d2.set_xlabel('x')
    ax3d2.set_ylabel('y')
    ax3d2.set_zlabel('t')
    ax3d2.set_xlim3d(-1000, 1000)
    ax3d2.set_ylim3d(-1000, 1000)
    ax3d2.set_zlim3d(0, max(o.tf1.value[0], o.tf2.value[0]))

    # ts = o.time * o.tf
    # fig4
    # num_plots = 4
    # plt.subplot(num_plots,1,1)
    # plt.plot(ts, o.car.a, 'r-')
    # plt.ylabel('acceleration')

    # plt.subplot(num_plots,1,2)
    # plt.plot(ts, o.car.orientation.yaw, 'r-')
    # # plt.plot(ts, o.car.ang_vel.z, 'b-')
    # plt.ylabel('yaw orientation (red), yaw_dot (blue)')

    # plt.subplot(num_plots, 1, 3)
    # plt.plot(ts, o.car.v_mag, 'b-')
    # plt.ylabel('vmag')

    # plt.subplot(num_plots, 1, 4)
    # plt.plot(o.car.kv_fine, o.car.ka_fine, 'r-')
    # plt.ylabel('kv vs ka')

def plot_aerial_car(o: Optimizer):
    # Plot 3d position of ball and car on fig1
    Axes3D.plot(ax3d1, o.car.pos.x, o.car.pos.y, o.car.pos.z, c = 'r', marker = 'o')
    # Limits
    ax3d1.set_xlabel('x')
    ax3d1.set_ylabel('y')
    ax3d1.set_zlabel('z')
    ax3d1.set_xlim3d(-2000, 2000)
    ax3d1.set_ylim3d(-5000, 5000)
    ax3d1.set_zlim3d(0, 2000)

    # Plot goal position on fig 1
    gx = 0
    gy = 5120
    gz = 320
    Axes3D.plot(ax3d1, gx, gy, gz, c = 'g', marker = '*', markersize=10)

    # Plot final velocity vector on position plot
    shotx = o.car.vel.x[-1]
    shoty = o.car.vel.y[-1]
    shotz = o.car.vel.z[-1]
    fx = o.car.pos.x[-1]
    fy = o.car.pos.y[-1]
    fz = o.car.pos.z[-1]
    fx2 = fx+shotx
    fy2 = fy+shoty
    fz2 = fz+shotz    
    Axes3D.plot(ax3d1, [fx, fx2], [fy, fy2], [fz, fz2], c = 'k', marker = '^', markersize=10)

    # Plot velocity of plot
    Axes3D.plot(ax3d2, o.car.vel.x, o.car.vel.y, o.car.vel.z, c = 'r', marker = 'o')
    # Limits
    ax3d2.set_xlabel('x')
    ax3d2.set_ylabel('y')
    ax3d2.set_zlabel('z')
    ax3d2.set_xlim3d(-1000, 1000)
    ax3d2.set_ylim3d(-1000, 1000)
    ax3d2.set_zlim3d(0, 2000)

    # Use oimizer quaternion to rotate 1,0,0 vector.
    ux = np.array([1, 0, 0])
    print('qnorm from o')
    for i in range(len(o.car.q_norm)):
        print(o.car.q_norm[i].value)

    hx = None
    hy = None
    hz = None
    for i in range(len(o.car.q_norm[0].value)):
        if(i==0):
            q_temp = [o.car.q_norm[0].value[i], o.car.q_norm[1].value[i], o.car.q_norm[2].value[i], o.car.q_norm[3].value[i]]
            q = Quaternion(q_temp)
            q=q.unit
            v = q.rotate(ux)
            hx = np.array(v[0])
            hy = np.array(v[1])
            hz = np.array(v[2])
            px = np.array([o.car.pos.x.value[i]])
            py = np.array([o.car.pos.y.value[i]])
            pz = np.array([o.car.pos.z.value[i]])
        q_temp = [o.car.q_norm[0].value[i], o.car.q_norm[1].value[i], o.car.q_norm[2].value[i], o.car.q_norm[3].value[i]]
        q = Quaternion(q_temp)
        q=q.unit
        v= q.rotate(ux)
        hx = np.append(hx, v[0])
        hy = np.append(hy, v[1])
        hz = np.append(hz, v[2])
        px = np.append(px, o.car.pos.x.value[i])
        py = np.append(py, o.car.pos.y.value[i])
        pz = np.append(pz, o.car.pos.z.value[i])
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

        Axes3D.plot(ax3d3, x, y, z, c=[i/cmax,i/cmax,i/cmax,1], marker ='o')
        # Axes3D.plot(ax, qx, qy, qz, c = 'b', marker = '^')
        plt.draw()
    # from util.mpl3dObjects.sphere import Sphere
    # s = Sphere(ax3d3, x=0, y=0, z=0, radius = 1, detail_level=10)
    

    Axes3D.plot(ax3d3, [0],[0],[1], c='b', marker = '*')
    # Axes3D.plot(ax, hx, hy, hz, c='r', marker ='o')
    Axes3D.plot(ax3d3, [0],[0],[0], c='y', marker = '*')
    Axes3D.plot(ax3d3, [1],[0],[0], c='r', marker = '*')
    Axes3D.plot(ax3d3, [0],[1],[0], c='g', marker = '*')
    ax3d3.set_xlim3d(1, -1)
    ax3d3.set_ylim3d(1, -1)
    ax3d3.set_zlim3d(-1, 1)
    ax3d3.auto_scale_xyz([-1, 1], [-1, 1], [-.2, .2])
    plt.xlabel('x')
    plt.ylabel('y')

    ts = o.time * o.tf
    plt.figure(4)
    num_plots = 4
    plt.subplot(num_plots,1,1)
    plt.plot(ts, o.car.a, 'r-')
    plt.ylabel('acceleration')
   
def plot_ball(o: Optimizer):
    Axes3D.plot(ax3d1, o.ball.pos.x, o.ball.pos.y, o.time*o.tf, c = 'b', marker = '*')

def plot_balls(o:Optimizer):
    Axes3D.plot(ax3d1, o.ball.bs1[0], o.ball.bs1[1], o.time*o.tf1, c = 'b', marker = 'o')
    Axes3D.plot(ax3d1, o.ball.bs2[0], o.ball.bs2[1], o.time*o.tf2, c = 'c', marker = 'o')

def plot_aerial_ball(o: Optimizer):
    Axes3D.plot(ax3d1, o.ball.pos.x, o.ball.pos.y, o.ball.pos.z, c = 'b', marker = '*')
    Axes3D.plot(ax3d2, o.ball.vel.x, o.ball.vel.y, o.ball.vel.z, c = 'b', marker = '*')

def show():
    # plt.ion()
    plt.show(block=True)
    # plt.draw()
    plt.pause(0.1) 