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
fig4 = plt.figure(4)

def plot_car(o: Optimizer):
    # Plot 3d position of ball and car on fig1
    Axes3D.plot(ax3d1, o.car.pos.x, o.car.pos.y, o.time*o.tf, c = 'r', marker = 'o')
    # Limits
    ax3d1.set_xlabel('x')
    ax3d1.set_ylabel('y')
    ax3d1.set_zlabel('z')
    ax3d1.set_xlim3d(-1000, 1000)
    ax3d1.set_ylim3d(-1000, 1000)
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
    plt.plot(ts, o.car.orientation.yaw, 'r-')
    # plt.plot(ts, o.car.ang_vel.z, 'b-')
    plt.ylabel('yaw orientation (red), yaw_dot (blue)')

    plt.subplot(num_plots, 1, 3)
    plt.plot(ts, o.car.v_mag, 'b-')
    plt.ylabel('vmag')

    plt.subplot(num_plots, 1, 4)
    plt.plot(o.car.kv_fine, o.car.ka_fine, 'r-')
    plt.ylabel('kv vs ka')
    
def plot_ball(o: Optimizer):
    Axes3D.plot(ax3d1, o.ball.x, o.ball.y, o.ball.z, c = 'b', marker = '*')

def show():
    # plt.ion()
    plt.show(block=True)
    # plt.draw()
    plt.pause(0.1) 