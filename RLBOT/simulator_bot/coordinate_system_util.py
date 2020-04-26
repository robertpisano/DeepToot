  #imports for LQR functions
from __future__ import division, print_function

import math
from pyquaternion import Quaternion

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState

import random
import control #Control system python library

import numpy as np
import scipy.linalg
import sys
import linecache

class CoordinateSystems:
    def PrintException(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

    def __init__(self):

        #World Coordinate system vectors
        self.ball_world = None #Position, Velocities in world frame
        self.car_world = None #Position, Velocities in world frame
        self.q_car_to_world = None #Unit Quaternion to change car frame to world frame
        self.R_car_to_world = None #Rigid Transformation matrix to change car frame to world frame

        #Car Coordinate system vectors
        self.ball_car = None #Position, Velocities to ball in car coordinates
        self.q_car_to_ball = None #Unit Quaternion pointing to ball from car frame
        self.q_world_to_car = None #Unit Quaternion to change from world frame to car frame
        self.R_world_to_car = None #Rigid Transformation matrix to change world frame to car frame
        self.heading = None

    def update(self, car_state, ball_state):
        car = car_state
        ball = ball_state
        #Rcar_to_world
        r = -1*car.euler.roll #rotation around roll axis to get car to world frame
        p = -1*car.euler.pitch #rotation around pitch axis to get car to world frame
        y = car.euler.yaw #rotation about the world z axis to get the car to the world frame
        self.Rx = np.matrix([[1, 0, 0], [0, math.cos(r), -1*math.sin(r)], [0, math.sin(r), math.cos(r)]])
        self.Ry = np.matrix([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-1*math.sin(p), 0, math.cos(p)]])
        self.Rz = np.matrix([[math.cos(y), -1*math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        #Order of rotations from car to world is z then y then x
        self.Rinter = np.matmul(self.Rz, self.Ry)
        self.Rcar_to_world = np.matmul(self.Rinter, self.Rx)

        #Rworld_to_car
        r = car.euler.roll #rotation around roll axis to get world to car frame
        p = car.euler.pitch #rotation around pitch axis to get world to car frame
        y = -1*car.euler.yaw #rotation about the world z axis to get world to car frame
        self.Rx = np.matrix([[1, 0, 0], [0, math.cos(r), -1*math.sin(r)], [0, math.sin(r), math.cos(r)]])
        self.Ry = np.matrix([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-1*math.sin(p), 0, math.cos(p)]])
        self.Rz = np.matrix([[math.cos(y), -1*math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        #Order of rotations from world to car is x then y then z
        self.Rinter = np.matmul(self.Rx, self.Ry)
        self.Rworld_to_car = np.matmul(self.Rinter, self.Rz)

        #Rworld_to_ball
        ux = np.array([1.,0.,0.])
        uy = np.array([0.,1.,0.])
        uz = np.array([0.,0.,1.])
        Pb = np.array([ball.position.x, ball.position.y, ball.position.z])
        Pc = np.array([car.position.x, car.position.y, car.position.z])
        Pbc = np.subtract(Pb, Pc) #Get vector to ball from car in car coordinates

        xyz = np.cross(ux, Pbc) #xyz of quaternion is rotation between Pbc and unit x vector
        w = math.sqrt(1 * ((Pbc.item(0) ** 2) + (Pbc.item(1) ** 2) + (Pbc.item(2) ** 2))) + np.dot(ux, Pbc) #scalr of quaternion
        qbcworld = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2))

        #Quaternions for rotations between frames ALL COORDINATE SYSTEMS ORIGINS ARE AT THE CARS MASS CENTER
        self.Qcar_to_world = Quaternion(matrix = self.Rcar_to_world).normalised
        self.Qworld_to_car = Quaternion(matrix = self.Rworld_to_car).normalised
        self.Qworld_to_ball = Quaternion(matrix = qbcworld.rotation_matrix).normalised

        #Random rotation to try to point to
        r = 1
        p = 0
        y = 1

        self.Rx = np.matrix([[1, 0, 0], [0, math.cos(r), -1*math.sin(r)], [0, math.sin(r), math.cos(r)]])
        self.Ry = np.matrix([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-1*math.sin(p), 0, math.cos(p)]])
        self.Rz = np.matrix([[math.cos(y), -1*math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        #Order of rotations from world to car is x then y then z
        self.Rinter = np.matmul(self.Rx, self.Ry)
        self.Rworld_to_point = np.matmul(self.Rinter, self.Rz)

        self.Qworld_to_point = Quaternion(matrix = self.Rworld_to_point).normalised.normalised
        # print('r', r, 'p', p, 'y', y, 'q', self.Qtocar)

        ux_world = np.array([1.,0.,0.])
        uy_world = np.array([0.,1.,0.])
        uz_world = np.array([0.,0.,1.])
        self.ux_world = np.array([1.,0.,0.])
        self.uy_world = np.array([0.,1.,0.])
        self.uz_world = np.array([0.,0.,1.])
        self.headingx = self.Qworld_to_car.normalised.rotate(ux_world)
        self.headingy = self.Qworld_to_car.normalised.rotate(uy_world)
        self.headingz = self.Qworld_to_car.normalised.rotate(uz_world)
        #get point vectors for car and ball


        Pb_world = np.array([ball.position.x, ball.position.y, ball.position.z]) #multiply by negative one to keep axis orientations consistent between car and world
        Pc_world = np.array([car.position.x, car.position.y, car.position.z])
        self.Pbc_world = np.subtract(Pb_world, Pc_world) #Get vector to ball from car in world coordinate system but origin at car center
        self.Pbc_world_normalised = self.Pbc_world/np.linalg.norm(self.Pbc_world, axis = 0)
        self.Pbc_world_magnitude = np.linalg.norm(self.Pbc_world, axis=0)


        xyz = np.cross(self.headingx, self.Pbc_world/np.linalg.norm(self.Pbc_world, axis=0)) #xyz of quaternion is rotation between the UNIT Pbc vector and normalised x vector
        w = math.sqrt(1 * ((self.Pbc_world.item(0) ** 2) + (self.Pbc_world.item(1) ** 2) + (self.Pbc_world.item(2) ** 2))) + np.dot(ux_world, self.Pbc_world) #scalr of quaternion
        self.qbcworld = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2))
        # eulerAngles = toEulerAngle(qbcworld.normalised) #convert quaternion to euler angles
        # print("Pbc", self.Pbc)

        #Get quaternion pointing to ball in car coordinates
        self.Pbc_car = self.Qworld_to_car.normalised.rotate(self.Pbc_world)
        self.P0_world = np.array([1,0,0])
        self.P0_car = self.Qworld_to_car.normalised.rotate(self.P0_world)
        self.P1_car = np.array([1,0,0])
        self.P1_world = self.Qcar_to_world.normalised.rotate(self.P1_car)
        vector = self.Qcar_to_world.rotate(self.Pbc_car)
        # print("Pbc_world", self.Pbc_world, 'Pbc_car', self.Pbc_car, 'Pbc_world2', vector, 'r:', car.roll, 'p:', car.pitch, 'y:', car.yaw)

        #Rotational Velocity conversion from world to car coordinates
        wx_world = np.array([car.ang_vel.x, 0, 0])
        wy_world = np.array([0, car.ang_vel.y, 0])
        wz_world = np.array([0, 0, car.ang_vel.z])

        self.wx_car = self.toCarCoordinates(wx_world)
        self.wy_car = self.toCarCoordinates(wy_world)
        self.wz_car = self.toCarCoordinates(wz_world)
        self.w_car = self.wx_car + self.wy_car + self.wz_car

        # print('wx_car:', self.wx_car, 'wy_car', self.wy_car, 'wz_car', self.wz_car)
    def getDesiredQuaternion(self, vector):#vector is the 3d point vector to where we want the car to face in world coordinates / attitude is the current attitude of the car [roll, pitch, yaw]
        #Get quaternion that rotates world coordinates to car Coordinates
        xyz = np.cross(self.ux_world, vector/np.linalg.norm(vector, axis=0)) #xyz of quaternion is rotation between the UNIT Pbc vector and normalised x vector
        w = math.sqrt(1 * ((vector.item(0) ** 2) + (vector.item(1) ** 2) + (vector.item(2) ** 2))) + np.dot(self.ux_world, vector) #scalr of quaternion
        desired = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2))
        return desired
        #Convert vector to Car Coordinates

        #Convert new vector in car coordinates to a quaternion

    def getVectorToBall_world(self):
        vector = self.Qcar_to_world.rotate(self.Pbc_car)

        return vector

    def toWorldCoordinates(self, vector):
        vec = self.Qcar_to_world.rotate(vector)
        return vec

    def toCarCoordinates(self, vector): #vector is the 3d point vector to rotate to car coordintaes
        #get world to car quaternion
        vec = self.Qworld_to_car.rotate(vector)
        return vec

    def createQuaternion_world_at_car(self, point):
        #convert point into quaternion
        #point MUST be in world axes with origin at the car
        #Rworld_to_ball
        ux = np.array([1.,0.,0.])
        uy = np.array([0.,1.,0.])
        uz = np.array([0.,0.,1.])
        Ppoint = np.array([point.item(0), -1*point.item(1), -1*point.item(2)])
        #Pcar = np.array([car.item(0), car.item(1), car.item(2)]) #negate z because z axis for car is pointed downwards
        P = Ppoint #np.subtract(Ppoint, Pcar) #Get vector to ball from car in car coordinates

        xyz = np.cross(ux, P) #xyz of quaternion is rotation between Pbc and unit x vector
        w = math.sqrt(1 * ((P.item(0) ** 2) + (P.item(1) ** 2) + (P.item(2) ** 2))) + np.dot(ux, P) #scalr of quaternion
        Qworld_to_point = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2)).normalised

        #return quaternion
        return Qworld_to_point

    def createQuaternionFromHeading(self, heading):
        try:
            #convert point into quaternion
            #point MUST be in world axes with origin at the car
            #Rworld_to_ball
            ux = np.array([1.,0.,0.])
            uy = np.array([0.,1.,0.])
            uz = np.array([0.,0.,1.])
            Ppoint = np.array([heading[0], heading[1], -1*heading[2]])
            Ppoint = Ppoint/np.linalg.norm(Ppoint)
            #Pcar = np.array([car.item(0), car.item(1), car.item(2)]) #negate z because z axis for car is pointed downwards
            P = Ppoint #np.subtract(Ppoint, Pcar) #Get vector to ball from car in car coordinates

            xyz = np.cross(ux, P) #xyz of quaternion is rotation between Pbc and unit x vector
            w = math.sqrt(1 * ((P[0] ** 2) + (P[1] ** 2) + (P[2] ** 2))) + np.dot(ux, P) #scalr of quaternion
            Qworld_to_point = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2)).normalised

            #return quaternion
            return Qworld_to_point
        except:
            self.PrintException()
            return Quaternion(1, 0, 0, 0)

    def createQuaternion_world_at_car2(self, point, car):
        #convert point into quaternion
        #point MUST be in world axes with origin at the car
        #Rworld_to_ball
        ux = np.array([1.,0.,0.])
        uy = np.array([0.,1.,0.])
        uz = np.array([0.,0.,1.])
        Ppoint = np.array([point.item(0), -1*point.item(1), -1*point.item(2)])
        #Pcar = np.array([car.item(0), car.item(1), car.item(2)]) #negate z because z axis for car is pointed downwards
        P = Ppoint #np.subtract(Ppoint, Pcar) #Get vector to ball from car in car coordinates

        xyz = np.cross(ux, P) #xyz of quaternion is rotation between Pbc and unit x vector
        w = math.sqrt(1 * ((P.item(0) ** 2) + (P.item(1) ** 2) + (P.item(2) ** 2))) + np.dot(ux, P) #scalr of quaternion
        Qworld_to_point = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2)).normalised

        #return quaternion
        return Qworld_to_point
    def createQuaternion_from_point_and_roll(self, point, roll):
        r = roll
        p = -1 * math.atan2(point[2], math.sqrt((point[0]*point[0]) + (point[1] * point[1])))
        y = -1 * math.atan2(point[0], point[1]) + math.pi/2

        Rx = np.matrix([[1, 0, 0], [0, math.cos(r), -1*math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.matrix([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-1*math.sin(p), 0, math.cos(p)]])
        Rz = np.matrix([[math.cos(y), -1*math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        #Order of rotations from car to world is z then y then x
        Rinter = np.matmul(Rz, Ry)
        R = np.matmul(Rinter, Rx)

        Qto_point = Quaternion(matrix = R).normalised.conjugate.normalised
        # print(r, p, y)
        return Qto_point

    # Convert from rocket league's coordinate system which is totally messed up to a coherent coordinate system, to keep the math easy to follow without a ridiculous amount of negatives
    def convert_to_standard_coordinates(self, car, ball):
        None
