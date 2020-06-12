import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import CarState
from UsefulConversions import ZYX_to_quaternion, quaternionToZYX
from pyquaternion import Quaternion
import sys
import linecache

class Controller():
    def PrintException(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

    def __init__(self):
        self.on = False # Turn the controller on or off with this variable. Essentially this allows the controller to set inputs on the joypad
        self.t0 = 0.0 # The initial game time that the trajectory is started
        self.t_now = 0.0 # The running realtime of the game to follow the trajectory

        # current car state
        self.currentState = CarState()

        # Trajectory Data Types
        self.ts = []
        self.sx = []
        self.sy = []
        self.sz = []
        self.vx = []
        self.vy = []
        self.vz = []

        #Aerial specific trajectory data
        self.q_norm = []
        self.q_omega = []
        self.heading = []

        #Driving Specific trajectory data
        self.yaw = []
        self.omega = []
        self.curvature = []

        # Input Data Types
        self.a = []  # Thruster input

        #Aerial specific input data
        self.alpha = [] # Torque input

        #Drivi1ng specific input data
        self.turning = [] # Steering input

        # joypad
        self.joypadState = SimpleControllerState() # Updating joypad state to read in get_output function for base agent

        # Live variables to make rendering easier
        self.Qcur = Quaternion()# The current quaternion that the controller is getting
        self.Qerr = Quaternion()
        self.Qdes = Quaternion()

        # Controller Gains
        self.kq = np.array([500,500,500])
        self.kw = np.array([100,100,100])

    def openLoop(self):
        if(self.on == True):
            # Find the correct input data to use based on time
            deltaT = self.t_now - self.t0
            idx = 0

            for i in range(0, len(self.ts) - 1):
                idx = i
                if(idx == len(self.ts)):
                    break
                if(deltaT < self.ts[idx+1]):
                    break
            # idx = np.searchsorted(self.ts, deltaT, side="left")
            # if (idx > 0) and (idx == len(self.ts) or math.fabs(t - self.ts[idx-1]) < math.fabs(t - self.ts[idx])):
            #     index = idx-1
            # else:
            #     index = idx

            # Stop trajectory
            if(deltaT > self.ts[-1]):
                self.on = False

            print("index: ", idx)
            self.joypadState.steer = self.getTurning(self.vx[idx], self.vy[idx], self.curvature[idx])
            # self.joypadState.steer = self.turning[idx]

            if(self.a[idx] > 0.9):
                self.joypadState.boost = 1
                self.joypadState.throttle = 1
            else:
                self.joypadState.boost = 0
                self.joypadState.throttle = 0.03

            return self.joypadState
        else:
            return SimpleControllerState()

    def getTurning(self, vx, vy, curvature):
        #getting the actual steering value since optmizer uses a polynomial approximation
        curvature_max = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_for_steer = np.array([0, 500, 1000, 1500, 1750, 2300])
        velocity = np.linalg.norm(np.array([vx, vy]))
        curvature_current = np.interp(velocity, v_for_steer, curvature_max)

        c = curvature / curvature_current
        print('c: ', c)
        return -1*c

    def getNormalizedTurning(self, u, vx, vy):
        #getting the actual steering value since optmizer uses a polynomial approximation
        curvature_max = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_for_steer = np.array([0, 500, 1000, 1500, 1750, 2300])
        velocity = np.linalg.norm(np.array([vx, vy]))
        curvature_current_max = np.interp(velocity, v_for_steer, curvature_max)

        c = u / curvature_current_max

        return c

    def getHeadingError(self, yaw_ref):
        # get unit veclotiy vector
        heading_ref = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0])

        yaw = self.currentState.physics.rotation.yaw
        heading_cur = np.array([np.cos(yaw), np.sin(yaw), 0])

        heading_ref_mag = np.linalg.norm(heading_ref)
        heading_cur_mag = np.linalg.norm(heading_cur)

        heading_error = np.arcsin(np.cross(heading_ref, heading_cur) / (heading_ref_mag * heading_cur_mag))
        # print("heading ref: ", heading_ref)
        # print("heading mag: ", heading_mag)
        print("heading error: ", heading_error)
        return heading_error[2]

    def getVelocityError(self, vx, vy):
        v_mag_ref = np.sqrt((vx*vx) + (vy*vy))
        vx_cur = self.currentState.physics.velocity.x
        vy_cur = self.currentState.physics.velocity.y
        v_mag_cur = np.sqrt((vx_cur * vx_cur) + (vy_cur * vy_cur))
        v_error = v_mag_ref - v_mag_cur
        # print("v_error: ", v_error)
        return v_error

    def getLateralError(self, traj_idx):
        # Reference position of trajectory
        x_ref = self.sx[traj_idx]
        y_ref = self.sy[traj_idx]
        pos_ref = np.array([x_ref, y_ref, 0])

        # current car position
        x_cur = self.currentState.physics.location.x
        y_cur = self.currentState.physics.location.y
        pos_cur = np.array([x_cur, y_cur, 0])

        # delta phi (delta yaw)
        vx_ref = self.vx[traj_idx]
        vy_ref = self.vy[traj_idx]
        v_ref = np.array([vx_ref, vy_ref, 0])
        v_ref_mag = np.linalg.norm(v_ref)
        vx_cur = self.currentState.physics.velocity.x
        vy_cur = self.currentState.physics.velocity.y
        v_cur = np.array([vx_cur, vy_cur, 0])
        v_cur_mag = np.linalg.norm(v_cur)
        phi = np.arcsin((np.cross(v_ref, v_cur)/(v_ref_mag * v_cur_mag))[2]) # Getting heading error angle

        # error vector
        if(v_ref_mag != 0):
            v_ref_unit = np.array([vx_ref/v_ref_mag, vy_ref/v_ref_mag, 0]) # Velocity reference unit to find lateral component of error
        else:
            v_ref_unit = np.array([0, 0, 0])
        error = np.array([pos_cur - pos_ref]) # total error vector
        lat_error = np.linalg.norm(np.subtract(error, (v_ref_unit * np.dot(error, v_ref_unit)))) # Getting lateral componnent of error scalar value

        # Calculate lateral error sign
        lat_error_sign = np.sign(np.cross(pos_cur, pos_ref)[2]) # Getting lateral error sign
        # print("phi: ", phi)
        print("lateral e: ", lat_error)
        return (lat_error * lat_error_sign) + (phi * v_cur_mag) # Lateral error plus lookahead (using v_cur_mag as a variable lookahead value it gets larger as your velocity gets larger this may help)


    def feedFoward(self):
        None

    def feedBack(self):
        # Determine which index we are at in trajectory
        if(self.on == True):
            # Find the correct input data to use based on time
            deltaT = self.t_now - self.t0
            idx = 0

            for i in range(0, len(self.ts) - 1):
                idx = i
                if(idx == len(self.ts)):
                    break
                if(deltaT < self.ts[idx+1]):
                    break
            # idx = np.searchsorted(self.ts, deltaT, side="left")
            # if (idx > 0) and (idx == len(self.ts) or math.fabs(t - self.ts[idx-1]) < math.fabs(t - self.ts[idx])):
            #     index = idx-1
            # else:
            #     index = idx

            # Stop trajectory
            if(deltaT > self.ts[-1]):
                self.on = False


            # Feedback gains
            ks = -0.5 # heading error feedback gain
            ke = -0.01 # lateral error feedback gain
            ka = 1 # Acceleratoin feedback gain

            # Normalize u vector inputs
            # u_heading_normalized = self.getHeadingError(self.yaw[idx]))
            # u_lateral_normalized = self.getLaterError(idx)

            # Set steering and boost
            # Here i get a normalized turning parameter in relation to the current maximum possible turning curvature which is velocity dependent, since my u vector should be in the units of curvature I believe.
            # This is adjusting the u value in relation to the controller output to a changing maximum value (will this cause unwanted behavior?)
            presteer = self.getNormalizedTurning((ks * self.getHeadingError(self.yaw[idx])) + (ke * self.getLateralError(idx)), self.currentState.physics.velocity.x, self.currentState.physics.velocity.y)
            steer = np.clip(presteer, -1, 1)
            # steer = (np.clip((ks * self.getHeadingError(self.yaw[idx])) + (ke * self.getLateralError(idx)), -1, 1))
            boost = np.clip(np.sign(self.getVelocityError(self.vx[idx], self.vy[idx])), 0, 1)

            print("steer: ", steer)
            print("boost: ", boost)

            #Set joypad values
            self.joypadState.steer = float(steer)
            self.joypadState.boost = float(boost)

            return self.joypadState
        else:
            return SimpleControllerState()

    def aerialFeedForward(self):
        if(self.on == True):
            # Find the correct input data to use based on time
            deltaT = self.t_now - self.t0
            idx = 0

            for i in range(0, len(self.ts) - 1):
                idx = i
                if(idx == len(self.ts)):
                    break
                if(deltaT < self.ts[idx+1]):
                    break
            # idx = np.searchsorted(self.ts, deltaT, side="left")
            # if (idx > 0) and (idx == len(self.ts) or math.fabs(t - self.ts[idx-1]) < math.fabs(t - self.ts[idx])):
            #     index = idx-1
            # else:
            #     index = idx

            # Stop trajectory
            if(deltaT > self.ts[-1]):
                self.on = False

            print("index: ", idx)
            self.joypadState.roll = self.q_omega[1].value[idx]
            self.joypadState.pitch = self.q_omega[2].value[idx]
            self.joypadState.yaw = self.q_omega[3].value[idx]*-1
            # self.joypadState.steer = self.turning[idx]

            if(self.a[idx] > 0.0):
                self.joypadState.boost = 1
                self.joypadState.throttle = 1
            else:
                self.joypadState.boost = 0

            return self.joypadState
        else:
            return SimpleControllerState()

    def aerialFeedBack(self, coord):
        try:
            if(self.on == True):
                # Find the correct input data to use based on time
                deltaT = self.t_now - self.t0
                idx = 0

                for i in range(0, len(self.ts)-1):
                    idx = i
                    if(idx == len(self.ts)):
                        break
                    if(deltaT < self.ts[idx+1]):
                        break
                # idx = np.searchsorted(self.ts, deltaT, side="left")
                # if (idx > 0) and (idx == len(self.ts) or math.fabs(t - self.ts[idx-1]) < math.fabs(t - self.ts[idx])):
                #     index = idx-1
                # else:
                #     index = idx

                # Stop trajectory
                if(deltaT > self.ts[-1]):
                    self.on = False

                # Qw2c = self.q_norm[idx]

                # wdes = np.array([self.q_omega[1][idx], self.q_omega[2][idx], self.q_omega[3][idx]])
                wdes = np.array([0,0,0])
                wcur = coord.w_car


                #define gains
                # kq = np.array([150, 150., 150.]) #quaternion error gain
                kq = self.kq
                # kw = np.array([15,15,15]) #omega error gain
                # kw = np.array([15,20,20]) #omega error gain
                kw = self.kw
                # kw = np.array([0,0,0])

                # T_r = 38.34; # torque coefficient for roll
                # T_p = 12.46; # torque coefficient for pitch
                # T_y =   9.11; # torque coefficient for yaw
                T_r = 36.07956616966136; # torque coefficient for roll
                T_p = 12.14599781908070; # torque coefficient for pitch
                T_y =   8.91962804287785; # torque coefficient for yaw
                # Get Heading vector car currently points in
                Qcur = coord.Qcar_to_world.unit

                self.Qcur = Qcur

                vec = np.array([1/np.linalg.norm(np.array([1,1,0])), 1/np.linalg.norm(np.array([1,1,0])), 0])
                # Qdes = coord.createQuaternion_world_at_car(vec).unit
                h = [self.heading[0][idx], self.heading[1][idx], -1*self.heading[2][idx]]
                Qdes = coord.createQuaternionFromHeading(h)
                self.Qdes = Qdes

                #get error Quaternion
                Qerr = Qdes.normalised * Qcur.normalised.conjugate
                self.Qerr = Qerr

                # print('Qerr:', Qerr)
                q = Qerr.imaginary

                #check for q0 < 0 and if this is true negate axis to give closest rotation
                if(Qerr.scalar < 0):     #use the conjugate of Qerr
                    q = -1*q



                werr = np.array(np.subtract(wdes, wcur))

                # q = np.matrix(qerrnew)
                # print("term1:", kq.T * q, " term2: ", kw.T * werr)
                torques = -1*(kq.T * q) - (kw.T * werr)
                # print(torques)
                # Normalize controller input by torques coefficients for each axis
                # torques[0] = torques[0] / T_r
                # torques[1] = torques[1] / T_p
                # torques[2] = torques[2] / T_y
                # print('theta', theta, 'torques', torques, 'qerr', Qerr.unit, 'qw2c', Qw2c.unit, 'qc2b', Qw2b.unit)
                # print(q)

                # self.joypadState.roll = 0 # Shut of roll control, let pitch and yaw only do control
                self.joypadState.roll = max(min(float(torques.item(0)), 1), -1)
                self.joypadState.pitch = max(min(float(torques.item(1)), 1), -1)
                self.joypadState.yaw = max(min(float(torques.item(2)), 1), -1)*-1
                # print("Heading: ", h, " | Qdes: ", Qdes, "| Qcur: ", Qcur, " | Qerr: ", Qerr, " | torques: ", torques[0])
                # print("wdes: ", wdes, " | wcur: ", wcur, " | werr: ", werr, " | torques: ", torques)
                # self.joypadState.steer = self.turning[idx]
                # This needs to be come a parallel velocity controller (probably also need to do some timescale characterization)
                if(self.a[idx] > 0.1):
                    self.joypadState.boost = 1
                    # self.joypadState.throttle =
                else:
                    self.joypadState.boost = 0

                return self.joypadState
            else:
                return SimpleControllerState()
        except Exception as e:
            self.PrintException()
            return SimpleControllerState()

    def setTrajectoryData(self, ts, sx, sy, vx, vy, yaw, omega, curvature):
        self.ts = ts
        self.sx = sx
        self.sy = sy
        self.vx = vx
        self.vy = vy
        self.yaw = yaw
        self.omega = omega
        self.curvature = curvature

    def setAerialTrajectoryData(self, ts, sx, sy, sz, vx, vy, vz, heading, q, q_omega):
        self.ts = ts
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.heading = heading
        self.q = q
        self.q_omega = q_omega

    def setInputData(self, a, turning):
        self.a = a
        self.turning = turning

    def setAerialInputData(self, a, alpha):
        self.a = a
        self.alpha = alpha

    def setTNOW(self, t):
        self.t_now = t

    def setCurrentState(self, cs):
        self.currentState = cs

    def setAerialControllerGains(self, kq, kw):
        self.kq = kq
        self.kw = kw
