from rlbot.agents.base_agent import BaseAgent
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.game_data_struct import GameTickPacket
# from controller_input import controller
from simulator_utilities import GUI, convert_from_euler_angles, rotation_to_quaternion, State, Trajectory, Controller, euler_to_right_handed_rotation_and_quaternion, euler_to_left_handed_rotation_and_quaternion
from threading import Thread
from threading import Thread
from tkinter import Tk
import numpy as np
from pyquaternion import Quaternion
import render_util
from coordinate_system_util import CoordinateSystems


class Agent(BaseAgent):
    def initialize_agent(self):
        # Initialize GUI class, put mainloop() function in thread to it can run concurrently with get_output()
        self.gui = GUI()
        self.gui_thread = Thread(target=self.gui_loop, daemon = False)
        self.gui_thread.start()
        if __name__ != '__main__': self.default_state()

    def get_output(self, game_tick_packet):

        controller_state = self.gui.run_manager.run(game_tick_packet, self.index)

        if __name__ != '__main__': 
            try:
                render_util.basic_render(self, game_tick_packet, self.gui.sim_state.trajectory)
            except:
                render_util.basic_render(self, game_tick_packet, Trajectory())


        # The following code is just retesting my quaternion math as a sanity check before i start writing the class strutures that handle all the math
        # controller_state = controller.get_output()# Get raw controller state from joypad

        if self.gui.run_manager.running == False:
            controller_state = self.math_debugging(game_tick_packet)
            # _ = render_util.sanity_checking(self, game_tick_packet)
        else:
            print_trajectory(self.gui.sim_state.trajectory)
        return controller_state

    def gui_loop(self):
        try:
            self.gui.__init__() # Re-initialize the GUI for the fuck of it
            while(1): # Continuiously run mainloop() to update data from the gui 
                self.gui.master.mainloop()
        except Exception as e:
            print(e)

    def default_state(self):
        # Force bot to specified position and rotation for testing
        pi = np.pi
        class Empty(): None
        o=Empty()
        o.roll = 0.5*pi
        o.pitch = 0.25*pi
        o.yaw = 0*pi 
        p = [0, 0, 500]
        car_state = CarState(Physics(location=Vector3(x=p[0], y=p[1], z=p[2]), rotation=Rotator(o.pitch, o.yaw, o.roll)))
        # car_state = CarState(Physics(location=Vector3(x=p[0], y=p[1], z=p[2])))
        game_info_state = GameInfoState(world_gravity_z=0.0001)
        game_state = GameState(cars={self.index: car_state}, game_info = game_info_state)
        self.set_game_state(game_state)
    


    def math_debugging(self, packet):


            #get Euler angles from packet data
            rotation = packet.game_cars[self.index].physics.rotation
            roll = rotation.roll
            pitch = rotation.pitch
            yaw = rotation.yaw
            todeg = 180/np.pi

            #Initialize State from packet
            s = State()
            s.init_from_packet(packet, self.index)
            # initlaize ball from packet
            b = State()
            b.init_from_packet(packet, 'ball')

            #Update coordinate systems
            cs = CoordinateSystems()
            cs.update(s, b)

            # create quaternion from rotation matrix


            # Get utilities params from gui for debugging
            e = self.gui.utilities_entrys
            ename = self.gui.utilities_entrys_names
            kq = float(e[ename.index('kq')].get()) # controller q gain
            kw = float(e[ename.index('kw')].get()) # controller qdot gain
            w = float(e[ename.index('w')].get()) # quaternion w
            i = float(e[ename.index('i')].get()) # quaternion i component
            j = float(e[ename.index('j')].get()) # quaternion j component
            k = float(e[ename.index('k')].get()) # quaternion k component
            wx = float(e[ename.index('wx')].get()) # angular velocity x
            wy = float(e[ename.index('wy')].get()) # angular velocity y
            wz = float(e[ename.index('wz')].get()) # angular velocity z
                                    


            # Get all necessary quatenrion and vectors for controller
            desired = Quaternion([w,i,j,k])
            desired = desired.unit
            av = packet.game_cars[self.index].physics.angular_velocity
            omega_current = np.array([av.x, av.y, av.z], dtype=np.float64)
            omega_desired = np.array([wx, wy, wz])
            


            # do simple quaternion p control feedback
            c = Controller('state_feed_back') # Initialize controller class to be of type state based feedback
            c.kq = kq  # Force update controller gains
            c.kw = kw  # Force update controller gains
                        # Create quaternion from rigid body tick
            qq = self.get_rigid_body_tick().players[self.index].state.rotation
            current_quat = Quaternion(w = qq.w, x=qq.x, y=qq.y, z=qq.z)

            torques = c.algorithm(current_quat, desired, omega_current, omega_desired)

            # set controller
            T_r = 36.07956616966136 # torque coefficient for roll
            T_p = 12.14599781908070 # torque coefficient for pitch
            T_y = 8.91962804287785 # torque coefficient for yaw
            controller_state = SimpleControllerState()
            controller_state.roll = max(min(float(torques.item(0)/T_r), 1), -1) # Convert raw torque value to controller input equivalent
            controller_state.pitch = max(min(float(torques.item(1)/T_p), 1), -1) # Limit between -1 and 1
            controller_state.yaw = max(min(float(torques.item(2)/T_y), 1), -1)

            # Print stuff
            q = self.get_rigid_body_tick().players[self.index].state.rotation
            p = [q.w, q.x, q.y, q.z]
            for elem in p:
                print(" | ", end="")
                print(elem, end="")
            print()
            # print(desired.inverse*current_quat)
            # print_rot_mat(s.rot_mat)
            # print_orientation_from_quaternion(s.orientation)
            # print('roll: ' + str(int(roll*todeg)) + ' p: ' + str(int(pitch*todeg)) + ' y: ' + str(int(yaw*todeg)))
            # print('angT_vel:' + str(omega_current))
            # print('me: ' + str(current) + ' other: ' + str(current2))
            # return controller
            return controller_state

def print_trajectory(t: Trajectory):
    print('----------------------------')
    for state in t.states:
        print(state.orientation)
    print('----------------------------')

def print_orientation_from_quaternion(q):
    vecx = q.inverse.rotate([1,0,0])
    vecy = q.inverse.rotate([0,1,0])
    vecz = q.inverse.rotate([0,0,1])
    print('vecx: ' + str(vecx) + ' vecy: ' + str(vecy) + ' vecz: ' + str(vecz))

def print_rot_mat(r):
    x = r[0, :]
    y = r[1,:]
    z = r[2,:]
    vec = [x,y,z]
    for v in vec:
        print(" | ", end=" ")
        for elem in np.nditer(v):
            print(round(float(elem), 2), end=" ")
    print()

def p_quat_control(current, desired, wc, wd, kq, kw):
    Qerr = desired * current
    Qerr = Qerr.unit
    q = Qerr.imaginary

    if Qerr.scalar < 0:
        q = -1*q

    # Convert omega_current to car coordinate system, we need to use the inverse of the state quaternion since we're going from world to car
    wx = current.inverse.rotate(np.array([wc[0], 0, 0]))
    wy = current.inverse.rotate(np.array([0, wc[1], 0]))
    wz = current.inverse.rotate(np.array([0, 0, -1*wc[2]])) # Remember to negate yaw since yaw rotation is backwards in RL
    wc = wx + wy + wz

    werr = np.array(np.subtract(wd, wc))
    torques = -1*(kq * q) - (kw * werr)
    # print(str(wc) + ' | ' + str(torques) + ' | ' + str(current))
    return torques



# Run this file with F5 in VSCode to run without the overhead of RLbot and the interface to rocket league. Only for debugging
if __name__ == "__main__":
    # Run class to make debugging code
    agent = Agent('name', 'team', 2)
    agent.initialize_agent()
    while(1):
        agent.get_output(GameTickPacket())


