from rlbot.agents.base_agent import BaseAgent
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.game_data_struct import GameTickPacket
# from controller_input import controller
from simulator_utilities import GUI, convert_from_euler_angles, rotation_to_quaternion, State, Trajectory, euler_to_rotation_to_quaternion
from threading import Thread
from tkinter import Tk
import numpy as np
from pyquaternion import Quaternion
import render_util


class Agent(BaseAgent):
    def initialize_agent(self):
        # Initialize GUI class, put mainloop() function in thread to it can run concurrently with get_output()
        self.gui = GUI()
        self.gui_thread = Thread(target=self.gui_loop, daemon = False)
        self.gui_thread.start()
        if __name__ != '__main__': self.default_state()

    def get_output(self, game_tick_packet):

        # print(self.gui.test_val) # Gui updating test
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
            _ = render_util.sanity_checking(self, game_tick_packet)
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

            #get some data
            rotation = packet.game_cars[self.index].physics.rotation
            roll = rotation.roll
            pitch = rotation.pitch
            yaw = rotation.yaw
            todeg = 180/np.pi

            # get orientation matrix
            s = State()
            s.init_from_packet(packet, self.index)
            # rot_mat, _ = euler_to_rotation_to_quaternion(s)
            rot_mat = convert_from_euler_angles(s)

            # get quaternion
            quat = Quaternion(matrix=rot_mat)
            quat = quat.unit

            # define desired quaternion to point towards, get it from gui for ease
            # find the entry names for the quaternion
            # index = self.gui.entry_names.index("car|orientation|i")
            # vals = self.gui.entrys
            # desired = Quaternion([vals[index+3].get(), vals[index].get(), vals[index+1].get(), vals[index+2].get()])
            
            # Get ultitliteis params from gui for debugging
            e = self.gui.utilities_entrys
            ename = self.gui.utilities_entrys_names
            kq = float(e[ename.index('kq')].get())
            kw = float(e[ename.index('kw')].get())
            w = float(e[ename.index('w')].get())
            i = float(e[ename.index('i')].get())
            j = float(e[ename.index('j')].get())
            k = float(e[ename.index('k')].get())
            wx = float(e[ename.index('wx')].get())
            wy = float(e[ename.index('wy')].get())
            wz = float(e[ename.index('wz')].get())
                                    

            desired = Quaternion([w,i,j,k])

            # Get all necessary quatenrion and vectors for controller
            current = quat
            current2 = Quaternion(rotation_to_quaternion(rot_mat))
            desired = desired.unit
            av = packet.game_cars[self.index].physics.angular_velocity
            omega_current = np.array([av.x, av.y, av.z], dtype=np.float64)
            omega_desired = np.array([wx, wy, wz])
            


            # do simple quaternion p control feedback
            torques = p_quat_control(current, desired, omega_current, omega_desired, kq, kw)

            # set controller
            T_r = 36.07956616966136 # torque coefficient for roll
            T_p = 12.14599781908070 # torque coefficient for pitch
            T_y = 8.91962804287785 # torque coefficient for yaw
            controller_state = SimpleControllerState()
            controller_state.roll = max(min(float(torques.item(0)/T_r), 1), -1) 
            controller_state.pitch = max(min(float(torques.item(1)/T_p), 1), -1)
            controller_state.yaw = max(min(float(torques.item(2)/T_y), 1), -1)

            # Print stuff
            print('roll: ' + str(int(roll*todeg)) + ' p: ' + str(int(pitch*todeg)) + ' y: ' + str(int(yaw*todeg)))
            print('angT_vel:' + str(omega_current))
            # print('me: ' + str(current) + ' other: ' + str(current2))
            # return controller
            return controller_state

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


