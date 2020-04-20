# This py file will house classes for the GUI and optimizer/simulator class

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from tkinter import Tk, filedialog, Toplevel, Frame, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar, Checkbutton, Canvas, PhotoImage, NW
from util.vec import Vec3
from threading import Thread
from AerialOptimization import AerialOptimizer
from pyquaternion import Quaternion
import copy
import numpy as np
import dill
# dill.settings['recurse'] = True

class GUI():
    def __init__(self):
        self.master = Tk() # root of gui
        self.master.title('Simulator/Optimizer') # Give the window a title
        self.frame = Frame(self.master, height=400, width=400) # Place a frame inside the root
        self.frame.pack(fill='both', expand=True) # pack the frame into the root and let it change size if we change root window size
        self.sim_params = SimulationParameters()
        self.sim_state = SimulationState()
        self.initialize_buttons()
        self.initialize_sim_params_frame()

        self.sim_params_frame.withdraw()
        self.initialize_utilities_frame()
        self.utilities_frame.withdraw()
        self.run_manager = RunManager(SimulationState())


    # initialize the buttons on the main frame
    def initialize_buttons(self):
        self.load_sim_state_button = Button(self.frame, text='load sim state', command=lambda:self.load_sim_state())
        self.load_sim_state_button.pack(fill='both', expand=True)

        self.save_sim_state_button = Button(self.frame, text='save sim state', command=lambda:self.save_sim_state())
        self.save_sim_state_button.pack(fill='both', expand=True)

        self.show_controller_params_button = Button(self.frame, text='Show controller params', command=lambda:self.show_controller_params())
        self.show_controller_params_button.pack(fill='both', expand=True)

        self.show_utilities_button = Button(self.frame, text='Show utilities', command=lambda:self.show_utilities_frame())
        self.show_utilities_button.pack(fill='both', expand=True)

        self.show_sim_params_button = Button(self.frame, text='show sim params', command=lambda:self.show_sim_params())
        self.show_sim_params_button.pack(fill='both', expand=True)

        self.generate_sim_state_button = Button(self.frame, text='generate sim state', command=lambda:self.generate_sim_state())
        self.generate_sim_state_button.pack(fill='both', expand=True)

        self.plot_sim_button = Button(self.frame, text='Plot Sim', command=lambda:self.plot_sim())
        self.plot_sim_button.pack(fill='both', expand=True)   

        self.run_sim_button = Button(self.frame, text='Run Sim', command=lambda:self.run_sim())
        self.run_sim_button.pack(fill='both', expand=True)           

    # intialize the sim params frame which holds the initial conditions of the car and ball
    def initialize_sim_params_frame(self):
        self.sim_params_frame = Toplevel(self.master)
        heading_cells = ['car', 'ball']
        entry_cells = ['i', 'j', 'k', 'w']
        rows = ['position', 'velocity', 'orientation', 'ang_vel']
        labels = []
        entrys = []
        entry_names = []
        buttons = []

        row_counter = 0
        column_counter = 1
        # Make top level labels first
        for i, hcell in enumerate(heading_cells):
            labels.append(Label(self.sim_params_frame, text=hcell, bg='white', borderwidth=2, relief='groove'))
            labels[-1].grid(row=0, column=1+(i*len(entry_cells)), columnspan=len(entry_cells),sticky='nsew')
            for j, e in enumerate(entry_cells): # Make ijkw for each heading cell (car and ball)
                labels.append(Label(self.sim_params_frame, text=e))
                labels[-1].grid(row=1, column=column_counter)
                column_counter += 1
    
        row_counter = 2
        column_counter = 1

        for i, r in enumerate(rows): # add label to each row
            label = labels.append(Label(self.sim_params_frame, text=r))
            labels[-1].grid(row=row_counter, column=0)
            for k, h in enumerate(heading_cells): # Add entry for each ijkw for ball and car
                for j, ecell in enumerate(entry_cells):
                    entry_names.append(str(h) + '|' + str(r) +  '|'  + str(ecell)) # Append name of current entry into list to find entrys easier later
                    entrys.append(Entry(self.sim_params_frame))
                    entrys[-1].grid(row=row_counter, column=column_counter)
                    entrys[-1].insert(0, '0.0')
                    column_counter += 1
            row_counter += 1
            column_counter = 1
        
        self.labels = labels
        self.entrys = entrys
        self.entry_names = entry_names
        self.buttons = buttons

        print(entry_names)
        # Save sim params button
        self.save_sim_param_button = Button(self.sim_params_frame, text='Save Sim Params', command=lambda:self.save_sim_params())
        self.save_sim_param_button.grid(row=row_counter, column = 0, columnspan=9, sticky='nsew')
        row_counter += 1
        self.load_sim_param_button = Button(self.sim_params_frame, text='load sim params', command=lambda:self.load_sim_params())
        self.load_sim_param_button.grid(row=row_counter, column=0, columnspan=9, sticky='nsew')
        row_counter += 1
        self.hide_sim_param_button = Button(self.sim_params_frame, text='Hide This Window', command = lambda:self.sim_params_frame.withdraw()) # Hides the sim params window 
        self.hide_sim_param_button.grid(row=row_counter, column=0, columnspan=9, sticky='nsew')

    def initialize_controller_params_frame(self):
        None
    
    def initialize_utilities_frame(self):
        self.utilities_frame = Toplevel(self.master)
        utilities_entrys_names = []
        utilities_entrys = []
        utilities_labels = []
        label_names = ['w', 'i','j','k','kq', 'kw', 'wx', 'wy', 'wz']
        row_counter = 0
        column_counter = 0
        for j, e in enumerate(label_names): # Make all labels on left side
            utilities_labels.append(Label(self.utilities_frame, text=e))
            utilities_labels[-1].grid(row=row_counter, column=column_counter)
            row_counter += 1
        column_counter += 1
        row_counter = 0
        for j, name in enumerate(label_names):
            utilities_entrys_names.append(str(name))
            utilities_entrys.append(Entry(self.utilities_frame))
            utilities_entrys[-1].grid(row=row_counter, column=column_counter)
            utilities_entrys[-1].insert(0, '0.0')
            row_counter += 1
        self.utilities_entrys = utilities_entrys
        self.utilities_entrys_names = utilities_entrys_names
        e = self.utilities_entrys
        ename = self.utilities_entrys_names
        e[ename.index('kq')].delete(0, 'end')
        e[ename.index('kq')].insert(0, '100')
        e[ename.index('kw')].delete(0, 'end')
        e[ename.index('kw')].insert(0, '12')
        e[ename.index('w')].delete(0, 'end')
        e[ename.index('w')].insert(0, '1')

    # Export the data in the entry's from sim param frame into the simulation parameters class
    def export_sim_param_frame_data(self):
        #Initialize car state

        cardata, balldata = self.get_raw_entry_data()
        car_state = State()
        car_state.init_from_raw_data(cardata.position, cardata.velocity, cardata.orientation, cardata.ang_vel)
        ball_state = State()
        ball_state.init_from_raw_data(balldata.position, balldata.velocity, balldata.orientation, balldata.ang_vel)
        self.sim_params.initialize_from_raw(car_state, ball_state)
        print('no issues')
        #Initialize ball state

        #Initialize simparams class

    # takes the raw entrys data thats on the self.sim_params_frame, and parses them into attributes to be passed into
    # the State() class later for both the car and the ball (that happens in the export_sim_param_frame_data) function
    def get_raw_entry_data(self):
        class Empty(): None # Just an empty class that way the variable i create from this class has the __dict__ attribute so i can run getattr() on the member
        cardata = Empty() # Initialize car_data
        balldata = Empty() # Initialize ball_data
        for i, e in enumerate(self.entry_names):
            sub = e.split('|') # Search entry_names[i] string and split by the '|' character as the delimter
            var_name = str(sub[1]) # This is the variables name (position, velocity, orientation, ang_vel)
            try: # check if attribute exists
                if(sub[0] == 'car'):
                    getattr(cardata, var_name) # Testing trigger of exception if attribute name doesn't exist
                    vars(cardata)[var_name].append(float(self.entrys[i].get())) # If above runs, attribute exists, append data to attribute that has the same name, order of entrys on gui is always ijkw
                if(sub[0] == 'ball'):
                    getattr(balldata, var_name) # Do same but for the ball_data
                    vars(balldata)[var_name].append(float(self.entrys[i].get()))                    
            except: # if attribute doesn't exist, create it
                if(sub[0] == 'car'):
                    vars(cardata)[var_name] = [] # Create an attribute with var_name iniside car_data, initialize it to an empty array
                    vars(cardata)[var_name].append(float(self.entrys[i].get())) # Append the value in the entry correlated to that var_name
                if(sub[0] == 'ball'):
                    vars(balldata)[var_name] = [] # Do the same but for the ball_data
                    vars(balldata)[var_name].append(float(self.entrys[i].get()))

        print('pause debug')
        return cardata, balldata
    
    # Function to show the sim parameters window using the show sim params button
    def show_sim_params(self):
        self.sim_params_frame.deiconify()  
    def show_controller_params(self):
        self.controller_params_frame.deiconify()
    def show_utilities_frame(self):
        self.utilities_frame.deiconify()

    def run_sim(self):
        try:
            #Initialize RunManager
            self.run_manager = RunManager(self.sim_state)

            # Start run_manager
            self.run_manager.start = True
        except Exception as e:
            print(e)

    # Will take the simulation parameters, and run them through the optimzier/simulator
    # Then will generate the full SimulationState that is given to the Agent() class 
    # which will allow the Agent to run the simulation autnomously without needing to interact with the GUI
    def generate_sim_state(self):
        print(self.sim_params)
        self.sim_state = SimulationState()
        traj = Trajectory()
        optimizer = AerialOptimizer()
        optimizer.optimizeAerial(self.sim_params)
        optimizer.solve()
        traj.init_from_optimization_model(optimizer)
        import copy
        self.current_model = copy.deepcopy(optimizer)
        print('debug')

        # Initialize a controller with a 'feed_back" controller type
        controller = Controller('feed_back')

        # Initialize SimulationState
        from copy import deepcopy as dc
        self.sim_state.default_init(dc(traj), dc(controller), self.current_model)
        self.run_manager = RunManager(self.sim_state)

    def plot_sim(self):
        import AerialOptimization
        AerialOptimization.plotData(5, self.current_model)
        import matplotlib.pyplot as plt
        plt.show()

    def push_sim_params_to_frame(self, sim_params):
        for i, e in enumerate(self.entry_names):
            sub = e.split('|') # Search entry_names[i] string and split by the '|' character as the delimter
            ijkw = ['i', 'j', 'k', 'w']
            quat_ijkw = ['w', 'i', 'j', 'k'] # Need the other index order for the quaternion since its wijk instead of ijkw in the gui ugh
            obj_name = str(sub[0]) # This is car or ball
            var_name = str(sub[1]) # This is the variables name (position, velocity, orientation, ang_vel)

            try:
                key_index = ijkw.index(str(sub[2])) # Get index of vector (i, j, k) from entry names delimited sub string
                if(obj_name == 'ball'): # Do if ball
                    value = getattr(sim_params.ball_state, var_name) # Search for attribute with name from substring
                    self.entrys[i].delete(0, 'end') # Delete text in entry box
                    self.entrys[i].insert(0,str(value[key_index])) # Insert the value from the loaded attribute

                elif(obj_name == 'car'): # Do if car
                    if(var_name == 'orientation'): # Take special care if we're dealing with the orientation parameter since this is a Quaternion() object not a list of floats
                        quat_index = quat_ijkw.index(str(sub[2])) # Quaternion indexes [w, i, j, k], GUI is [i, j, k, w], get the index related to the quaternion by using the substring
                        quat = getattr(sim_params.car_state, var_name) # Get the quaternion object from the loaded sim_params
                        self.entrys[i].delete(0, 'end') # Delete text in entry
                        val = copy.deepcopy(quat.elements[quat_index]) # Get proper value from the quaternion object, using the quat index to choose the right one
                        print('entry index: ' + str(i) + 'quatindex: ' + str(quat_index) + 'elements: ' + str(val)) # Debugging
                        self.entrys[i].insert(0, str(val)) # Put value into entry box
                    else: # If parameter is not orientation proceed as normal, position, velocity, ang_vel are all just list of floats in the order [i, j, k] w is ommitted for these vectors
                        value = getattr(sim_params.car_state, var_name)
                        self.entrys[i].delete(0, 'end')
                        self.entrys[i].insert(0, str(value[key_index]))
            except:
                continue


    def load_sim_params(self):
        try:
            import sys
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            root = Tk()
            save_path = filedialog.askopenfilename(parent=root) # Ask user for file to load from
            sim_params = dill.load(open(save_path, 'rb'))
            self.sim_params = sim_params
            root.destroy()
            #TODO: Initialize self.sim_params_frame from the data loaded!
            self.push_sim_params_to_frame(self.sim_params)
        except Exception as e:
            AerialOptimizer.PrintException()
            root.destroy()
            print('try  loading again')

    def save_sim_params(self):
        # Save the simulation parameters in the local variable
        try:
            import sys
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            sys.path.append('D:/Documents/DeepToot/RLBOT/simulator_bot')
            self.export_sim_param_frame_data()
            root = Tk()
            save_path = filedialog.askopenfilename(parent=root) # Ask user to file to save to
            dill.dump(self.sim_params, open(save_path, 'wb'))
            root.destroy()
        except:            
            AerialOptimizer.PrintException()
            print("Try saving again")
            root.destroy()

    def load_sim_state(self):
        try:
            import dill
            import sys
            sys.path.append('D:\Documents\DeepToot\RLBOT\simulator_utilities')
            root = Tk()
            save_path = filedialog.askopenfilename(parent=root) # Ask user for file to load from
            self.sim_state = dill.load(open(save_path, 'rb'))
            root.destroy()
        except Exception as e:
            AerialOptimizer.PrintException()
            root.destroy()
            print('try  loading again')


    def save_sim_state(self):
        # Save the simulation parameters in the local variable
        try:
            import dill
            import sys
            sys.path.append('D:\Documents\DeepToot\RLBOT\simulator_utilities')
            root = Tk()
            save_path = filedialog.askopenfilename(parent=root) # Ask user to file to save to
            dill.dump(copy.deepcopy(self.sim_state), open(save_path, 'wb'))
            root.destroy()
        except:            
            AerialOptimizer.PrintException()
            print("Try saving again")
            root.destroy()


# Has all the necessary data to run the AerialOptimizer functions, will be passed into AerialOptimizer.optimize
# or whatever the name of the function is
class SimulationParameters():

    def __init__(self):
        self.ball_state = None
        self.car_state = None
        self.intialized = False
    
    def initialize_from_raw(self, car, ball):
        try:
            self.car_state = car
            self.ball_state = ball
            self.initialzed = True
        except:
            self.intialized = False
            self.car_state = None
            self.ball_state = None

# Simulation state will house all the data for a specific scenario, trajectory, initial states of ball/car, the type of controller to use
class SimulationState():

    
    def __init__(self):
        self.trajectory = []
        self.controller = []
        self.optimizer = []
    
    def default_init(self, traj, cont, opt):
        self.trajectory = traj
        self.controller = cont
        self.optimizer = opt


# The full state of the car or ball
class State():

    def __init__(self):
        # self.default_init(None, None)
        self.time = []
        self.position = []
        self.velocity = []
        self.orientation = Quaternion([1,0,0,0])
        self.ang_vel = []

    def default_init(self, data, index):
        # Check if data is a GameTickPacket, initialize from packet data if so
        if isinstance(data, GameTickPacket):
            self.init_from_packet(data, index)
        else:
            self.position = []
            self.velocity = []
            self.orientation = Quaternion([1,0,0,0])
            self.ang_vel = []
    
    # Initialize from a model (optimizer class) 
    # this model should have the following position, velocity, orientation, ang_vel
    # that way the full state can be made
    # index is which index in the vector data we are using to generate the state
    # index=0 would be the initial state of the traj etc....

    def init_from_model(self, model: AerialOptimizer, index):
        self.time = model.ts[index]
        for pos in model.position:
            self.position.append(pos[index])
        print(self.position)

        for vel in model.velocity:
            self.velocity.append(vel[index])
        print(self.velocity)

        q = model.orientation[index]
        self.orientation = Quaternion(w=q[0], x=q[1], y=q[2], z=q[2]).unit

        for ang in model.ang_vel:
            self.ang_vel.append(ang[index])

    def init_from_packet(self, packet, index):
        if index == 'ball':
            self.position = packet.game_ball.location
            self.velocity = packet.game_ball.velocity
            self.euler = packet.game_ball.orientation
            self.ang_vel = packet.game_ball.angular_velocity
            self.time = packet.game_info.seconds_elapsed
        else:
            self.position = packet.game_cars[index].physics.location
            self.velocity = packet.game_cars[index].physics.velocity
            self.euler = packet.game_cars[index].physics.rotation # {Pitch, Yaw, Roll}
            self.ang_vel = packet.game_cars[index].physics.angular_velocity
            self.other_data = packet.game_cars[index]
            self.time = packet.game_info.seconds_elapsed
            # Get data from packet about rotation
            rotation = packet.game_cars[index].physics.rotation
            roll = rotation.roll
            pitch = rotation.pitch
            yaw = rotation.yaw
            # rot_mat = convert_from_euler_angles(-1*roll, -1*pitch, yaw)
            rot_mat, _ = euler_to_rotation_to_quaternion(self)
            # get quaternion
            self.orientation = Quaternion(matrix=rot_mat)
            # self.orientation = Quaternion([1,0,0,0])


    def init_from_raw_data(self, position, velocity, orientation, angular_velocity):
        self.position = position
        self.velocity = velocity
        if isinstance(orientation, Quaternion):
            self.orientation = orientation
            print('orientation input argument is Quaternion object')
        else:
            self.orientation = Quaternion(w=orientation[3], x=orientation[0], y=orientation[1], z=orientation[2]) # gui vector order is ijkw, quaternion order is wijk
            print('orientation input arguemnt is NOT a Quaternion object')
        # self.quaternion
        self.ang_vel = angular_velocity


# A class that has an array of states that make up the entire trajectory
# Time vector is also here
class Trajectory():
    def __init__(self):
        self.states = []
        self.time = []
    # Model is the AerialOptimizer class that has been run.
    def init_from_optimization_model(self, model):
        import copy
        try:
            self.time = model.ts
            # Iterate through vectors from model and create states and append to traj states
            for i, m in enumerate(model.ts):
                s = State()
                s.init_from_model(model, i)
                self.states.append(s)
            # Check to make sure states and time vector are same length if not throw exception
            if(len(self.time) != len(self.states)):
                raise ValueError('The time vector and the states vector are not the same length')
        
        except Exception as e:
            print(e)
    
# RunManager will 
class RunManager():


    def __init__(self, sim_state):
        self.start = False # Start flag, so I can get t0 from game, and initialize RL engine
        self.running = False # Flat to know if the trajectory is running
        self.t_zero = 0.0
        self.current_state = State()
        self.sim_state = sim_state
    
    def run(self, packet: GameTickPacket, self_index):
        if(self.start):
            self.running = True
            self.t_zero = packet.game_info.seconds_elapsed
            self.start = False # Reset self.start so we don't trigger this if state ment again
        
        if(self.running):
            current_state = self.initialize_current_state(packet, self_index)
            return self.get_controls(current_state, self.sim_state.trajectory, self.t_zero)

    def initialize_current_state(self, packet, self_index):
        # Initialize current state
        current = State()
        current.init_from_packet(packet, self_index)
        return current

    def get_controls(self, state, trajectory, t_zero):
        ret = self.sim_state.controller.algorithm(state, trajectory, t_zero)
        if ret == 'finished':
            self.running = False
        else:
            return ret

class Controller():

    # controller_type is a string of the name of the function we want to use for the controller
    def __init__(self, controller_type):
        try:
            hasattr(self, controller_type)
            self.algorithm = getattr(self, controller_type)
        except:
            print('Controller does not have the type queried')

        self.kq = 100
        self.kw = 12
        self.T_r = 36.07956616966136 # torque coefficient for roll
        self.T_p = 12.14599781908070 # torque coefficient for pitch
        self.T_y = 8.91962804287785 # torque coefficient for yaw


    def feed_back(self, state, traj, t0):
        try:
            dt = float(state.time) - t0

            for i in range(0, len(traj.time) - 1):
                idx = i
                if(idx == len(traj.time)):
                    break
                if(dt < traj.time[idx+1]):
                    break
                if(dt > traj.time[-1]):
                    print('trajectory is finished')
                    return 'finished'
                    # Trajectory is done, do stuff here?



            # Get the current state in quaternion form
            current = state.orientation.unit
            desired = traj.states[idx].orientation.unit

            Qerr = desired*current
            Qerr = Qerr.unit
            q = Qerr.imaginary

            if Qerr.scalar < 0:
                q = -1*q

            wd = np.array([0,0,0]) # desired angular velocity
            w = state.ang_vel # ang vel non numpy form
            wc = np.array([w.x, w.y, w.z], dtype=np.float) # ang_vel numpy form

            # Convert omega_current to car coordinate system, we need to use the inverse of the state quaternion since we're going from world to car
            wx = current.inverse.rotate(np.array([wc[0], 0, 0]))
            wy = current.inverse.rotate(np.array([0, wc[1], 0]))
            wz = current.inverse.rotate(np.array([0, 0, -1*wc[2]])) # Remember to negate yaw since yaw rotation is backwards in RL
            wc = wx + wy + wz

            werr = np.array(np.subtract(wd, wc))
            torques = -1*(self.kq * q) + (self.kw * werr)
            controller_state = SimpleControllerState()
            controller_state.roll = max(min(float(torques.item(0)/self.T_r), 1), -1) 
            controller_state.pitch = max(min(float(torques.item(1)/self.T_p), 1), -1)
            controller_state.yaw = max(min(float(torques.item(2)/self.T_y), 1), -1)
            # print('torques: ' + str(torques))
            # print('ang_vel: ' + str(wc))
            # print('idx' + str(idx))
            print('desired: ' + str(desired) + ' current: ' + str(current))
            return controller_state
        except:
            AerialOptimizer.PrintException()
            print('whats the error?')
            print('will it update')
            return SimpleControllerState()

    def feed_forward(self, state, trajectory, t0):
        return SimpleControllerState()

    def convert_to_quaternion(self, state: State):
        if isinstance(state.orientation, Quaternion):
            return state.orientation.unit
        roll = state.orientation.roll
        pitch = state.orientation.pitch
        yaw = state.orientation.yaw
        # Negate roll and pitch since they use LHR
        rot_mat = euler_to_rotation_to_quaternion(state)
        return Quaternion(matrix=rot_mat).unit
        # return Quaternion(rotation_to_quaternion(rot_mat)).unit

def convert_from_euler_angles(state: State()):
    roll = -1*state.euler.roll
    pitch = -1*state.euler.pitch
    yaw = state.euler.yaw

    CR = np.cos(roll)
    SR = np.sin(roll)
    CP = np.cos(pitch)
    SP = np.sin(pitch)
    CY = np.cos(yaw)
    SY = np.sin(yaw)

    theta = np.zeros((3,3))

    # front direction
    theta[0, 0] = CP * CY
    theta[1, 0] = CP * SY
    theta[2, 0] = SP; 

    # left direction
    theta[0, 1] = CY * SP * SR - CR * SY
    theta[1, 1] = SY * SP * SR + CR * CY
    theta[2, 1] = -CP * SR; 

    # up direction
    theta[0, 2] = -CR * CY * SP - SR * SY
    theta[1, 2] = -CR * SY * SP + SR * CY
    theta[2, 2] = CP * CR

    return theta

def rotation_to_quaternion(m):
    trace = np.trace(m)

    q = [0,0,0,0]

    if (trace > 0.0):
        s = np.sqrt(trace + 1.0)
        q[0] = s * 0.5
        s = 0.5/ s
        q[1] = (m[2, 1] - m[1, 2]) * s
        q[2] = (m[0, 2] - m[2, 0]) * s
        q[3] = (m[1, 0] - m[0, 1]) * s

    else:
        if (m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            invS = 0.5 / s
            q[1] = 0.5 * s
            q[2] = (m[1, 0] + m[0, 1]) * invS
            q[3] = (m[2, 0] + m[0, 2]) * invS
            q[0] = (m[2, 1] - m[1, 2]) * invS

        elif (m[1, 1] > m[2, 2]):
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            invS = 0.5 / s
            q[1] = (m[0, 1] + m[1, 0]) * invS
            q[2] = 0.5 * s
            q[3] = (m[1, 2] + m[2, 1]) * invS
            q[0] = (m[0, 2] - m[2, 0]) * invS

        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            invS = 0.5 / s
            q[1] = (m[0, 2] + m[2, 0]) * invS
            q[2] = (m[1, 2] + m[2, 1]) * invS
            q[3] = 0.5 * s
            q[0] = (m[1, 0] - m[0, 1]) * invS



    return q

def euler_to_rotation_to_quaternion(state: State()):
    r = -1*state.euler.roll #rotation around roll axis to get car to world frame
    p = -1*state.euler.pitch #rotation around pitch axis to get car to world frame
    y = state.euler.yaw #rotation about the world z axis to get the car to the world frame
    Rx = np.matrix([[1, 0, 0], [0, np.cos(r), -1*np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.matrix([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-1*np.sin(p), 0, np.cos(p)]])
    Rz = np.matrix([[np.cos(y), -1*np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    #Order of rotations from car to world is z then y then x
    Rinter = np.matmul(Rz, Ry)
    Rcar_to_world = np.matmul(Rinter, Rx)

    return Rcar_to_world, Quaternion(matrix=Rcar_to_world)