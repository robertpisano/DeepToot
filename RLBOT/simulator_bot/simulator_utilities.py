# This py file will house classes for the GUI and optimizer/simulator class


from rlbot.utils.structures.game_data_struct import GameTickPacket
from tkinter import Tk, filedialog, Toplevel, Frame, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar, Checkbutton, Canvas, PhotoImage, NW
from util.vec import Vec3
from threading import Thread

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


    # initialize the buttons on the main frame
    def initialize_buttons(self):
        self.load_sim_state_button = Button(self.frame, text='load sim state', command=lambda:self.load_sim_state())
        self.load_sim_state_button.pack(fill='both', expand=True)

        self.save_sim_state_button = Button(self.frame, text='save sim state', command=lambda:self.save_sim_state())
        self.save_sim_state_button.pack(fill='both', expand=True)
        self.generate_sim_state_button = Button(self.frame, text='generate sim state', command=lambda:self.generate_sim_state())
        self.generate_sim_state_button.pack(fill='both', expand=True)
        self.show_sim_params_button = Button(self.frame, text='show sim params', command=lambda:self.show_sim_params())
        self.show_sim_params_button.pack(fill='both', expand=True)
    
    # intialize the sim params frame which  holds the initial conditions of the car and ball
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
        self.save_sim_param_button = Button(self.sim_params_frame, text='Save Sim Params', command=lambda:self.save_sim_params(self))
        self.save_sim_param_button.grid(row=row_counter, column = 0, columnspan=9, sticky='nsew')
        row_counter += 1
        self.load_sim_param_button = Button(self.sim_params_frame, text='load sim params', command=lambda:self.load_sim_params(self))
        self.load_sim_param_button.grid(row=row_counter, column=0, columnspan=9, sticky='nsew')
        row_counter += 1
        self.hide_sim_param_button = Button(self.sim_params_frame, text='Hide This Window', command = lambda:self.sim_params_frame.withdraw())
        self.hide_sim_param_button.grid(row=row_counter, column=0, columnspan=9, sticky='nsew')

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
        class Empty(): None
        cardata = Empty()
        balldata = Empty()
        return_vals = []
        return_val_strings=[]
        for i, e in enumerate(self.entry_names):
            sub = e.split('|')
            var_name = str(sub[1])
            try: # check if attribute exists
                if(sub[0] == 'car'):
                    getattr(cardata, var_name)
                    vars(cardata)[var_name].append(float(self.entrys[i].get()))
                if(sub[0] == 'ball'):
                    getattr(balldata, var_name)
                    vars(balldata)[var_name].append(float(self.entrys[i].get()))                    
            except: # if attribute doesn't exist, create it and also add it to list of return-vals
                if(sub[0] == 'car'):
                    vars(cardata)[var_name] = []
                    vars(cardata)[var_name].append(float(self.entrys[i].get()))
                    # return_vals.append(getattr(cardata, var_name))
                    # return_val_strings.append(var_name)
                if(sub[0] == 'ball'):
                    vars(balldata)[var_name] = []
                    vars(balldata)[var_name].append(float(self.entrys[i].get()))

        print('pause debug')
        return cardata, balldata
    
    def show_sim_params(self):
        self.sim_params_frame.deiconify()  

    # Will take the simulation parameters, and run them through the optimzier/simulator
    # Then will generate the full SimulationState that is given to the Agent() class 
    # which will allow the Agent to run the simulation autnomously without needing to interact with the GUI
    def generate_sim_state(self):
        print(self.sim_params)

    @staticmethod
    def load_sim_params(self):
        import dill
        root = Tk()
        save_path = filedialog.askopenfilename(parent=root) # Ask user for file to load from
        self.sim_params = dill.load(open(save_path, 'rb'))
        root.destroy()
        #TODO: Initialize self.sim_params_frame from the data loaded!

    @staticmethod
    def load_sim_state():
        None
    @staticmethod
    def save_sim_params(self):
        # Save the simulation parameters in the local variable
        try:
            self.export_sim_param_frame_data()
            import dill
            root = Tk()
            save_path = filedialog.askopenfilename(parent=root) # Ask user to file to save to
            dill.dump(self.sim_params, open(save_path, 'wb'))
            root.destroy()
        except:
            print("Try saving again")
    @staticmethod
    def save_sim_state():
        import dill
        None


# Has all the necessary data to run the AerialOptimizer functions, will be passed into AerialOptimizer.optimize
# or whatever the name of the function is
class SimulationParameters():
    def __init__(self):
        self.ball_state = None
        self.car_state = None
    
    def initialize_from_raw(self, car, ball):
        self.car_state = car
        self.ball_state = ball

# Simulation state will house all the data for a specific scenario, trajectory, initial states of ball/car, the type of controller to use
class SimulationState():
    None


# The full state of the car or ball
class State():
    def __init__(self):
        None

    def default_init(self, data, index):
        # Check if data is a GameTickPacket, initialize from packet data if so
        if isinstance(data, GameTickPacket):
            self.init_from_packet(data, index)
        else:
            self.position = []
            self.velocity = []
            self.orientation = []
            self.ang_vel = []
    
    def init_from_packet(self, packet, index):
        if index == 'ball':
            self.position = packet.game_ball.location
            self.velocity = packet.game_ball.velocity
            self.orientation = packet.game_ball.orientation
            self.ang_vel = packet.game_ball.angular_velocity
        else:
            self.position = packet.game_cars[index].physics.location
            self.velocity = packet.game_cars[index].physics.velocity
            self.orientation = packet.game_cars[index].physics.rotation # {Pitch, Yaw, Roll}
            self.ang_vel = packet.game_cars[index].physics.angular_velocity
            self.other_data = packet.game_cars[index]

    def init_from_raw_data(self, position: Vec3, velocity: Vec3, orientation: Vec3, angular_velocity: Vec3):
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.ang_vel = angular_velocity


# A class that has an array of states that make up the entire trajectory
# Time vector is also here
class Trajectory():
    # Model is the AerialOptimizer class that has been run.
    def init_from_optimization_model(self, model):
        self.time = model.ts
        self.stats = []
        

class Controller():
    None

