# This file has some classes that will make running a replay scenario simple
# InitialGameState will be the initial state of the game to start the scenario
# PlayerControls will be the calculated inputs for that specific scenario subset
# These together will give an RLBot agent the ability to test the inputs then
# compare the raw replay data to the new generated data to compare and see how precise the input calulation is

from NeuralNetworkDataGenerator import PlayerState, BallState
from hitFinder import RawReplayData
from pandas import DataFrame
import threading
import time
import numpy as np

# Plotting Library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# RLBOT library stuff
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState as rlBallState


# from RLBOT.src.util.orientation import Orientation
# from util.vec import Vec3

# Initial game state 
class InitialGameState():

    def __init__(self, p1: PlayerState, p2: PlayerState, ballstate: BallState):
        self.orangeCar = p1
        self.blueCar = p2
        self.ball = ballstate
    
    def get_ball_data(self):
        pos = [self.ball.frameData['pos_x'], self.ball.frameData['pos_y'], self.ball.frameData['pos_z']]
        vel = [self.ball.frameData['vel_x'], self.ball.frameData['vel_y'], self.ball.frameData['vel_z']]
        return np.asarray(pos), np.asarray(vel)
    
    # TODO: Add other player parameters for return here (rotation, ang_vel)
    def get_orange_data(self):
        pos = [self.orangeCar.frameData['pos_x'], self.orangeCar.frameData['pos_y'], self.orangeCar.frameData['pos_z']]
        vel = [self.orangeCar.frameData['vel_x'], self.orangeCar.frameData['vel_y'], self.orangeCar.frameData['vel_z']]
        rot = [self.orangeCar.frameData['rot_x'], self.orangeCar.frameData['rot_z'], self.orangeCar.frameData['rot_z'],]
        angvel = [self.orangeCar.frameData['ang_vel_x'], self.orangeCar.frameData['ang_vel_z'], self.orangeCar.frameData['ang_vel_z'],]
        return np.asarray(pos), np.asarray(vel), np.asarray(rot), np.asarray(angvel)

    def get_blue_data(self):
        pos = [self.blueCar.frameData['pos_x'], self.blueCar.frameData['pos_y'], self.blueCar.frameData['pos_z']]
        vel = [self.blueCar.frameData['vel_x'], self.blueCar.frameData['vel_y'], self.blueCar.frameData['vel_z']]
        rot = [self.blueCar.frameData['rot_x'], self.blueCar.frameData['rot_z'], self.blueCar.frameData['rot_z'],]
        angvel = [self.blueCar.frameData['ang_vel_x'], self.blueCar.frameData['ang_vel_z'], self.blueCar.frameData['ang_vel_z'],]
        return np.asarray(pos), np.asarray(vel), np.asarray(rot), np.asarray(angvel)

# All player controls
class PlayerControls():
    def __init__(self, players, frameStart, scenarioLength):
        self.frameStart = frameStart
        self.scenarioLength = scenarioLength
        if(players[0].is_orange):
            self.orangeControls = players[0].controls.loc[frameStart:frameStart+scenarioLength]
            self.blueControls = players[1].controls.loc[frameStart:frameStart+scenarioLength]
        else:
            self.orangeControls = players[1].controls.loc[frameStart:frameStart+scenarioLength]
            self.blueControls = players[0].controls.loc[frameStart:frameStart+scenarioLength]

    def get_blue_controls(self, counter): # Return the controls in the SimpleControllerState() form
        return self.get_controller_state(self.blueControls, counter)

    def get_orange_controls(self, counter):
        return self.get_controller_state(self.orangeControls, counter)

    def get_controller_state(self, controls, counter): # take player controls data and parse into the SimpleControllerState() class and return
        con = SimpleControllerState()
        controls_fixed = self.fix_controls(controls) # Fix NAN, etc..
        con.throttle = controls_fixed.loc[self.frameStart+counter].throttle
        con.steer = controls_fixed.loc[self.frameStart+counter].steer
        con.pitch = controls_fixed.loc[self.frameStart+counter].pitch
        con.yaw = controls_fixed.loc[self.frameStart+counter].yaw
        con.roll = controls_fixed.loc[self.frameStart+counter].roll
        con.jump = controls_fixed.loc[self.frameStart+counter].jump
        con.boost = controls_fixed.loc[self.frameStart+counter].boost
        con.handbrake = controls_fixed.loc[self.frameStart+counter].handbrake
        con.use_item = False
        return con

    def fix_controls(self, controls): #TODO: Takes control data, fixes NAN etc...
        controls.replace(to_replace=[None, np.nan], value=0.0, inplace=True)
        return controls

# DataComparison will convert an array of packets into a dataframe similar to the replay dataframe
# That way we can easily plot positions, velocities, etc... to analyze and see where
# the calculated controls went wrong
class DataComparison():
    replay_data_frame = []
    live_data_frame = []
    def __init__(self):
        None
    
    # Convert array of packets into properly sized dataframe for analysis
    @staticmethod
    def convert_to_data_frame(packet_array):
        None

    # Plot the 3axis positions and 3axis velocities
    # Plot 1 3d plot of positions
    # Plot 3x1 subplot for each position axis
    # Plot 3x1 subplot for each velocity axis
    def plot_linear(self):
        None

    # Plot 3x1 subplot for each axis orientation (maybe 1 3d plot with orientation as a line)
    # Plot 3x1 subplot fo each axis ang_velocity
    def plot_rotational(self):
        None
    


class ScenarioCreator():
    start = False
    running = False
    counter = 0
    counter_length = 600
    packets = []

    def __init__(self):
        self.event = threading.Event()
        self.starttime = time.time()
        self.thread = threading.Thread(target=self.get_user_input)
        self.thread.start()
        self.counter = 0
        self.start = True
        

    # Get initial state, and get controls for each player, return for use with RLBot
    def get_scenario_data(self, rrd: RawReplayData, frameStart: int, scenarioLength: int):
        self.frameStart = frameStart
        self.scenarioLength = scenarioLength
        object_keys = rrd.game.columns.levels[0] # Get the objects in game
        data_keys = rrd.game.columns.levels[1] # Get the data keys from game
        game_data_at_frame = rrd.game.loc[frameStart] # get the dataframe at frameStart
        p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
        p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
        ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
        # Get the initial game state to set in RLBot
        self.initialState = InitialGameState(p1, p2, ball)
        # Get control data frame subset, send to PlayerControls class, save in self.
        self.playerControls = PlayerControls(rrd.players, frameStart, scenarioLength)
        # Get the rest of the scenario data for the same subset for comparison etc...
        self.scenarioData = rrd.game.loc[frameStart:frameStart+scenarioLength]
        
    # Load hardcoded replay data into self
    def hardcoded_load(self):
        self.rrd = RawReplayData.load()
        self.get_scenario_data(self.rrd, 3300, 1700)
    
    # Control the rl environment
    def control_environment(self, bot):
        try:
            
            if self.start == False and self.running == True: #Iterate through controls
                controller = self.playerControls.get_controller_state(self.playerControls.blueControls, self.counter)
                bot.controller_state = controller
                if(self.counter > self.counter_length): self.event.set()
                self.counter = self.counter + 1
                # Append bot packet to self.packets
                self.packets.append(bot.packet)
                
            if self.start == True and self.running == False: # Set initial state
                bpos, bvel = self.initialState.get_ball_data()
                cpos, cvel, crot, cangvel = self.initialState.get_blue_data()
                b = rlBallState(Physics(location=Vector3(bpos[0],bpos[1],bpos[2]), velocity=Vector3(bvel[0],bvel[1],bvel[2])))
                c = CarState(Physics(location=Vector3(cpos[0], cpos[1], cpos[2]), velocity=Vector3(cvel[0], cvel[1],cvel[2]), rotation=Rotator(roll=crot[0], pitch=crot[1], yaw=crot[2]), angular_velocity=Vector3(x=cangvel[0], y=cangvel[1], z=cangvel[2])))
                bot.set_game_state(GameState(ball = b, cars={bot.index: c}))
                self.start = False
                self.running = True
                print('location: ', b.physics.location.x, b.physics.location.y, b.physics.location.z)
                print('velocity: ', b.physics.velocity.x, b.physics.velocity.y, b.physics.velocity.z)
                controller = self.playerControls.get_controller_state(self.playerControls.blueControls, self.counter)
                bot.controller_state = controller
                self.counter = 0
                # Set self.packets to the initial packet
                self.packets = [bot.packet]


        except Exception as e:
            e.with_traceback()


    def get_user_input(self): # Supposed to be CLI, but rn is just running on a thread hardcoded to change every "length" amount of seconds
        # length = 10.0
        while(True):
            try:
                # time.sleep(time.time() % length)
                self.event.wait()

                # Plot replay data compared to collected data
                self.plot_comparison()

                self.start = not self.start
                if self.start == True:
                    self.running = False
                self.event.clear()
                # if self.start == True:
                #     self.running = False
                #     self.counter = 0
                
                # print(self.start)
            except Exception as e:
                print(e)
                continue
    
    def initialize_plot(self):
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1,1,1,projection='3d')
        self.ax.set_xlim3d(-3000, 3000)
        self.ax.set_ylim3d(-3000, 3000)
        self.ax.set_zlim3d(0, 2000)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.invert_yaxis()
        self.ax.set_ylabel('Position y')
        self.ax.set_xlabel('Position x')
        self.ax.set_zlabel('time')

    def update_plot(self):
        # Plot Data 

        # t = np.linspace(0,self.scenarioLength,self.scenarioLength)# Time vector for plotting
        x = self.scenarioData
        
        Axes3D.plot(self.ax, self.sim_results.sx, self.sim_results.sy, ts, c='r', marker ='o')


if __name__ == '__main__':
    rrd = RawReplayData.load()
    print(rrd)
    sc = ScenarioCreator()
    sc.get_scenario_data(rrd, 100, 60)
    fixed = sc.playerControls.fix_controls(sc.playerControls.blueControls)
    print('end')