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

from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState as rlBallState

from util.orientation import Orientation
from util.vec import Vec3

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
    
    # TODO: Add other player parameters for return here
    def get_blue_data(self):
        pos = [self.blueCar.frameData['pos_x'], self.blueCar.frameData['pos_y'], self.blueCar.frameData['pos_z']]
        vel = [self.blueCar.frameData['vel_x'], self.blueCar.frameData['vel_y'], self.blueCar.frameData['vel_z']]
        return np.asarray(pos), np.asarray(vel)

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

class ScenarioCreator():
    start = False
    running = False

    def __init__(self):
        self.starttime = time.time()
        self.thread = threading.Thread(target=self.get_user_input)
        self.thread.start()
        

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
        self.initalState = InitialGameState(p1, p2, ball)
        # Get control data frame subset, send to PlayerControls class, save in self.
        self.playerControls = PlayerControls(rrd.players, frameStart, scenarioLength)
        # Get the rest of the scenario data for the same subset for comparison etc...
        self.scenarioData = rrd.game.loc[frameStart:frameStart+scenarioLength]

    # Load hardcoded replay data into self
    def hardcoded_load(self):
        self.rrd = RawReplayData.load()
        self.get_scenario_data(self.rrd, 2000, 60)
    
    # Control the rl environment
    def control_environment(self, bot):
        if self.start == False and self.running == True: # Iterate through contorls
            None
        if self.start == True and self.running == False: # Set initial state
            bpos, bvel = self.initalState.get_ball_data()
            b = rlBallState(Physics(location=Vector3(bpos[0],bpos[1],bpos[2]), velocity=Vector3(bvel[0],bvel[1],bvel[2])))
            bot.set_game_state(GameState(ball = b))
            self.start = False
            # self.running = True
            print('location: ', b.physics.location.x, b.physics.location.y, b.physics.location.z)
            print('velocity: ', b.physics.velocity.x, b.physics.velocity.y, b.physics.velocity.z)
    
        


    def get_user_input(self):
        length = 5.0
        while(True):
            time.sleep(time.time() % length)
            self.start = not self.start
            # print(self.start)

if __name__ == '__main__':
    rrd = RawReplayData.load()
    print(rrd)
    sc = ScenarioCreator()
    sc.get_scenario_data(rrd, 100, 60)
    print('end')