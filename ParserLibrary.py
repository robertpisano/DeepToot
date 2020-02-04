from rlbot.utils.game_state_util import BallState, CarState
import tkinter as tk
from tkinter import filedialog
import glob
import carball
import os
import gzip
import pickle

from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# This py file contains all the classes dealing with parsing replays into usable pandas data formats (numpy formatsas well),
# also contains a structure of classes to help analze the replay data. Ultimately this is used to create IN:OUT data for the training of a neural Network
#
# Use metrics to find situations that happened in replay that came out positive. Those trajectories are good. In this case we'll consider the
# ball not going into the goal good and going into the other guys goal good as well
#
# The input to the network is the initial states of throughout that "positive" trajectory, each frame along the trajectory is a different set of in/out data
# The output data is a +/- metric. How strong in each direction is a metric to be considered. I think angle away from your own net / into enemy net should define
# how far along 0-1 the "strength" of that trajectory was.



# A class which will have relevant calculations for the ball relative to each player(s)
class BallCalculator():
    def __init__(self, ball: BallState):
        self.b = ball
        self.ball_goal0 = self.calc_goal_vectors()[0,:]
        self.ball_goal1 = self.calc_goal_vectors()[1,:]

# raw goal is a function that evaluates the state of the ball only and the "score"  of the ball relative to each player
#
# The function can take many forms but to keep it simple currently f<> = -1 * (a * (1/r) + b * dot(v, ball_towards_own_goal_vector))
#
# a and b are tunable parameters
#
# r is the ball's proximity to each goal, put in the denominator since closer is considered worse
#
# dot(v, ball_towards_own_goal_vector) is the dot product between the balls velocity and the vector from the ball to the center of the goal
# as this value increases that means the ball's velocity is not only faster towards your goal, but also pointing closer to the center of the goal
#
# note the -1, this is just defining the ball into your own goal has negative value
#
    def raw_goal_chance(self):
        return np.array([0, 0])

# Calculate the vector that points from the ball towards the center of each goal [goal0, goal1]
    def calc_goal_vectors(self):
        return None

class CarCalculator():
    def __init__(self, car: CarState):
        self.c = car

# Parser allows you to parse saved replay data, or multiple replays depending on certain metrics to pull frames in the game that you want
#
#
#
#
#

class Parser():
    def __init__(self):
        replay_directory = None

# Open a dialog that lets you choose a directory where all the replays are saved
#
# Store in replay_directory
    def get_replay_directory(self):
        None


# The entire instance of a replay put into padas to allow for modular data searching,
# includes functional input data that can be parsed through to control a car in rocket league
class TotalData():
    None

# Load ReplayConverter from pickled save file
def factory():
    try:
        rc = ReplayConverter.load_state()
        rc.saveStateExists = True
        print('Save state found and loaded')
        return rc
    except Exception as e:
        print(e)
        rc = ReplayConverter()
        rc.saveStateExists = False
        print('No save state found for ReplayConverter')
        return rc


# Class that takes care of converting .replay to .json to .csv files for storing
#

class ReplayConverter():

    def __init__(self):
        None

# Go through all the .replay files, and convert to CSV
#
# Save CSV files in chosen folder
    def convert_and_save_replays(self):
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askdirectory() # open directory dialog
        self.replays_path = []
        for r in os.listdir(self.file_path):
            if r.endswith(".replay"):
                self.replays_path.append(self.file_path + '/' + r)
        self.output_path = filedialog.askdirectory() # open output directory dialog

        self.game = Game()
        for i, r in enumerate(self.replays_path):
            json_file = carball.decompile_replay(r,
                                output_path = self.file_path + str(i) + ".json",
                                overwrite=True)
            self.game.initialize(loaded_json = json_file) # Initialize game with each file path
            analysis_manager = AnalysisManager(self.game) # Initialize analysis_manager
            analysis_manager.create_analysis() # Analze replay
            data = analysis_manager.get_data_frame()
            data.to_csv(self.output_path + '/' + str(i) + '.csv')

    def save_current_state(self):
        pickle.dump(self, open('pickles/replayConverter.p', 'wb'))

    @ staticmethod
    def load_state():
        return pickle.load(open('pickles/replayConverter.p', 'rb'))

    @ staticmethod
    def get_controls_from_replay(g: Game):
        cc = ControlsCreator()
        ret = cc.get_controls(g)
        print('ret', ret)
        return cc
