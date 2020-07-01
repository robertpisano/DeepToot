from rlbot.utils.game_state_util import BallState, CarState
import tkinter as tk
from tkinter import filedialog
import glob
import carball
import os
import gzip
import pickle
import pandas as pd
from pandas import DataFrame

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


# Class that takes care of converting .replay to .json to .csv files for storing
#

class ReplayTransformer():

    def __init__(self, strategy):
        None

# Go through all the .replay files, and convert to CSV
#
# Save CSV files in chosen folder
    def convert_and_save_replays(self):
        from DeepToot.src.data_generation.hitFinder import HitFinder, RawReplayData
        root = tk.Tk()
        root.withdraw()
        # TODO: this and the replay generator should both use the same static directory by default so that we dont need to choose every time
        # we might be able to use relevant paths for this.
        print("Please select the folder where the replay data is found")
        self.file_path = filedialog.askdirectory() # open directory dialog
        self.replays_path = []
        # TODO: this logic can be its own private method
        for r in os.listdir(self.file_path):
            if r.endswith(".replay"):
                self.replays_path.append(self.file_path + '/' + r)
        # TODO: Would be better to make this a static directory so that we don't need to choose every time
        print("Please select the folder to save the converted data to")
        self.output_path = filedialog.askdirectory() # open output directory dialog
        self.game = Game()
        # TODO: make private method?
        for index, replay_file in enumerate(self.replays_path):
            json_file = carball.decompile_replay(replay_file,
                                output_path = self.file_path + str(index) + ".json",
                                overwrite=True)
            self.game.initialize(loaded_json = json_file) # Initialize game with each file path
            analysis_manager = AnalysisManager(self.game) # Initialize analysis_manager
            analysis_manager.create_analysis() # Analze replay

            # TODO: we should use a strategy pattern to inject the data parsing strategy. 
            #     It also should happen outside of the conversion from replay -> json data
            # Get hit raw data
            hit_finder = HitFinder()
            hit_finder.load_from_am(analysis_manager)
            hit_finder.get_hits()
            hit_frames = hit_finder.hit_frames # List of frame numbers that hits happened at

            # Get controls data
            controls_data = self.get_controls_from_replay(self.game)


            raw_replay = analysis_manager.get_data_frame() # Raw replay data frame

            # Initialize RawReplayData class with data from above
            raw_replay_data = RawReplayData()
            raw_replay_data.setData(raw_replay, controls_data, hit_frames)

            # Save raw replay data
            raw_replay_data.save_at(self.output_path + '/' + str(index))

            # data.to_csv(self.output_path + '/' + str(i) + '.csv')
            # self.gameDataList.append(data) # append pandas data to list
            # self.gameList.append(self.game)

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

    @ staticmethod
    def append_control_data(data: DataFrame, cc: ControlsCreator):
        for i, p in enumerate(cc.players):
            name = p.name # Name of player for this data
