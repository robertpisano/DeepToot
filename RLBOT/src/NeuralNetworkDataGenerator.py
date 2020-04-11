# This py file holds all the functions which will retrieve relavent data
# by request

import os
import subprocess
import pandas as pd
from pandas import DataFrame, Series
import tkinter as tk
from tkinter import filedialog
import carball
from carball.analysis.events.hit_detection.base_hit import BaseHit
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator
# from ....generated.api.stats.events_pb2 import Hit
from typing import Dict
import numpy as np

from rlbot.utils.structures.game_data_struct import GameTickPacket

import copy

# Frame of data for neural network, is flattened and put in the proper order
#
# Player 1 data, Player 2 data, Ball data
#
# Player data: x, y, z, vx, vy, vz, roll, pitch, yaw, omegax, omegay, omegaz, throttle, steer,
# handbrake, jump_active, boost, boost_active, double_jump_active, dodge_active
#
# Ball data: x, y, z, vx, vy, vz, roll, pitch, yaw, omegax, omegay, omegaz
#

class BaseState():
    def __init__(self, d: Series):
        self.frameData = dict()
        for key, value in d.items():
            if key not in self.excluded_labels:
                self.frameData[key] = value
        self.fix_nulls()
        self.fix_true_false()

    def fix_nulls(self):
        for key, item in self.frameData.items():
            if(item == None or np.isnan(item)):
                self.frameData[key] = 0.0
            else:
                continue
    def fix_true_false(self):
        for key, item in self.frameData.items():
            if(isinstance(item, bool)):
                if(item == True):
                    self.frameData[key] = 1.0
                if(item == False):
                    self.frameData[key] = 0.0

    def to_numpy(self):
        return np.array(list(self.frameData.values()), dtype=np.float)

class PlayerState(BaseState):
    excluded_labels = ['ball_cam', 'boost_collect']


class BallState(BaseState):
    excluded_labels = ['rx', 'ry', 'rz']


class GameFrame():
    def __init__(self, p1, p2, ball):
        self.p1 = p1
        self.p2 = p2
        self.ball = ball
    def to_numpy(self):
        # Flatten data into 1D and return
        # TODO: USE NP.FLATTEN U DUMBASS
        return np.append(np.append(self.p1.to_numpy(), self.p2.to_numpy()), self.ball.to_numpy())


class FrameWindow():
    def __init__(self):
        self.frames = []
    def append_game_frame(self, gf: GameFrame):
        self.frames.append(gf)
    def to_numpy(self):
        for i, w in enumerate(self.frames):
            if i == 0:
                temp = np.expand_dims(w.to_numpy(), axis=1)
            else:
                temp = np.append(temp, np.expand_dims(w.to_numpy(), axis=1), axis=1)
        return temp

class TrainingBatch():
    def __init__(self):
        self.batch = []
    def append_window(self, fw: FrameWindow):
        self.batch.append(fw)
    def to_numpy(self):
        for i, b in enumerate(self.batch):
            if i == 0:
                temp = np.expand_dims(b.to_numpy(), axis = 2)
                # temp = b.to_numpy()
            else:
                print('i: ' + str(i))
                temp = np.append(temp, np.expand_dims(b.to_numpy(), axis=2), axis=2)
        # Convert data to properly structured shape for neural network training
        # Axis 0: batch axis
        # Axis 1: Frame window axis
        # Axis 2: Flattened state p1, p2, ball
        tempfix = np.swapaxes(temp, 0, 2) # Data comes in with axis 0 and 2 swapped, so swap them to proper
        return tempfix

class GameMemory():
    def __init__(self, window_length):
        self.short_term_memory_length = window_length
        self.memory_queue = []
    
    def _short_term_memory_start(self):
        return (self.memory_queue.length - self.memory_length)

    def append_memory(self, g: GameTickPacket):
        self.memory_queue.append(g)
    
    def get_short_term_memory(self):
        return self.memory_queue[self._short_term_memory_start():]
        
    def get_memory(self):
        return self.memory_queue

class NeuralNetworkManager():
    WINDOW_LENGTH = 10

    def __init__(self):
        self.memory = GameMemory(self.WINDOW_LENGTH)
    
    def get_control_update(self, g: GameTickPacket):
        self.memory.append_memory(g)
        # Calculate output from nn

# This class will interface with bot.py to start recording data of myself playing the game
# To do this bot.py will have to be in the game, then a local player will also be in the game
# Then through some method (maybe ask for input?) you will start to record data for some amount of time
# Each GameTickPacket will be placed in an array then pickled to a file at the end of the generation 
# This class should also "clean" the data to get only the player we are about's data
class DataGeneratorManager():
    def __init__(self, length: int):
        import threading, datetime
        self.frames = [] # The array of gametickpackets
        self.started = False # The class is ready to take append frame data
        self.finished = False
        self.length = length # The length of the raw data (in frames) that we want to save
        self.start_time = datetime.datetime.now()
        self.export_thread = threading.Thread(target=self.export)
        self.thread = threading.Thread(target=self.start) # Thread that will
        self.thread.start()

    def start(self): # Start the recording of data
        import datetime
        now = datetime.datetime.now() 
        while((now - self.start_time).seconds < 3): # 3 second Timer to give me some time to get into game with controller
            now = datetime.datetime.now()
            print(now) # Debug
        self.started = True # Set state to started
        return True
    
    def append(self, packet): # Append packet to array of packets
        self.frames.append(copy.deepcopy(packet)) # Need to deepcopy since i think packet is a pointer, and when i go to dump at the end it dumps the current packet over and over again, so it must be a pointer
        self.check_complete() # Check if we should do the complete action

    def check_complete(self): # check to see if we've recorded enough data and trigger export
        if(len(self.frames) >= self.length): # If frames >= length then we set the state to finished and export the data
            
            self.finished = True
            self.export_thread.start()
        else:
            print(len(self.frames))
            print(self.frames[len(self.frames) - 1].game_cars[1].physics.location)
            return

    def export(self):
        print(len(self.frames)) # Debug
        import dill, os
        dir_path = os.path.dirname(os.path.realpath(__file__)) # Get local file path
        output_path = os.path.join(dir_path, "generated_data") # Add directory generated_data
        file_path = os.path.join(output_path, "generated_data.p") # Add filename to path
        dill.dump(self.frames, open(file_path, 'wb')) # Write frames to file as pickle
        print('finished pickling') # Debug

# Class that will load packet data that was saved from the DataGeneratorManager class
# Will plot the data for analysis, choose which player to analyze
class DataAnalyzer():

    def __init__(self):
        self.frames = []
        self.load_generated_data()
        self.initialize_plots()


    # Initialize axes and figures to plot on
    def initialize_plots(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1,1,1,projection = '3d')

    # Get saved data from known location
    def load_generated_data(self):
        import dill, os
        dir_path = os.path.dirname(os.path.realpath(__file__)) # Get local file path
        output_path = os.path.join(dir_path, "generated_data") # Add directory generated_data
        file_path = os.path.join(output_path, "generated_data.p") # Add filename to path
        self.frames = dill.load(open(file_path, 'rb')) # Write frames to file as pickle
    
    # Do a "full analysis" on the data
    def full_analysis(self):
        # Parse through all game cars and give names at each index
        cars = self.frames[0].game_cars
        for i, car in enumerate(cars):
            print(str(i) + ": " + str(car.name))
        
        while(True): # Wait for user to put in correct input
            try:
                index = int(input('Input index of car to plot data for: '))
                break
            except Exception as e:
                print(e)
                continue

        self.full_plot(index)
    
    # Plot the positions on a 3d axis
    def full_plot(self, index):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        # Extract position data
        x = []
        y = []
        z = []
        for i, d in enumerate(self.frames):
            print(self.frames[i].game_cars[index].physics.location)
            x.append(self.frames[i].game_cars[index].physics.location.x)
            y.append(self.frames[i].game_cars[index].physics.location.y)
            z.append(self.frames[i].game_cars[index].physics.location.z)

        Axes3D.plot(self.ax, x, y, z, c='r', marker='o')
        self.ax.set_xlim3d(-3000, 3000)
        self.ax.set_ylim3d(-3000, 3000)
        self.ax.set_zlim3d(0, 2000)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # self.ax.invert_yaxis()
        # self.ax.set_ylabel('Position y')
        # self.ax.set_xlabel('Position x')
        # self.ax.set_zlabel('Position z')
        plt.show(block=False) # Show plot
        import code
        code.interact(local=locals())

# Generate data from when ball was hit and in/outWindow
def generate_nn_data_from_hits(am: AnalysisManager, hits: Dict, inWindow: int, outWindow: int):
    hit_frames = list(hits.keys())
    
    object_keys = am.data_frame.columns.levels[0]
    data_keys = am.data_frame.columns.levels[1]
    inBatch = TrainingBatch()
    outBatch = TrainingBatch()

    for h in hit_frames:
        fw1 = FrameWindow() # Refresh current frame window for appending to inBatch
        fw2 = FrameWindow() # Refresh current frame window for appending to outBatch
        for i in range(h-inWindow, h):
            # Pull gameobject data from analysis manager object
            game_data_at_frame = am.data_frame.loc[i]
            # goal_position = GoalPosition() # Own Goal position (0/1)
            p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
            p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
            ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
            # Create Game frame object
            g1 = GameFrame(p1, p2, ball)
            # Append game frame to frame window
            fw1.append_game_frame(g1)

        for j in range(h+1, h+outWindow+1): # make output window data
            game_data_at_frame = am.data_frame.loc[j]
            # goal_position = GoalPosition() # Own Goal position (0/1)
            p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
            p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
            ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
            # Create Game frame object
            g2 = GameFrame(p1, p2, ball)
            # Append game frame to frame window
            fw2.append_game_frame(g2)
        # Append frame window to training input batch
        inBatch.append_window(fw1)
        # Append frame window to training output batch
        outBatch.append_window(fw2)
    
    
    return inBatch, outBatch


def generate_inputs_from_first_hit(am: AnalysisManager, hits: Dict, inWindow: int, outWindow: int):
    hit_frames = list(hits.keys())
    
    object_keys = am.data_frame.columns.levels[0]
    data_keys = am.data_frame.columns.levels[1]
    inBatch = TrainingBatch()
    outBatch = TrainingBatch()

    # Force h as first hit
    h = hit_frames[0]

    fw1 = FrameWindow() # Refresh current frame window for appending to inBatch
    fw2 = FrameWindow() # Refresh current frame window for appending to outBatch
    for i in range(h-inWindow, h):
        # Pull gameobject data from analysis manager object
        game_data_at_frame = am.data_frame.loc[i]
        # goal_position = GoalPosition() # Own Goal position (0/1)
        p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
        p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
        ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
        # Create Game frame object
        g1 = GameFrame(p1, p2, ball)
        # Append game frame to frame window
        fw1.append_game_frame(g1)

    for j in range(h+1, h+outWindow+1): # make output window data
        game_data_at_frame = am.data_frame.loc[j]
        # goal_position = GoalPosition() # Own Goal position (0/1)
        p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
        p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
        ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
        # Create Game frame object
        g2 = GameFrame(p1, p2, ball)
        # Append game frame to frame window
        fw2.append_game_frame(g2)
    # Append frame window to training input batch
    inBatch.append_window(fw1)
    # Append frame window to training output batch
    outBatch.append_window(fw2)

def generate_nn_data_from_saved_data_frames(inWindow: int, outWindow: int):
    from RLBOT.src.hitFinder import RawReplayData
    file_path = filedialog.askdirectory() #Ask for directory of pickled files
    replays_path = []
    for r in os.listdir(file_path):
        if r.endswith(".p"):
            replays_path.append(file_path + '/' + r)

    inBatch = TrainingBatch()
    outBatch = TrainingBatch()
    for r in replays_path:
        # From replay file, initialize RawReplayData
        raw_replay_data = RawReplayData().load_from(r)

        # Data frame and Hit framess
        # print(raw_replay_data.hits)
        # type(raw_replay_data.hits)
        hit_frames = list(raw_replay_data.hits)
        data = raw_replay_data.game
        
        object_keys = data.columns.levels[0]
        data_keys = data.columns.levels[1]


        for hitnum, h in enumerate(hit_frames):
            try:
                fw1 = FrameWindow() # Refresh current frame window for appending to inBatch
                fw2 = FrameWindow() # Refresh current frame window for appending to outBatch
                for i in range(h-inWindow, h):
                    # Pull gameobject data from analysis manager object
                    game_data_at_frame = data.loc[i]
                    # goal_position = GoalPosition() # Own Goal position (0/1)
                    p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
                    p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
                    #TODO: Make sure to check if object_keys[2] is the ball, also make sure not downloading 2v2 games
                    ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
                    # Create Game frame object
                    g1 = GameFrame(p1, p2, ball)

                    # Append game frame to frame window
                    fw1.append_game_frame(g1)

                for j in range(h+1, h+outWindow+1): # make output window data
                    game_data_at_frame = data.loc[j]
                    # goal_position = GoalPosition() # Own Goal position (0/1)
                    p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
                    p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
                    ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
                    # Create Game frame object
                    g2 = GameFrame(p1, p2, ball)
                    # Append game frame to frame window
                    fw2.append_game_frame(g2)
            except Exception as e:
                print(e)
            # Append frame window to training input batch
            inBatch.append_window(fw1)
            # Append frame window to training output batch
            outBatch.append_window(fw2)
            
        
        
    return inBatch, outBatch