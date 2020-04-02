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
            # Append frame window to training input batch
            inBatch.append_window(fw1)
            # Append frame window to training output batch
            outBatch.append_window(fw2)

            #Debugging purposes
            print(inBatch)
            
        
        
    return inBatch, outBatch