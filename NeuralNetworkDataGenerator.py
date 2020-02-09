# This py file holds all the functions which will retrieve relavent data
# by request

import os
import subprocess
import pandas as pd
from pandas import DataFrame
import tkinter as tk
from tkinter import filedialog
import carball
from carball.analysis.events.hit_detection.base_hit import BaseHit
from carball.analysis.analysis_manager import AnalysisManager
# from ....generated.api.stats.events_pb2 import Hit
from typing import Dict

# Frame of data for neural network, is flattened and put in the proper order
#
# Player 1 data, Player 2 data, Ball data
#
# Player data: x, y, z, vx, vy, vz, roll, pitch, yaw, omegax, omegay, omegaz, throttle, steer,
# handbrake, jump_active, boost, boost_active, double_jump_active, dodge_active
#
# Ball data: x, y, z, vx, vy, vz, roll, pitch, yaw, omegax, omegay, omegaz
#
class PlayerFrame():
    def __init__(self, d: DataFrame):
        self.x = d.loc['pos_x']
        self.y = d.loc['pos_y']
        self.z = d.loc['pos_z']
        self.vx = d.loc['vel_x']
        self.vy = d.loc['vel_y']
        self.vz = d.loc['vel_z']
        self.rx = d.loc['rot_x']
        self.ry = d.loc['rot_y']
        self.rz = d.loc['rot_z']
        self.wx = d.loc['ang_vel_x']
        self.wy = d.loc['ang_vel_y']
        self.wz = d.loc['ang_vel_z']
        self.throttle = d.loc['throttle']
        self.steer = d.loc['steer']
        self.handbrake = d.loc['handbrake']
        self.double_jump_active = d.loc['double_jump_active']
        self.dodge_active = d.loc['dodge_active']
        self.jump_active = d.loc['jump_active']
        self.boost = d.loc['boost']
        self.boost_active = d.loc['boost_active']
    def to_numpy(self):
        return
        
class BallFrame():
    def __init__(self, d: DataFrame):
        self.x = d.loc['pos_x']
        self.y = d.loc['pos_y']
        self.z = d.loc['pos_z']
        self.vx = d.loc['vel_x']
        self.vy = d.loc['vel_y']
        self.vz = d.loc['vel_z']
        self.rx = d.loc['rot_x']
        self.ry = d.loc['rot_y']
        self.rz = d.loc['rot_z']
        self.wx = d.loc['ang_vel_x']
        self.wy = d.loc['ang_vel_y']
        self.wz = d.loc['ang_vel_z']
        self.hit_team_no = d.loc['hit_team_no']
    def to_numpy(self):
        return

class GameFrame():
    def __init__(self, p1, p2, ball):
        self.p1 = p1
        self.p2 = p2
        self.ball = ball
    def to_numpy(self):
        return

class FrameWindow():
    def __init__(self):
        self.window = []
    def append_game_frame(self, gf: GameFrame):
        self.window.append(gf)
    def to_numpy(self):
        # Convert data to properly structured shape
        return

class TrainingBatch():
    def __init__(self):
        self.batch = []
    def append_window(self, fw: FrameWindow):
        self.batch.append(fw)
    def to_numpy(self):
        # Convert data to properly structured shape for neural network training
        return


# Generate data from when ball was hit and in/outWindow
def generate_nn_data_from_hits(am: AnalysisManager, hits: Dict, inWindow: int, outWindow: int):
    hit_frames = list(hits.keys())
    object_keys = am.data_frame.columns.levels[0]
    data_keys = am.data_frame.columns.levels[1]
    batch = TrainingBatch()

    for h in hit_frames:
        fw1 = FrameWindow() # Refresh current frame window for appending to training batch
        for i in range(h-inWindow, h+outWindow):
            # Pull gameobject data from analysis manager object
            game_data_at_frame = am.data_frame.loc[i]
            p1 = PlayerFrame(game_data_at_frame.loc[object_keys[0]])
            p2 = PlayerFrame(game_data_at_frame.loc[object_keys[1]])
            ball = BallFrame(game_data_at_frame.loc[object_keys[2]])
            # Create Game frame object
            g1 = GameFrame(p1, p2, ball)
            # Append game frame to frame window
            fw1.append_game_frame(g1)
        # Append frame window to training batch
        batch.append_window(fw1)
    return batch
