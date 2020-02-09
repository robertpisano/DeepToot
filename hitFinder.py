# Hit finder will open up a csv file, or loop through a list of csv files
# to find hits the frame number that the hit occured at
import os
import subprocess
import pandas as pd
from pandas import DataFrame
import tkinter as tk
from tkinter import filedialog
import carball
from carball.analysis.events.hit_detection.base_hit import BaseHit
# from ....generated.api.stats.events_pb2 import Hit
from typing import Dict
import NeuralNetworkDataGenerator as nndg

class HitFinder():
    def __init__(self):
        self.hits = [] # Dictionary that carball BaseHit.get_hits_from_game() returns
        self.hit_frames = [] # List of frame numbers that hits happened at to make accessing self.hits easier since it is an unordered dictionary
    def load_replay(self):
        # Tkinter frame for windows explorer dialog to pop up
        root = tk.Tk()
        root.withdraw()
        print("Choose the .replay file")
        path = filedialog.askopenfilename(initialdir = '')
        o_path = os.path.dirname(path)
        self.am = carball.analyze_replay_file(path, output_path = o_path + '0.json', overwrite=True) # Analysis manager

    def load_replay_from_path(self, path):
        o_path = os.path.dirname(path)
        self.am = carball.analyze_replay_file(path, output_path = o_path + '0.json', overwrite=True) # Analysis manager

    def get_hits(self):
        kickoff_frames, first_touch_frames = self.am.get_kickoff_frames(self.am.game, self.am.protobuf_game, self.am.data_frame)
        self.hits = BaseHit.get_hits_from_game(self.am.game, self.am.protobuf_game, self.am.id_creator, self.am.data_frame, first_touch_frames)
        self.hit_frames = list(self.hits.keys())

if __name__ == "__main__":
    try:
        h = HitFinder()
        h.load_replay()
        h.get_hits()
        # print(h.hits)
        b = nndg.generate_nn_data_from_hits(h.am, h.hits, 10, 5)
        print(b)
    except Exception as e:
        print(e)
    import code
    code.interact(local=locals())
