# Hit finder will open up a csv file, or loop through a list of csv files
# to find hits the frame number that the hit occured at
import sys
import linecache
import pickle
import dill
import os
import subprocess
import pandas as pd
from pandas import DataFrame
import tkinter as tk
from tkinter import filedialog
import carball
from carball.analysis.events.hit_detection.base_hit import BaseHit
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.analysis.analysis_manager import AnalysisManager
from carball.json_parser.game import Game
from carball.generated.api.stats.events_pb2 import Hit
from typing import Dict
import NeuralNetworkDataGenerator as nndg
from NeuralNetworkDataGenerator import TrainingBatch

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

    def load_from_protobuf_hardcoded(self):
        file = open(r'D:\Documents\RL Replays\gameproto.proto', 'rb')
        self.proto_game = ProtobufManager.read_proto_out_from_file(file)
        
    def save_protobuf_hardcoded(self):
        file = open(r'D:\Documents\RL Replays\gameproto.proto', 'wb')
        ProtobufManager.write_proto_out_to_file(file, self.am.protobuf_game)

    def get_hits(self):
        kickoff_frames, first_touch_frames = self.am.get_kickoff_frames(self.am.game, self.am.protobuf_game, self.am.data_frame)
        self.hits = BaseHit.get_hits_from_game(self.am.game, self.am.protobuf_game, self.am.id_creator, self.am.data_frame, first_touch_frames)
        self.hit_frames = list(self.hits.keys())

class HitFinderFactory():
    def __init__(self):
        root = tk.Tk()
        root.withdraw()

    @staticmethod
    def save_analysis_manager(am):
        dill.dump(am, open(r'D:\\Documents\\RL Replays\\am.p', 'wb'))

    @staticmethod
    def load_analysis_manager():
        am = dill.load(open(r'D:\\Documents\\RL Replays\\am.p', 'rb'))
        return am
    


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


if __name__ == "__main__":
    try:
        raw = input('Load Hit Finder? (y/n)')
        if raw == 'y':
            # b = HitFinderFactory.load_batch_data()
            # load hard coded for now for quickness
            # path = os.path.join('Documents', 'RL Replays', 'batchdata1.p')
            # b = pickle.load(open(r'D:\\Documents\\RL Replays\\batchdata.p', 'rb'))
            h = HitFinder()
            # h.load_from_protobuf_hardcoded()
            h = HitFinderFactory.load_analysis_manager()
            h.get_hits()
            b = nndg.generate_nn_data_from_hits(h.am, h.hits, 10, 5)

        else:
            h = HitFinder()
            h.load_replay()
            h.get_hits()
            # print(h.hits)
            b = nndg.generate_nn_data_from_hits(h.am, h.hits, 10, 5)
            # HitFinderFactory.save_batch_data(b)
            # h.save_protobuf_hardcoded()
            try:
            
                HitFinderFactory.save_analysis_manager(h.am)
            except Exception as e:
                PrintException()
                print('Trying trace before pickling')
                dill.detect.trace(True)
                dill.detect.errors(h.am)
                dill.pickles(h.am)
                
        
        print(b)

        frameWindow = b.batch[0] # quicker debugging
        gameFrame = frameWindow.frames[0]
        playerState = gameFrame.p1
    except Exception as e:
        PrintException()

    import code
    code.interact(local=locals())
