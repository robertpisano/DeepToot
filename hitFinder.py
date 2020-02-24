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
import NeuralNetworkTrainer

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

class RawDataManager():
    def __init__(self, data):
        self.data = data
    @staticmethod
    def save_raw_data(datain, dataout):
        dill.dump(datain, open(r'D:\\Documents\\RL Replays\\rawin.p', 'wb'))
        dill.dump(dataout, open(r'D:\\Documents\\RL Replays\\rawout.p', 'wb'))
    @staticmethod
    def load_raw_data():
        datain = dill.load(open(r'D:\\Documents\\RL Replays\\rawin.p', 'rb'))
        dataout = dill.load(open(r'D:\\Documents\\RL Replays\\rawout.p', 'rb'))
        return datain, dataout
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
            h = HitFinder()
            h.am = HitFinderFactory.load_analysis_manager()
            h.get_hits()
            b_in, b_out = nndg.generate_nn_data_from_hits(h.am, h.hits, 10, 1)
            # rawin, rawout = RawDataManager.load_raw_data()
        if raw == 't': # Test network quickly
            rawin, rawout = RawDataManager.load_raw_data()

        else:
            h = HitFinder()
            h.load_replay()
            h.get_hits()
            # print(h.hits)
            rawin, rawout = nndg.generate_nn_data_from_hits(h.am, h.hits, 10, 5)
            # HitFinderFactory.save_batch_data(b)
            # h.save_protobuf_hardcoded()
            try:
            
                HitFinderFactory.save_analysis_manager(h.am)
                RawDataManager.save_raw_data(rawin.to_numpy(), rawout.to_numpy())
            except Exception as e:
                PrintException()
                # print('Trying trace before pickling')
                # dill.detect.trace(True)
                # dill.detect.errors(h.am)
                # dill.pickles(h.am)
                
        
        # print(b_in)
        # rawin = b_in.to_numpy()
        # rawout = b_out.to_numpy()
        # frameWindow = b_in.batch[0] # quicker debugging
        # gameFrame = frameWindow.frames[0]
        # playerState = gameFrame.p1

        #Neural Network testing
        nn = NeuralNetworkTrainer.LSTM_Model()
        nn.train(rawin, rawout)
    except Exception as e:
        PrintException()

    import code
    code.interact(local=locals())
