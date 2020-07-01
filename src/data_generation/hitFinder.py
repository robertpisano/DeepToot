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
from DeepToot.src.data_generation.NeuralNetworkDataGenerator import NeuralNetworkManager
from carball.analysis.events.hit_detection.base_hit import BaseHit
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator
from carball.json_parser.game import Game
from carball.generated.api.stats.events_pb2 import Hit
from typing import Dict


class HitFinder():
    def __init__(self):
        self.hits = [] # Dictionary that carball BaseHit.get_hits_from_game() returns
        self.hit_frames = [] # List of frame numbers that hits happened at to make accessing self.hits easier since it is an unordered dictionary
    
    def load_from_am(self, analysis_manager: AnalysisManager):
        self.am = analysis_manager

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

class RawReplayData():
    def __init__(self):
        None
    
    def setData(self, gamedata: DataFrame, playerControlData: list, hitData: Dict):
        self.game = gamedata
        self.players = playerControlData
        self.hits = hitData
        
    def save_at(self, path):
        try:
            dill.dump(self, open(path + '.p', 'wb'))
        except Exception:
            print(e)

    def save(self):
        try:
            dill.dump(self, open('D:\\Documents\\RL Replays\\rawForControls\\rawReplay.p', 'wb'))
        except Exception:
            os.umask(0)
            os.makedirs(r'D:\\Documents\\RL Replays\\rawForControls', exist_ok=True, mode=0o777)
            dill.dump(self, open('D:\\Documents\\RL Replays\\rawForControls\\rawReplay.p', 'wb'))

    @staticmethod
    def load_from(path):
        try:
            return dill.load(open(path, 'rb'))
        except:
            print('error with dill.load on pickle path')
    # TODO: Hard coded loading right now, need to find way to do dynamic loading? or better type of loading
    @staticmethod
    def load():
        try:
            return dill.load(open('D:\\Documents\\RL Replays\\rawForControls\\rawReplay.p', 'rb'))
        except Exception as e:
            print('Error loading rawreplay')
            print(e)
            return None
            

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
        print('y: load analysis manager and generate nn data test')
        print('n: initialize analysis manager from .replay file')
        print('t: load raw nn data')
        print('c: calculate controls from replay file')
        print('loadc: load raw game/controls/hit data saved from running c previously')

        raw = input('Choose from above:')
        if raw == 'y':
            import NeuralNetworkDataGenerator as nndg
            from NeuralNetworkDataGenerator import TrainingBatch
            import NeuralNetworkTrainer
            h = HitFinder()
            h.am = HitFinderFactory.load_analysis_manager()
            h.get_hits()
            b_in, b_out = nndg.generate_nn_data_from_hits(h.am, h.hits, 10, 1)
            # rawin, rawout = RawDataManager.load_raw_data()

        # get input from scenario, save scenario dataframe 
        if raw == 't': # Test network quickly
            rawin, rawout = RawDataManager.load_raw_data()

        # Calculate controls from replay file
        if raw == 'c':
            h = HitFinder()
            h.load_replay()
            cc = ControlsCreator()
            cc.get_controls(h.am.game)
            print(cc)
            rrd = RawReplayData()
            rrd.setData(h.am.data_frame, cc.players, h.hits)
            rrd.save()
        
        if raw == 'loadc':
            rrd = RawReplayData.load()

        if raw == 'n':
            import NeuralNetworkDataGenerator as nndg
            from NeuralNetworkDataGenerator import TrainingBatch
            import NeuralNetworkTrainer
            h = HitFinder()
            h.load_replay()
            h.get_hits()
            rawin, rawout = nndg.generate_nn_data_from_hits(h.am, h.hits, NeuralNetManager.window_length, 5)

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
