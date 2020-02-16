import sys
import linecache
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


# Load .replay, save as .proto
# Tkinter frame for windows explorer dialog to pop up
root = tk.Tk()
root.withdraw()
print("Choose the .replay file")
print('Please choose the .replay file you wish to load')
path = filedialog.askopenfilename(initialdir = '')
o_path = os.path.dirname(path)
am = carball.analyze_replay_file(path, output_path = o_path + '0.json', overwrite=True) # Analysis manager

# Save .proto file
print('Choose file to save .proto, may have to create a new file and give it .proto ending')
file = open(filedialog.askopenfilename(initialdir = ''), 'wb')
ProtobufManager.write_proto_out_to_file(file, am.protobuf_game)

# I would like to start here when looking for data to pull out for neural network training, pretend I already converted
# all the .replay files to .proto files and I now have a library of .proto files to use for data collection

# Load .proto
print('Please choose the .proto file you wish to load')
file = open(filedialog.askopenfilename(initialdir = ''), 'rb') # Find .proto file
proto_game = ProtobufManager.read_proto_out_from_file(file)
        

# Get full gamestate data from specific frame#
# self.proto_game.frame[987].player1 (this is the command i'm looking for)
# For example the above would give me a strucuture of the 
# pos, vel, orientation, angular_vel etc...
# for frame# 987 for player1
# Is that possible from the .proto data?
print(proto_game)