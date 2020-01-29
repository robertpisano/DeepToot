# import carball
#
# analysis_manager = carball.analyze_replay_file('D:/Documents/RL Replays/1.replay',
#                                       output_path='D:/Documents/RL Replays/1.json',
#                                       overwrite=True)
# proto_game = analysis_manager.get_protobuf_data()
# print('something')
# # you can see more example of using the analysis manager below


import carball
import os
import gzip
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import numpy as np
import matplotlib.pyplot as plt


# analysis_manager = carball.analyze_replay_file('D:/Documents/RL Replays/2.replay',
#                                       output_path='D:/Documents/RL Replays/2.json',
#                                       overwrite=True)
# _json is a JSON game object (from decompile_replay)
game = Game()
game.initialize(file_path = 'D:/Documents/RL Replays/2.json')# loaded_json=_json)

analysis_manager = AnalysisManager(game)
analysis_manager.create_analysis()

# write proto out to a file
# read api/*.proto for info on the object properties
with open(os.path.join('output.pts'), 'wb') as fo:
    analysis_manager.write_proto_out_to_file(fo)

# write pandas dataframe out as a gzipped numpy array
with gzip.open(os.path.join('output.gzip'), 'wb') as fo:
    analysis_manager.write_pandas_out_to_file(fo)

# return the proto object in python
proto_object = analysis_manager.get_protobuf_data()

# perform full analysis
# analysis_manager.perform_full_analysis(game, proto_object)

# return the pandas data frame in python
df1 = analysis_manager.get_data_frame()
df2 = analysis_manager.get_data_frame()

# Dave data to csv file
df1.to_csv(r'test.csv')

# Search through car and balls velocity state and check for violation of conservation of energy
carvel = np.array([df1['Cheerio']['vel_x'], df1['Cheerio']['vel_y'], df1['Cheerio']['vel_z']], dtype=float)
ballpos = np.array([df1['ball']['pos_x'], df1['ball']['pos_y'], df1['ball']['pos_z']], dtype=float)
ballvel = np.array([df1['ball']['vel_x'], df1['ball']['vel_y'], df1['ball']['vel_z']], dtype=float)
ballvel = np.nan_to_num(ballvel)
time = np.array(df1['game']['time'], dtype=float)
carvelmag = np.linalg.norm(carvel, axis=0)
ballvelmag = np.linalg.norm(ballvel, axis=0)
ballposmag = np.linalg.norm(ballpos, axis=0)
t = np.mean(time)
ballvelcalcx = np.gradient(ballpos[0,:], time)
ballvelcalcy = np.gradient(ballpos[1,:], time)
ballvelcalcz = np.gradient(ballpos[2,:], time)
ballacc = np.gradient(ballvelmag, time)
ballvelmagcalc = np.gradient(ballposmag, time)
# clip velocity calculation
ballvel = np.clip(ballvel, -2500, 2500)

plt.figure(1)
plt.plot(time, ballvelmag, color='r')
plt.plot(time, ballvelmagcalc, color='b')
plt.show()
