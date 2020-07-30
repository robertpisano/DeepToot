# from DeepToot.src.data_generation.simple_controller_state_generator import SimpleControllerStateGenerator
import numpy as np
# import tensorflow as tf
# import keras
import pandas as pd
import os
import DeepToot
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder
from DeepToot.src.data_generation.state_transformer import StateTransformer
from DeepToot.src.data_generation.simple_controller_state_generator import SimpleControllerStateGenerator
# from DeepToot.src.repositories.neural_net_model_repository import NeuralNetModelRepository
# from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel

class ControllerStateTrainer:
    None
    # position = range(-100, 100)
    # velocity = range(-100, 100)

    # for x in poisiton:
    #     for y in velocity

if __name__ == "__main__":
    data_path = os.path.dirname(DeepToot.__file__) + "\SavedData\stupid.csv"
    dataframe = pd.read_csv(data_path)

    #Define numpy arrays with proper dimensions
    in_data = np.array([], ndmin = 3) #Time and State
    out_data = np.array([], ndmin = 2) # Controller
    states = []
    controller_states = []

    #init game trajectory builder
    gtb = GameTrajectoryBuilder(len(dataframe.index))

    #Make array of all states from dataframe
    for index in range(2, len(dataframe.index)):
        df_buff = dataframe.iloc[index]
        # print(df_buff)
        state = StateTransformer.from_pandas_frame_to_car_state(df_buff)
        gtb.add_bot_state(state)
    gtb.zero_fill_unneeded_queues()
    gtb.build() #BUild trajectory

    #Make array of all controller states
    for index in range(0, len(dataframe.index)-2):
        df_buff = dataframe.iloc[index]
        control_state = SimpleControllerStateGenerator.from_data_frame(df_buff)
        controller_states.append(control_state)

    # Create Numpy Arrays
    # NB: each batch should have two states in them, dimensionality is (batch size, state n, state n+1)
    for i, state in enumerate(gtb.bot_queue):
        if i == 0:
            in_data = np.array([i,[state.time, state.position.x, state.position.y, state.velocity.x, state.velocity.y, state.ang_vel.z], ['put future state here']], ndmin=3)
        else:
            in_data = np.append(in_data, np.array([i,[state.time, state.position.x, state.position.y, state.velocity.x, state.velocity.y, state.ang_vel.z], ['put future state here']], ndmin=3), axis=0)





    print('some bs')








    amax = 900 #maximum acceleration
    controls = np.linspace(-1, 1, 21) #controls vector

    time = np.linspace(0, 10, 11) #time vector
    vi = 0.0 #intial velocity
    trajectories = [] #list to append trajectories
    controls_list = []
    for u in controls: # for every control in controls -> 21 betweeon -1 and 1
        buffer = u*amax*time + vi # the individual control input * 900 * time vector
        trajectories.append(buffer) 
        controls_list.append(np.repeat(u, 11))
 
    inbuff = []
    outbuff = []
    
    print("controls list")
    print(controls_list)
    print("trajectories")
    print(trajectories)
    
    #iterate through trajectories 
    for j, traj in enumerate(trajectories):
        for i, state in enumerate(traj): #through states in trajectory
            if j==0 and i==0: #if were on the first state of the first trajectory
                indata = np.array([traj[i], traj[i+1]], dtype = np.float)  # create a tuple of the two states
                indata = np.expand_dims(indata, axis=1) #change shape from (2,)- [1,2] to (2, 1) - [[1], [2]]
                outdata = np.array([controls_list[j][i]]) #
                outdata = np.expand_dims(outdata, axis=1)
            elif i == len(traj)-1: # if were in the last state in the trajectory, skip because tehres no two elements
                break
            else:
                indata = np.append(indata, np.expand_dims(np.array([traj[i], traj[i+1]]), axis=1), axis = 1) #if its not the first of the first, append another two 
                buff = np.array([controls_list[j][i]]) 
                buff = np.expand_dims(buff, axis=1)
                outdata = np.append(outdata, buff, axis = 1)
            print("trajectory")
            print(traj)
            print("states")
            print(np.array([traj[i], traj[i+1]]))
            print("states to add")
            print(np.expand_dims(np.array([traj[i], traj[i+1]]), axis=1))
            print("indata")
            print(indata)
            print("outdata")
            print(outdata)

    


    simp = ControllerStateNeuralNetModel(11)
    print("about to fit the data")
    print(indata.T)
    print(outdata.T)

    simp.fit(indata.T, outdata.T, epochs=1000, batch_size=100)
    print("was actually able to fit the model....")
    print("model layers")
    print(simp.layers)
    repository = NeuralNetModelRepository(model = simp)
    repository.save()
    print("serializeddd")
    print(tf.__version__)
    loaded_model = repository.load()
    print("loaded model")
    print(loaded_model)
    print(loaded_model.layers)
    print(simp.layers)
    print(loaded_model.optimizer)     
    print(simp.optimizer)
    t1 = trajectories[3][5]
    t2 = trajectories[3][6]
    u_model = controls_list[3][5]
    intest = np.array([t1, t2], dtype=np.float)
    intest = np.expand_dims(intest, axis=1)
    nnout = simp.predict(intest.T)
    nn_loaded_out = loaded_model.predict(intest.T)

    print('t1: ' + str(t1))
    print('t2: ' + str(t2))
    print('nnout: ' + str(nnout))
    print('nnout2: ' + str(nn_loaded_out))
    print('math model out: '  + str(u_model))

    # print(u)
    # print(trajectories)
