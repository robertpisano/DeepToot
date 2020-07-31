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
from DeepToot.src.repositories.neural_net_model_repository import NeuralNetModelRepository
from DeepToot.src.data_generation.entities.neural_net.controller_state.controller_state_neural_net_model import ControllerStateNeuralNetModel

class ControllerStateTrainer:
    None

if __name__ == "__main__":
    data_path = os.path.dirname(DeepToot.__file__) + "\SavedData\stupid.csv"
    dataframe = pd.read_csv(data_path)

    #init game trajectory builder
    traj_length = 3
    data_frames_length = len(dataframe.index) - 2
    gtb = GameTrajectoryBuilder(traj_length)

    #Make array of all states from dataframe
    # NB: Convert State in_data into relative coordinates
    # Use deltax, deltay, and rotate all other values to new coordinate system where we rotate the world coordainte system to align with
    # the car's coordinate system. Angular velocity should not be effected numerically.
    for index in range(0, data_frames_length):
        for i in range(0, traj_length):
            gtb.add_bot_state(
                StateTransformer.from_pandas_frame_to_car_state(dataframe.iloc[index+i])
            )
        gtb.zero_fill_unneeded_queues()
        gtb.build() #BUild trajectory
    
        # make a state vector buffer from game trajectory builder
        state_buff = [gtb.bot_queue[1].time, gtb.bot_queue[1].position.x(), gtb.bot_queue[1].position.y(), gtb.bot_queue[1].velocity.x(), gtb.bot_queue[1].velocity.y(), gtb.bot_queue[1].ang_vel.z(),
                        gtb.bot_queue[2].time, gtb.bot_queue[2].position.x(), gtb.bot_queue[2].position.y(), gtb.bot_queue[2].velocity.x(), gtb.bot_queue[2].velocity.y(), gtb.bot_queue[2].ang_vel.z()]
        
        # output data vector buffer
        control_buff = [dataframe.loc[index, 'controller_throttle'], dataframe.loc[index, 'controller_steer'], dataframe.loc[index, 'controller_boost']]
        
        # Fill the elements of the in_data, out_data arrays
        if index == 0: # Initialize in_data and out_data as empty arrays with proper size
            in_data = np.empty(shape = (data_frames_length, len(state_buff)))
            out_data = np.empty(shape = (data_frames_length, len(control_buff)))
        # Fill the arrays
        for j, elem in enumerate(state_buff):
            in_data[index][j] = elem
        for k, celem in enumerate(control_buff): 
            out_data[index][k] = celem

    # Initialize Controller State NN
    simp = ControllerStateNeuralNetModel(2)

    # Fit neural net to data
    simp.fit(in_data, out_data, epochs=200, batch_size=100)
    print("was actually able to fit the model....")
    print("model layers")
    print(simp.layers)
    repository = NeuralNetModelRepository(model = simp)
    repository.save()
    print("serializeddd")
    loaded_model = repository.load()
    print("loaded model")

    # Comparing traned label with a predicted label from two random consecutive states
    test_index = 81

    gtb_new = GameTrajectoryBuilder(2)
    gtb_new.add_bot_state(
                StateTransformer.from_pandas_frame_to_car_state(dataframe.iloc[test_index])
            )
    gtb_new.add_bot_state(
        StateTransformer.from_pandas_frame_to_car_state(dataframe.iloc[test_index+1])
    )
    gtb_new.zero_fill_unneeded_queues()
    gtb_new.build()
    state_buff = np.empty(shape = (1, 12))
    buff = [gtb_new.bot_queue[0].time, gtb_new.bot_queue[0].position.x(), gtb_new.bot_queue[0].position.y(), gtb_new.bot_queue[0].velocity.x(), gtb_new.bot_queue[0].velocity.y(), gtb_new.bot_queue[0].ang_vel.z(),
                    gtb_new.bot_queue[1].time, gtb_new.bot_queue[1].position.x(), gtb_new.bot_queue[1].position.y(), gtb_new.bot_queue[1].velocity.x(), gtb_new.bot_queue[1].velocity.y(), gtb_new.bot_queue[1].ang_vel.z()]

    # Fill state+buff array
    for i, elem in enumerate(buff):
        state_buff[0][i] = elem
    # Create desired control array for comparison
    desired_control = np.array([dataframe.loc[test_index-1, 'controller_throttle'], dataframe.loc[test_index-1, 'controller_steer'], dataframe.loc[test_index-1, 'controller_boost']])

    # Predict controls for state_buff input
    nnout = simp.predict(state_buff)
    # Same as above but for loaded model instead
    nn_loaded_out = loaded_model.predict(state_buff)

    # PrInt for comparison
    print('des_controls: ' + str(desired_control))
    print('nnout: ' + str(nnout))
    print('nnout2: ' + str(nn_loaded_out))
