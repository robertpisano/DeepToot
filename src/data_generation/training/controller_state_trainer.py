# from DeepToot.src.data_generation.simple_controller_state_generator import SimpleControllerStateGenerator
import numpy as np
# import tensorflow as tf
# import keras
import pandas as pd
import os
import DeepToot
from DeepToot.src.data_generation.entities.neural_net.neural_net_package_factory import NeuralNetPackageFactory
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder
from DeepToot.src.data_generation.entities.state.state_transformer import StateTransformer
from DeepToot.src.data_generation.simple_controller_state_generator import SimpleControllerStateGenerator
from DeepToot.src.data_generation.entities.state.ball_state import BallStateBuilder
from DeepToot.src.data_generation.entities.state.car_state import CarStateBuilder
from DeepToot.src.repositories.neural_net_model_repository import NeuralNetModelRepository
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
            gtb.add_game_state(
                bot_state = StateTransformer.from_pandas_frame_to_car_state(dataframe.iloc[index+i]),
                ball_state = BallStateBuilder().build(),
                car_state = CarStateBuilder().build()
            )

        neural_net_package = NeuralNetPackageFactory(game_trajectory = gtb.build()).create("controller", params={length:2})
        neural_net_package.model.fit(
            neural_net_package.transformer.pandas_to_label(), 
            neural_net_packate.transformer.pandas_to_feature(), 
            epochs=200, 
            batch_size=100
        )
        NeuralNetModelRepository(model=neural_net_package.model).save()


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
