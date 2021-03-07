from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.src.data_generation.entities.state.base_state import BaseState
from DeepToot.src.data_generation.entities.state.car_state import CarState, CarStateBuilder
from DeepToot.src.data_generation.entities.state.ball_state import BallState, BallStateBuilder

class InitialConditionsTraining(InitialConditions):
    params = {'Car':CarState, 'Ball':BallState}
w