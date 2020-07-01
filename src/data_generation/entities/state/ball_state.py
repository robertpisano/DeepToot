from DeepToot.src.data_generation.entities.state.base_state import BaseState
from DeepToot.src.data_generation.entities.state.base_state import BaseStateBuilder


class BallState(BaseState):
    def __init__(self, position, velocity, ang_vel, orientation, time):
        super().__init__(position=position, velocity=velocity, ang_vel=ang_vel, orientation=orientation, time=time)

class BallStateBuilder(BaseStateBuilder):
    None