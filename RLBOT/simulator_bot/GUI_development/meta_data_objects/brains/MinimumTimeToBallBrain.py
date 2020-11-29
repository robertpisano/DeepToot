from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.Brain import Brain
from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory

class MinimumTimeToBallBrain(Brain):
    params = {'utime': 1, 'ufuel': 0, 'guided':True}
    miscOptions = {}
    pre_calculate: bool
    state_trajectory: Trajectory
    control_trajectory: list

    def __init__(self):
        self.pre_calculate = False