from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects import InitialConditions
from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject

class Brain(MetaDataObject):

    pre_calculate: bool
    parameters: dict
    miscOptions: dict
    state_trajectory: Trajectory
    control_trajectory: list

    def calculate(self, conditions: InitialConditions):
        raise NotImplementedError

    def think(self):
        raise NotImplementedError    