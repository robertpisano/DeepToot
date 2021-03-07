from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects import InitialConditions
from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject

class Brain(MetaDataObject):
    pre_calculate: bool # Tells system manager that you should pre-calculate a trajectory before running controller
    parameters: dict
    miscOptions: dict
    state_trajectory: Trajectory # The calculated trajectroy. This can be updated once like in calculate, or updated each frame with think()
    control_trajectory: list # The control trajectory. same idea as state trajectory.
    
    

# Calculate will be used to pre-calculate a trajectory to follow
    def calculate(self, conditions: InitialConditions):
        raise NotImplementedError

# This is used for "real-time" calculations. For example a real time MPC algorithm or a NN calculation
    def think(self):
        raise NotImplementedError    