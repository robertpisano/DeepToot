from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.AerialController import AerialController
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.DrivingController import DrivingController
from rlbot.agents.base_agent import SimpleControllerState


class OpenLoopAerialController(AerialController):
    """OpenLoop type of controller. This controller will just take the control prescribed in
        the control_trajectory passed into the calculate_controls method at the time specified by
        the argument dt.

    Inherits:
        Controller
    """    
    def __init__(self):
        self.controllerParams = {'kf':0.0, }
        self.miscOptions = {'approximationType':'linear', 'time_agnostic':False}
        print(self.controllerParams['kf'])
    
    def calculate_controls(self, time:float, control_trajectory):
        """Find the control at the time specified t. update_control_state() should be called directly before calling this
        Args:
            trajectory (Trajectory): 

        Raises:
            NotImplementedError: [description]
        """        

        # """ update the controller's current time """
        self.update_control_state(time) 

        # """ Do math here to calculate controler """

        raise NotImplementedError

        # """ return a SimpleControllerState filled with calculated controls """
        return SimpleControllerState()

    def update_control_state(self, time):
        t = time

    def set_t0(self, time: float):
        t0 = time

