from DeepToot.src.meta_data_objects.controllers import Controller
from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.controllers.DrivingController import DrivingController
from rlbot.agents.base_agent import SimpleControllerState


class OpenLoopDrivingController(DrivingController):
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
    
    def calculate_controls(self, brain, packet, index):
        """Find the control at the time specified t. update_control_state() should be called directly before calling this
        Args:
            trajectory (Trajectory): 

        Raises:
            NotImplementedError: [description]
        """        

        # """ update the controller's current time """
        self.t = packet.game_info.seconds_elapsed

        # """ Do math here to calculate controler """
        dt = self.t - self.t0
        # Get index
        for i in range(0, len(brain.tf) - 1):
            idx = i
            if(idx == len(brain.tf)):
                break
            if(dt < brain.tf[idx+1]):
                break
            if(dt > brain.tf[-1]):
                print('trajectory is finished')
                return 'finished'
        accel = brain.opt.car.accel[idx]
        steer = brain.opt.car.u_turn[idx]
        controller = SimpleControllerState()
        controller.throttle = 800
        if accel > 1600: controller.boost = True
        controller.steer = steer

        # """ return a SimpleControllerState filled with calculated controls """
        return controller

    def update_control_state(self, time):
        self.t = time

    def set_t0(self, time: float):
        self.t0 = time

