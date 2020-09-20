from rlbot.utils.game_state_util import CarState
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers import Controller, AerialController, DrivingController


class ControllerController:
    """ControllerController is a class that will activate/deactive controllers and set the proper one active since the car has highly
    different dynamics when driving vs flying which means the control algorithm will need to change depending on which
    state the car is in. Furthermore, driving on the wall might require a separate controller vs ground driving.
    """    
    drivingController: DrivingController
    aerialController: AerialController
    activatedController: Controller

    def __init__(self, drivingControllerType, aerialControllerType):
        drivingController = drivingControllerType
        aerialController = aerialControllerType

    def update_activated_controller(self, carState: CarState):
        raise NotImplementedError