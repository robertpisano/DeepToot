from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from rlbot.agents.base_agent import SimpleControllerState

class DrivingController(Controller):
    name = 'DrivingController'
    params = {"kp":1, "kd":1, "A":2, "B":3, "C":4}
    miscOptions = {"opt1":None}
    def __init__(self):
        self.__name__ = 'DrivingController'
        pass

    def calculate_control(self, packet, index) -> SimpleControllerState:
        controller_state = SimpleControllerState()
        controller_state.throttle = .1
        controller_state.steer = 1
        return controller_state