from rlbot.agents.base_agent import SimpleControllerState
from DeepToot.src.meta_data_objects.MetaDataObject import MetaDataObject

class Controller(MetaDataObject, object):
    """Controller Abstract class. Holds controller parameters, miscilaneous options, and
    a calculate_control method which should return a SimpleControllerState()
    """    
    
    t0: float  # initial time that controller was started"""
    t: float   # the current time the controller should use when doing its time based calculations"""
    # params: dict # all the parameters that define the controller, typically gains/coefficients """
    # miscOptions: dict      # misc options that the controller may need to operate properly"""
    
    def __init__(self):
        pass
    
    def calculate_control(self, *args) -> SimpleControllerState:
        raise NotImplementedError

    def update_controller_state(self):
        raise NotImplementedError

    def update_t0(self, t0: float):
        self.t0 = t0
        

