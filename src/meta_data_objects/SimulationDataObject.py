from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder, GameTrajectory
from DeepToot.src.meta_data_objects.controllers import Controller, AerialController, DrivingController
from DeepToot.src.meta_data_objects.brains.Brain import Brain
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.src.meta_data_objects.AbstractMetaDataObjectFactory import AbstractMetaDataObjectFactory
from DeepToot.src.meta_data_objects.controllers.ControllerFactory import ControllerFactory, ControllerSchema
from DeepToot.src.meta_data_objects.brains.BrainFactory import BrainSchema, BrainFactory
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory, InitialConditionsSchema

from marshmallow import Schema, fields, post_load

class SimulationDataObject():
    """Simulation Data Object will hold all the data for a simultaion. It in instantiated from the GUI
    and then sent over the socket to the server which the bot ultimately gets. The bot will then fill in
    the "SimulationTrajectory" object with a GameTrajectory() and then send the object back to the GUI.

    Inherits:
        MetaDataObject
    """    
    drivingController:Controller
    aerialController:Controller
    brain:Brain
    simulationTrajectory:GameTrajectory

    def __init__(self, dc:Controller, ac:Controller, b:Brain, ic:InitialConditions):
        self.simulationTrajectory = GameTrajectoryBuilder(0).build() # Initialize with empty game trajectory
        self.drivingController = dc
        self.aerialController = ac
        self.brain = b
        self.initialConditions = ic
        self.__name__ = 'SimulationDataObject'
    
    
    def init_from_dict(self, dict: dict):
        cls = SimulationDataObject(AbstractMetaDataObjectFactory.create(dict['drivingController'].__name__)\
                                                                        .init_from_dict(dict, 'drivingController'), 
                                    dict['aerialController'], 
                                    dict['brain'], 
                                    dict['initialConditions'])
        return cls

class SimulationDataObjectSchema(Schema):
    __name__ = fields.Str()
    drivingController = fields.Nested(ControllerSchema)
    aerialController = fields.Nested(ControllerSchema)
    brain = fields.Nested(BrainSchema)
    initialConditions = fields.Nested(InitialConditionsSchema)
    
    @post_load
    def make_obj(self, data, **kwargs):
        sd = SimulationDataObject(dc = data['drivingController'], ac = data['aerialController'], b=data['brain'], ic=data['initialConditions'])
        return sd