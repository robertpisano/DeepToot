# Import base classes
from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.brains.Brain import Brain
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko
from DeepToot.src.meta_data_objects.SimulationDataObject import SimulationDataObject, SimulationDataObjectSchema

# Import Factory modules
import DeepToot.src.meta_data_objects.controllers.ControllerFactory as controller_fac
import DeepToot.src.meta_data_objects.brains.BrainFactory as brain_fac
import DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsFactory as init_fac

# Comms imports
from DeepToot.src.comms_util.server import Server
from DeepToot.src.comms_util.comms_protocol import CommsProtocol

# Dynamic class import for testing
import DeepToot.src.dynamic_class_util.dynamic_class as dc
from  DeepToot.src.dynamic_class_util.dynamic_class import TestClass

# Statemachine import
from DeepToot.src.state_machine_util.state_machine import StateMachine

# from rlbot.messages.flag.RigidBodyTick import RigidBodyTick
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder, GameTrajectory
from DeepToot.src.meta_data_objects.SerializationFactory import SerializationFactory


from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from threading import Thread, Lock
import select
import socket
import pickle
import dill
import yaml
import time
from importlib import reload
import numpy as np

class SystemManager(Server):
    # drivingController: Controller
    # aerialController: Controller
    # brain: Brain
    # initialCondition: InitialConditions
    # simulationTrajectory: GameTrajectory
    # dataObject: SimulationDataObject

    def __init__(self, index, ip: "", port: int):
        # Initialize socket reading thread
        Server.__init__(self, ip, port)

        self.execute = False # Execute flag, lets manager know to terminate/or execute
        self.initialized = False # Initialize flag, will let manager know if the game state has been initialized or not

        self.object_types = {'dc':'DrivingController', 'ac':'DrivingController', 'brain':'MinimumTimeToBall', 'initial_conditions':'InitialConditionsGekko'}
        
        self.drivingController = Controller()
        self.aerialController = Controller()
        self.brain = Brain()
        self.initialCondition = InitialConditionsGekko()
        self.simulationTrajectory = GameTrajectoryBuilder(0).build()
        self.dataObject = SimulationDataObject(Controller(), Controller(), Brain(), InitialConditionsGekko())
        self.test = TestClass()

    def update(self, dataObject: SimulationDataObject):
        self.drivingController = dataObject.drivingController
        self.aerialController = dataObject.aerialController
        self.brain = dataObject.brain
        self.initialConditions = dataObject.initialConditions
        self.simulationTrajectory = dataObject.simulationTrajectory
        # Reset started flag
        self.started=False

    def run_bot(self, packet: GameTickPacket, rigid_packet, bot):
        # Check msg_queue
        try:
            msg = self.msg_queue.get(block=False)
            print("msg type: ", msg.type)
        except:
            msg = None
        
        try:
            # Do things with message
            if msg != None: self.msg_update(msg)
            # print("val: ", self.test.val)
        except Exception as e:
            print(e)

        if(self.execute == True):
            # Initialize if necessary
            if(not self.initialized): 
                # Do pre-calculations
                self.brain.calculate(self.initialCondition)
                self.drivingController.set_t0(packet.game_info.seconds_elapsed)
                self.initialize_game_state(bot)
                self.initialized = True

            # Get controller action
            controller_state = SimpleControllerState()
            controller_state = self.drivingController.calculate_control(packet=packet, index=bot.index)
            return controller_state
        else:
            return SimpleControllerState()

    def initialize_game_state(self, bot):
        # Get the initial car state
        ic = self.initialConditions.params
        pos = Vector3(x=ic['sxi'], y=ic['syi'], z=17.01)
        vel = Vector3(x=ic['v_magi']*np.cos(ic['yawi']), y=ic['v_magi']*np.sin(ic['yawi']), z=0)
        rot = Rotator(roll = ic['rolli'], pitch=ic['pitchi'], yaw=ic['yawi'])
        cs = CarState(Physics(location=pos, velocity=vel, rotation=rot), boost_amount=100)

        # Get the inital ball state
        bpos = Vector3(x=ic['bxi'], y=ic['byi'], z=92.75)
        bvel = Vector3(x=ic['bvxi'], y=ic['bvyi'], z=0)
        bs = BallState(Physics(location=bpos, velocity=bvel))

        # Set game state
        gs = GameState(ball = bs, cars={bot.index: cs})

        # Set environment
        bot.set_game_state(gs)

    def msg_update(self, msg):
# Update classes dynamically
        if(msg.type == CommsProtocol.types["update"]):
            # reload factory modules
            reload(controller_fac)
            reload(brain_fac)
            reload(init_fac)
            from DeepToot.src.meta_data_objects.controllers.ControllerFactory import ControllerFactory, ControllerSchema
            from DeepToot.src.meta_data_objects.brains.BrainFactory import BrainFactory, BrainSchema
            from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory, InitialConditionsSchema
            from DeepToot.src.dynamic_class_util.dynamic_class import TestClass
            self.test = TestClass()
            self.drivingController = ControllerFactory.create('OpenLoopDrivingController')
            self.aerialController = ControllerFactory.create(self.object_types['ac'])
            self.brain = BrainFactory.create('MinimumTimeToBall')
            self.initialConditions = InitialConditionsFactory.create('InitialConditionsGekko')
            # print(self.test.val)
            pass

# Start Execution
        if(msg.type == CommsProtocol.types['execute']):
            # execute
            print('executing... ')
            # Update execute flag
            self.execute = True

            # Pull object types from message, update local objects using marshmallow loads.
            self.dataObject = SimulationDataObjectSchema().loads(msg.data)
            print(self.dataObject)
            pass

# Terminate the execution
        if(msg.type == CommsProtocol.types['terminate']):
            print('terminating... ')
            # update execute flag
            self.execute = False
            self.initialized = False