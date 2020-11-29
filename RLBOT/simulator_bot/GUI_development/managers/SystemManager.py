from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.ControllerFactory import ControllerFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.Brain import Brain
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.BrainFactory import BrainFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.AbstractMetaDataObjectFactory import AbstractMetaDataObjectFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.SimulationDataObject import SimulationDataObject
# from rlbot.messages.flag.RigidBodyTick import RigidBodyTick
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder, GameTrajectory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.SerializationFactory import SerializationFactory

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

from threading import Thread
import select
import socket
import pickle
import dill
import yaml

class SystemManager(Thread):
    # drivingController: Controller
    # aerialController: Controller
    # brain: Brain
    # initialCondition: InitialConditions
    # simulationTrajectory: GameTrajectory
    # dataObject: SimulationDataObject

    def __init__(self, index, ip: "", port: int):
        # Initialize socket reading thread
        Thread.__init__(self)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #Setup TCP Socket
        self.socket.bind((ip, port))
        self.socket.listen(5)
        self.setDaemon(True)
        self.start()

        self.started = False
        self.rlbot_index = index

        self.drivingController = Controller()
        self.aerialController = Controller()
        self.brain = Brain()
        self.initialCondition = InitialConditions()
        self.simulationTrajectory = GameTrajectoryBuilder(0).build()
        self.dataObject = SimulationDataObject(Controller(), Controller(), Brain(), InitialConditions())

    def update(self, dataObject: SimulationDataObject):
        self.drivingController = dataObject.drivingController
        self.aerialController = dataObject.aerialController
        self.brain = dataObject.brain
        self.initialConditions = dataObject.initialConditions
        self.simulationTrajectory = dataObject.simulationTrajectory
        # Reset started flag
        self.started=False

    def run_bot(self, packet):
        # If simulation hasn't started, do initialization routine on brain and controller
        if(not self.started):
            # Convert frames from rigidbodytick to seconds (120 ticks / second)
            t0 = packet.players[self.rlbot_index].state.frame / 120
            self.drivingController.update_t0(t0)

            #TODO: Update environment

            self.started = True
        
        try:
            # Always run this logic
            trajectory = self.brain.calculate(packet)
            controllerstate = self.drivingController.calculate_control(trajectory)

            #TODO: Append data to simulationTrajectory
        except:
            controllerstate = SimpleControllerState()

        return controllerstate
    

    # SOCKET FUNCTIONS

    # Get data from socket in loop in case serialization is > buffer size
    def run(self):
        """Run the socket "server"
        """        

        while True:
            try:
                client, addr = self.socket.accept()
        
                ready = select.select([client,],[], [],2)
                if ready[0]:
                    print('ready[0] true')
                    dataObject = self.recvall(client)
                    self.dataObject = dataObject
                    print("got packet", self.dataObject.drivingController.params)
                    
            except socket.error as error:#, msg:
                print("Socket error! %s" % error.strerror)
                pass
            
        # shutdown the socket
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
    
        self.socket.close()
        
    def recvall(self, sock):
        BUFF_SIZE = 4096 # 4 KiB
        data = b''
        while True:

            part = sock.recv(BUFF_SIZE)
            data += part
            if len(part) <= 0:
                # either 0 or end of data
                break
            print('receiving buffer from socket length: ' + str(len(part)))

        received_class = SerializationFactory.delistify(data)
        return received_class  