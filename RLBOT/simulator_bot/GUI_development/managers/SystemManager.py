from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.ControllerFactory import ControllerFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.Brain import Brain
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.MinimumTimeToBallBrain import MinimumTimeToBallBrain
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.BrainFactory import BrainFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditions import InitialConditions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.AbstractMetaDataObjectFactory import AbstractMetaDataObjectFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.SimulationDataObject import SimulationDataObject
# from rlbot.messages.flag.RigidBodyTick import RigidBodyTick
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder, GameTrajectory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.SerializationFactory import SerializationFactory

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

from threading import Thread, Lock
import select
import socket
import pickle
import dill
import yaml
import time

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

        # Setup a thread lock for syncronization
        global lock
        lock = Lock()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #Setup TCP Socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ip, port))
        self.socket.listen(5)
        self.setDaemon(True)
        self.start()

        global started, first_received, flag_changed
        flag_changed = True
        started = False
        self.rlbot_index = index
        first_received = False # Make sure to not do things till we've received data from socket
        

        self.drivingController = Controller()
        self.aerialController = Controller()
        self.brain = MinimumTimeToBallBrain()
        self.initialCondition = InitialConditionsGekko()
        self.simulationTrajectory = GameTrajectoryBuilder(0).build()
        self.dataObject = SimulationDataObject(Controller(), Controller(), Brain(), InitialConditionsGekko())

    def update(self, dataObject: SimulationDataObject):
        self.drivingController = dataObject.drivingController
        self.aerialController = dataObject.aerialController
        self.brain = dataObject.brain
        self.initialConditions = dataObject.initialConditions
        self.simulationTrajectory = dataObject.simulationTrajectory
        # Reset started flag
        self.started=False

    def run_bot(self, packet):
        global started, first_received, flag_changed
        # If simulation hasn't started, do initialization routine on brain and controller
        if((not started) and first_received):
            try:
                print('Simulation not started, starting...')
                flag_changed = True
                started = True

                print('out of lock while')

                # time.sleep(0.5)
                # Convert frames from rigidbodytick to seconds (120 ticks / second)
                t0 = packet.players[self.rlbot_index].state.frame / 120
                self.drivingController.update_t0(t0)
                self.brain.calculate(self.dataObject.initialConditions)
                print(self.brain.opt.car)
            except:
                controllerstate =  SimpleControllerState()


            #TODO: Update environment
        try:
            # Always run this logic
            if(flag_changed):
                print('started: ', started, ' | first recv: ', first_received)
                flag_changed = False

            # trajectory = self.brain.calculate()
            # controllerstate = self.drivingController.calculate_control(trajectory)
            controllerstate = SimpleControllerState()
            pass
            #TODO: Append data to simulationTrajectory
        except:
            controllerstate = SimpleControllerState()
        return controllerstate
    

    # SOCKET FUNCTIONS

    # Get data from socket in loop in case serialization is > buffer size
    def run(self):
        global started, first_received
        """Run the socket "server"
        """        

        while True:
            try:
                client, addr = self.socket.accept()
        
                ready = select.select([client,],[], [],2)
                if ready[0]:
                    print('ready[0] true')
                    dataObject = self.recvall(client)

                    # Lock algorithm to make sure bot sees the change on self.started
                    self.dataObject = dataObject
                    self.update(dataObject)
                    started = False
                    first_received = True
                    print("got packet", self.dataObject.initialConditions.params['sxi'])
                    
            except socket.error as error:#, msg:
                print("Socket error! %s" % error.strerror)
                pass
            
        # shutdown the socket
        try:
            print("Shutting down socket...")
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            print("Force closed socket")
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