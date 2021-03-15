from DeepToot.src.meta_data_objects.AbstractMetaDataObjectFactory import AbstractMetaDataObjectFactory
from DeepToot.src.meta_data_objects.SimulationDataObject import SimulationDataObject
import pickle
# from ruamel import yaml
import json

class SerializationFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_type(name, dict={}):
        return type(name, (), dict)

    # Convert simuilation data object into a proper 2D list for sending over ethernet
    @staticmethod
    def listify(t: SimulationDataObject):
        dc = t.drivingController
        ac = t.aerialController
        brain = t.brain
        ic = t.initialConditions
        dcList = ['dc', (dc.params, dc.miscOptions)]
        acList = ['ac', (ac.params, ac.miscOptions)]
        brainList = ['brain', (brain.pre_calculate, brain.params, brain.miscOptions)]
        icList = ['ic', (ic.params, ic.miscOptions)]

        nestedList = [dcList, acList, brainList, icList]
        return pickle.dumps(nestedList)
        
    # Convert the 2D list back into a simulation data object
    @staticmethod
    def delistify(l):
        fh = pickle.loads(l)
        # Create memebers of the Simulation Data OBject
        dc = AbstractMetaDataObjectFactory.create('DrivingController')
        ac = AbstractMetaDataObjectFactory.create('AerialController')
        brain = AbstractMetaDataObjectFactory.create('MinimumTimeToBall')
        ic = AbstractMetaDataObjectFactory.create('InitialConditionsGekko')
        # Set values of each class from the input argument list
        dc.params = fh[0][1][0]
        dc.miscOptions = fh[0][1][1]
        ac.params = fh[1][1][0]
        ac.miscOptions = fh[1][1][1]
        brain.pre_calculate = fh[2][1][0]
        brain.params = fh[2][1][1]
        brain.miscOptions = fh[2][1][2]
        ic.params = fh[3][1][0]
        ic.miscOptions = fh[3][1][1]

        print('delistify ic.params', ic.params)

        # a = cls.create_type(name, dict)()
        a = SimulationDataObject(dc, ac, brain, ic)
        # print(a)
        return a



if __name__ == "main":
    sim = SimulationDataObject(None, None, None, None)
    SerializationFactory.listify(sim)