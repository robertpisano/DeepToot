from DeepToot.src.meta_data_objects.controllers.ControllerFactory import ControllerFactory, ControllerSchema
from DeepToot.src.meta_data_objects.SimulationDataObject import SimulationDataObject, SimulationDataObjectSchema
from DeepToot.src.meta_data_objects.brains.BrainFactory import BrainFactory, BrainSchema
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory, InitialConditionsSchema
if __name__ == "__main__":
    ss = SimulationDataObjectSchema()
    dc = ControllerFactory.create('DrivingController')
    ac = ControllerFactory.create('AerialController')
    b = BrainFactory.create('MinimumTimeToBall')
    ic = InitialConditionsFactory.create('InitialConditionsGekko')
    sim = SimulationDataObject(dc, ac, b=b, ic=ic)

    serialized = ss.dump(sim)
    print(serialized)
    deserialized = ss.load(serialized)
    print('debug')