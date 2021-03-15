from DeepToot.src.meta_data_objects.brains.Brain import Brain
from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory
from DeepToot.src.gekko_util.gekko_util import Conditions, Optimizer
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko


class MinimumTimeToBallBrain(Brain):
    name = 'MinimumTimeToBall'
    params = {'utime': 1, 'ufuel': 0, 'guided':True}
    miscOptions = {'num_nodes':21}
    pre_calculate: bool
    state_trajectory: Trajectory
    control_trajectory: list

    def __init__(self):
        self.pre_calculate = True
        self.params = {'utime': 1, 'ufuel': 0, 'guided':True}
        self.miscOptions = {'num_nodes':21}


    def calculate(self, ic: InitialConditionsGekko):
        try:
            conditions = Conditions.build_initial_from_initial_conditions_object(ic)
            print('init condit succeed')
            conditions_final = Conditions.build_final_from_initial_conditions_object(ic)
            print('final condit succeed')
            num_nodes = int(self.miscOptions['num_nodes'])
            self.opt = Optimizer(conditions, conditions_final, num_nodes)
        
            print('Starting optimziation solve')
            print('Debug stuff:')
            print('sx: ', self.opt.initialConditions.s.x)
            print('sy: ', self.opt.initialConditions.s.y)
            print('vi: ', self.opt.initialConditions.v_mag)
            self.opt.open_folder()
            self.opt.solve()
        except Exception as e:
            print(e)
            print('optimization solver failed')