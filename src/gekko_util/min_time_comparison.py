from DeepToot.src.gekko_util.gekko_util import *

from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import InitialConditionsGekko, InitialConditionsGekko2

class MinTimeComparison(GEKKO):
    def __init__(self, ic1: Conditions, ic2: Conditions, fc: Conditions, num_nodes = 11):
        super().__init__(remote=False)
        self.initialConditions1 = ic1
        self.initialConditions2 = ic2
        self.finalConditions = fc
        self.c1 = GekkoDrivingCarState(suffix='1')
        self.c2 = GekkoDrivingCarState(suffix='2')
        self.ball = GekkoBallSplineState()
        
        # Setting Options first fixed no solution bug!!!
        self.set_options()



        # Time variables for simulation
        # tf: final time variable to optimize for minimizing final time
        # time: normalized time vector with num_nodes discretations
        self.tf1 = self.FV(value = 1, lb=1, ub=10, name='tf1')
        self.tf1.STATUS = 1 # Let optimizer adjust tf
        self.tf2 = self.FV(value = 1, lb=1, ub=10, name='tf2')
        self.tf2.STATUS = 1 # Let optimizer adjust tf
        self.time = np.linspace(0, 1, num_nodes)
        # Solver time for time based c splines
        self.solver_time1 = self.Var(value=0)
        self.Equation(self.solver_time1.dt()/self.tf1 == 1)
        # Solver time for time based c splines
        self.solver_time2 = self.Var(value=0)
        self.Equation(self.solver_time2.dt()/self.tf2 == 1)
        
        self.p_d = np.zeros(num_nodes)
        self.p_d[-1] = 1.0
        self.final = self.Param(value = self.p_d, name='final')


        self.set_all_equations()
        self.set_objectives(fc)

    def set_options(self):
        self.options.NODES=3
        self.options.SOLVER=3
        self.options.IMODE=6
        self.options.MAX_ITER=800
        self.options.MV_TYPE = 0
        # self.options.DIAGLEVEL=1

    def set_all_equations(self):
        self = self.c1.inject_variables(self)
        self = self.c1.inject_dynamics(self)
        self = self.c2.inject_variables(self)
        self = self.c2.inject_dynamics(self)
        self = self.ball.inject_variables(self)

    def set_objectives(self, fc):
        self = GekkoMinimumTimeComparison.inject_objectives(self)


if __name__ == "__main__":
    import DeepToot.src.gekko_util.gekko_plotting_util as plot
    initial1 = Conditions.build_initial_from_initial_conditions_object(InitialConditionsGekko())
    initial2 = Conditions.build_initial_from_initial_conditions_object(InitialConditionsGekko2())
    final = Conditions.build_final_from_initial_conditions_object(InitialConditionsGekko())

    o = MinTimeComparison(ic1=initial1, ic2=initial2, fc=final)
    # o.set_objectives(final)
    o.open_folder()
    o.solve()

    print('tf1: ', o.tf1.value[0], ' | tf2: ', o.tf2.value[0])

    plot.plot_cars(o)
    plot.plot_balls(o)
    plot.show()

    print(o)