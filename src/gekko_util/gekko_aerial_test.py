from DeepToot.src.gekko_util.gekko_util import AerialOptimizer, AerialConditions
import DeepToot.src.gekko_util.gekko_plotting_util as plot
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsGekko import AerialInitialConditionsGekko
from rlbot.utils.game_state_util import CarState, BallState, Vector3, Physics, Rotator
import numpy as np
from DeepToot.src.data_generation.game_trajectory_builder import GameTrajectoryBuilder, GameTrajectory
from DeepToot.src.data_generation.state_transformer import StateTransformer
import time

initial = AerialConditions.build_initial_from_initial_conditions_object(AerialInitialConditionsGekko())
final = AerialConditions.build_final_from_initial_conditions_object(AerialInitialConditionsGekko())

o = AerialOptimizer(initial, final, num_nodes=7)
# o.set_objectives(final)
o.open_folder()
o.solve(disp=True)

time.sleep(0.1)

tb = GameTrajectoryBuilder(len(o.time))
# Save solved trajectory
for i, s in enumerate(o.time):
    state = StateTransformer.from_gekko_solver_to_car_state(o, i)
    bstate = StateTransformer.from_gekko_solver_to_ball_state(o, i)
    tb.add_bot_state(state)
    tb.add_ball_state(bstate)
tb.zero_fill_unneeded_queues()
tb.build()

plot.plot_aerial_car(o)
plot.plot_aerial_ball(o)
plot.show()

print(o)
