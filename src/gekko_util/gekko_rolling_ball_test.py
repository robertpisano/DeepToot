from DeepToot.src.gekko_util.gekko_util import *
import DeepToot.src.gekko_util.gekko_plotting_util as plot
from rlbot.utils.game_state_util import CarState, BallState, Vector3, Physics, Rotator
import numpy as np

si = Vector3(500, 0, 0)
vi = Vector3(0, 0, 0)
v_mag = 1000
ri = Rotator(roll=0,pitch=0,yaw=0)
wi = Vector3(0,0,0)

bs = Vector3(0, 1000, 0)
bv = Vector3(0, -1000, 0)
br = Rotator()
bw = Vector3()

sf = Vector3(0, 0, 0)
vf = 0
rf = Rotator(0,0,0)
wf = Vector3()

initial = Conditions(s=si, v_mag=v_mag, r=ri, w=wi, bs=bs, bv=bv, br=br, bw=bw)
final = Conditions(s=sf, v_mag=vf, r=rf)

o = RollingOptimizer(initial, final)
# o.set_objectives(final)
o.open_folder()
o.solve()

plot.plot_car(o)
plot.plot_ball(o)
plot.show()

print(o)
