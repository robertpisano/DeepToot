from DeepToot.src.gekko_util.gekko_util import *
import DeepToot.src.gekko_util.gekko_plotting_util as plot
from rlbot.utils.game_state_util import CarState, BallState, Vector3, Physics, Rotator
import numpy as np

si = Vector3(750, 750, 0)
vi = Vector3(0, 0, 0)
v_mag = 100
ri = Rotator(roll=0,pitch=0,yaw=0)
wi = Vector3(0,0,0)

bs = Vector3(1000, 1000, 0)
bv = Vector3(-1000, -1000, 0)
br = Rotator()
bw = Vector3()

sf = Vector3(0, 0, 0)
vf = 0
rf = Rotator(0,0,0)
wf = Vector3()

initial = Conditions(s=si, v_mag=v_mag, r=ri, w=wi, bs=bs, bv=bv, br=br, bw=bw)
final = Conditions(s=sf, v_mag=vf, r=rf)

o = Optimizer(initial, final)
# o.set_objectives(final)
o.open_folder()
o.solve()

plot.plot_car(o)
plot.show()

print(o)
