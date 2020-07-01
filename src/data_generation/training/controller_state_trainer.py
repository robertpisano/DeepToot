from DeepToot.src.data_generation.simple_controller_state_generator import SimpleControllerStateGenerator
import DeepToot.src.data_generation.gekko_controls_generator
import numpy as np

class ControllerStateTrainer:
    None
    # position = range(-100, 100)
    # velocity = range(-100, 100)

    # for x in poisiton:
    #     for y in velocity

if __name__ == "__main__":
    amax = 900 #maximum acceleration
    controls = np.linspace(-1, 1, 21) #controls vector

    time = np.linspace(0, 10, 11) #time vector
    vi = 0.0 #intial velocity
    trajectories = [] #list to append trajectories
    controls_list = []
    for u in controls:
        buffer = u*amax*time + vi
        trajectories.append(buffer)
        controls_list.append(np.repeat(u, 11))

    inbuff = []
    outbuff = []
    
    for j, traj in enumerate(trajectories):
        for i, state in enumerate(traj):
            if j==0 and i==0:
                indata = np.array([traj[i], traj[i+1]], dtype = np.float)
                indata = np.expand_dims(indata, axis=1)
                outdata = np.array([controls_list[j][i]])
                outdata = np.expand_dims(outdata, axis=1)
            elif i == len(traj)-1:
                break
            else:
                indata = np.append(indata, np.expand_dims(np.array([traj[i], traj[i+1]]), axis=1), axis = 1)
                buff = np.array([controls_list[j][i]])
                buff = np.expand_dims(buff, axis=1)
                outdata = np.append(outdata, buff, axis = 1)
    
    simp = SimpleControllerStateGenerator(11)
    simp.model.fit(indata.T, outdata.T, epochs=1000, batch_size=100)
    t1 = trajectories[3][5]
    t2 = trajectories[3][6]
    u_model = controls_list[3][5]
    intest = np.array([t1, t2], dtype=np.float)
    intest = np.expand_dims(intest, axis=1)
    nnout = simp.model.predict(intest.T)

    print(nnout)
    print(u_model)
    # print(u)
    # print(trajectories)
