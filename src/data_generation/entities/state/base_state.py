import numpy as np
from DeepToot.src.data_generation.entities.physics.base_3d_vector import Base3DVector

class BaseState():
    def __init__(self, position: Base3DVector, velocity: Base3DVector, orientation: Base3DVector, ang_vel: Base3DVector, time):
        self.velocity = velocity
        self.position = position
        self.orientation = orientation
        self.time = time
        self.ang_vel = ang_vel


class BaseStateBuilder():
    velocity = Base3DVector(numpy_array=np.zeros(3, dtype=np.float))
    position = Base3DVector(numpy_array=np.zeros(3, dtype=np.float))
    orientation = Base3DVector(numpy_array=np.zeros(3, dtype=np.float))
    ang_vel = Base3DVector(numpy_array=np.zeros(3, dtype=np.float))
    time = float(0.0)

    def __init__(self):
        None
    
    def set_velocity(self, velocity):
        self.velocity = Base3DVector(numpy_array=velocity)
        return self

    def set_position(self, position):
        self.position = Base3DVector(numpy_array=position)
        return self
        
    def set_orientation(self, orientation):
        self.orientation = Base3DVector(numpy_array=orientation)
        return self
        
    def set_ang_vel(self, ang_vel):
        self.ang_vel = Base3DVector(numpy_array=ang_vel)
        return self
        
    def set_time(self, time):
        self.time = time
        return self
        
    def build(self):
        return BaseState(
            velocity=self.velocity,
            position=self.position,
            orientation=self.orientation,
            ang_vel=self.ang_vel,
            time=self.time
        )



