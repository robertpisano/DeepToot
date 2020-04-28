import numpy as np
from DeepToot.src.entities.physics.base_3d_vector import Base3DVector

class BaseState():
    def __init__(self, position, velocity, orientation, ang_vel, time):
        self.velocity = velocity
        self.position = position
        self.orientation = orientation
        self.time = time
        self.ang_vel = ang_vel
    
    def velocity(self): return self.velocity
        
    def position(self): return self.position

    def orientation(self): return self.orientation
    
    def ang_vel(self): return self.ang_vel
    
    def time(self): return time


class BaseStateBuilder():
    velocity = Base3DVector()
    position = Base3DVector()
    orientation = Base3DVector()
    ang_vel = Base3DVector()
    time = float(0.0)
    
    def velocity(self, velocity):
        self.velocity = Base3DVector(numpy_array=velocity)
        return self

    def position(self, position):
        self.position = Base3DVector(numpy_array=position)
        return self
        
    def orientation(self, orientation):
        self.orientation = Base3DVector(numpy_array=orientation)
        return self
        
    def ang_vel(self, ang_vel):
        self.ang_vel = Base3DVector(numpy_array=ang_vel)
        return self
        
    def time(self, time):
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



