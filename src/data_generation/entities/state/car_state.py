from DeepToot.src.data_generation.entities.state.base_state import BaseState
from DeepToot.src.data_generation.entities.state.base_state import BaseStateBuilder
from DeepToot.src.data_generation.entities.physics.base_3d_vector import Base3DVector
import numpy as np

class CarState(BaseState):
    def __init__(self, position, velocity, orientation, ang_vel, time, hit_box, is_demolished, has_wheel_contact, is_super_sonic, has_jumped, has_double_jumped, boost_amount):
        self.position=position
        self.velocity=velocity
        self.orientation=orientation
        self.ang_vel=ang_vel
        self.time=time
        self.hit_box = hit_box
        self.is_demolished = is_demolished
        self.has_wheel_contact = has_wheel_contact
        self.is_super_sonic = is_super_sonic
        self.has_jumped = has_jumped
        self.has_double_jumped = has_double_jumped
        self.boost_amount = boost_amount

class CarStateBuilder(BaseStateBuilder):
    hit_box = Base3DVector(numpy_array=np.zeros(3))
    is_demolished = bool(0)
    has_wheel_contact = bool(0)
    is_super_sonic = bool(0)
    has_jumped = bool(0)
    has_double_jumped = bool(0)
    boost_amount = int(0)

    def __init__(self):
        None

    def set_hit_box(self, hit_box):
        self.hitbox = Base3DVector(numpy_array=hit_box)
        return self

    def set_is_demolished(self, is_demolished):
        self.is_demolished = is_demolished
        return self

    def set_has_wheel_contact(self, has_wheel_contact):
        self.has_wheel_contact = has_wheel_contact
        return self
    
    def set_is_super_sonic(self, is_super_sonic):
        self.is_super_sonic=is_super_sonic
        return self

    def set_has_jumped(self, has_jumped):
        self.has_jumped = has_jumped
        return self
    
    def set_has_double_jumped(self, has_double_jumped):
        self.has_double_jumped = has_double_jumped
        return self
    
    def set_boost_amount(self, boost_amount):
        self.boost_amount = boost_amount
        return self

    def build(self):
        return CarState(velocity=self.velocity,
            position=self.position,
            orientation=self.orientation,
            ang_vel=self.ang_vel,
            time=self.time,
            hit_box=self.hit_box,
            is_demolished=self.is_demolished,
            has_wheel_contact=self.has_wheel_contact,
            is_super_sonic = self.is_super_sonic,
            has_jumped = self.has_jumped,
            has_double_jumped = self.has_double_jumped,
            boost_amount = self.boost_amount
        )