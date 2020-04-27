from RLBOT.src.entities.base_state import BaseState
from RLBOT.src.entities.base_state import BaseStateBuilder
from rlbot.src.entities.physics.base_3d_vector import Base3DVector

class CarState(BaseState):
    def __init(self, position, velocity, orientation, ang_vel, time, hit_box, is_demolished, has_wheel_contact, is_super_sonic, has_jumped, has_double_jumped, boost_amount):
        super().__init__(position=position,
                        velocity=velocity,
                        orientation=orientation,
                        ang_vel=ang_vel,
                        time=time)
        self.hit_box = hit_box
        self.is_demolished = is_demolished
        self.has_wheel_contact = has_wheel_contact
        self.is_super_sonic = is_super_sonic
        self.has_jumped = has_jumped
        self.has_double_jumped = has_double_jumped
        self.boost_amount = boost_amount
    
    def hitBox(self):
        return self.hit_box
    
    def isDemolished(self):
        return self.is_demolished

    def hasWheelContact(self):
        return self.has_wheel_contact
    
    def isSuperSonic(self):
        return self.is_super_sonic

    def hasJumped(self):
        return self.is_super_sonic
    
    def hasDoubleJumped(self):
        return self.double_jumped
    
    def boostAmount(self):
        return self.boost_amount

class BallStateBuilder(BaseStateBuilder):
    hit_box = Base3DVector()
    is_demolished = bool(0)
    has_wheel_contact = bool(0)
    is_super_sonic = bool(0)
    has_jumped = bool(0)
    has_double_jumped = bool(0)
    boost_amount = int(0)

    def hitBox(self, hit_box):
        self.hitbox = Base3DVector(numpy_array=hit_box)
        return self

    def isDemolished(self, is_demolished):
        self.is_demolished = is_demolished
        return self

    def hasWheelContact(self, has_wheel_contact):
        self.has_wheel_contact = has_wheel_contact
        return self
    
    def isSuperSonic(self, is_super_sonic):
        self.is_super_sonic=is_super_sonic
        return self

    def hasJumped(self, has_jumped):
        self.has_jumped = has_jumped
        return self
    
    def hasDoubleJumped(self, has_double_jumped):
        self.has_double_jumped = has_double_jumped
        return self
    
    def boostAmount(self, boost_amount):
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