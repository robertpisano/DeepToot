class Trajectory():
    def __init__(self, state_array=[]):
        self.states=state_array 

    def state_at_time(self, time=0.0):
        None
    
    def state_at_index(self, index):
        return self.states[index]
