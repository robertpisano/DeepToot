from DeepToot.src.data_generation.entities.state.base_state import BaseState
from DeepToot.src.data_generation.entities.state.ball_state import BallState
from DeepToot.src.data_generation.entities.state.car_state import CarState
from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory
from DeepToot.src.data_generation.entities.physics.game_trajectory import GameTrajectory
from DeepToot.src.data_generation.entities.exceptions.build_failed_exception import BuildFailedException

class GameTrajectoryBuilder():
    ball_queue = []
    bot_queue = []
    opp_queue = []
    MAX_QUEUE_LENGTH = 0
    are_unneeded_queues_zero_filled = False

    def __init__(self, length: int):
        self.MAX_QUEUE_LENGTH = length

    def add_ball_state(self, state: BallState):
        """add ball state to queue
            trims queue if > MAX_QUEUE_LENGTH

        Args:
            state (BallState): [description]
        """        
        self.ball_queue = self._add_state_to_queue(state, self.ball_queue)
        
    def add_bot_state(self, state: CarState):
        """add bot state to queue
            trims queue if > MAX_QUEUE_LENGTH

        Args:
            state (CarState): [description]
        """        
        self.bot_queue = self._add_state_to_queue(state, self.bot_queue)
        
    def add_opp_state(self, state: CarState):
        """add opponent state to queue
            trims queue if > MAX_QUEUE_LENGTH
            
        Args:
            state (CarState): [description]
        """        
        self.opp_queue = self._add_state_to_queue(state, self.opp_queue)

    def add_game_state(self, ball_state: BallState, bot_state: CarState, opp_state: CarState):
        """add all states to respective queues

        Args:
            ball_state (BallState): [description]
            bot_state (CarState): [description]
            opp_state (CarState): [description]
        """        
        self.add_ball_state(ball_state)
        self.add_bot_state(bot_state)
        self.add_opp_state(opp_state)

    def zero_fill_unneeded_queues(self):
        self.are_unneeded_queues_zero_filled = True

    def build(self):
        """build a GameTrajectory

        Args:
            length (int): injected length from neural network arch
        """        
        if((len(self.ball_queue) == len(self.bot_queue)) and (len(self.bot_queue) == len(self.opp_queue)) or self.are_unneeded_queues_zero_filled):
            ball_trajectory = Trajectory(state_array = self.ball_queue)
            bot_trajectory = Trajectory(state_array = self.bot_queue)
            opp_trajectory = Trajectory(state_array = self.opp_queue)
        else: 
            raise BuildFailedException("Could not build a trajectory because the queue lengths are not the same")
        
        return GameTrajectory(ball_trajectory = ball_trajectory, 
                                bot_trajectory = bot_trajectory, 
                                opp_trajectory = opp_trajectory)

    def _add_state_to_queue(self, state: BaseState, queue: list):
        """adds an injected state into the injected queue
            checks length, if not full of states, will fill the entire queue with 
            the state injected

        Args:
            state (BaseState): injected state (can be BallState or CarState)
            queue (list): the respective queue related to the injected state

        Returns:
            [list]: the newly modified queue
        """        
        if(len(queue) == 0): 
           queue = [state] * self.MAX_QUEUE_LENGTH # create a queue filled with initial state
        else:
            queue.append(state)
        if(len(queue) > self.MAX_QUEUE_LENGTH): queue.pop(0) # remove first element from list
        
        return queue

# Run testing code here
if __name__ == "__main__":
    t = GameTrajectoryBuilder(20)
    t.add_ball_state(BallState(0,0,0,0,0))

