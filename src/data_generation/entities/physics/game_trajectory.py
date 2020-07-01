from DeepToot.src.data_generation.entities.physics.trajectory import Trajectory
#this is what gets fed into the lstm neural net
class GameTrajectory():
    OPPONENT_TRAJECTORY = None
    BOT_TRAJECTORY = None
    BALL_TRAJECTORY = None

    def __init__(self, opp_trajectory:Trajectory, bot_trajectory:Trajectory, ball_trajectory:Trajectory):
        self.OPPONENT_TRAJECTORY = opp_trajectory
        self.BOT_TRAJECTORY = bot_trajectory
        self.BALL_TRAJECTORY = ball_trajectory