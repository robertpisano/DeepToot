# This file has some classes that will make running a replay scenario simple
# InitialGameState will be the initial state of the game to start the scenario
# PlayerControls will be the calculated inputs for that specific scenario subset
# These together will give an RLBot agent the ability to test the inputs then
# compare the raw replay data to the new generated data to compare and see how precise the input calulation is

from NeuralNetworkDataGenerator import PlayerState, BallState
from hitFinder import RawReplayData
from pandas import DataFrame

# Initial game state 
class InitialGameState():

    def __init__(self, p1: PlayerState, p2: PlayerState, ballstate: BallState):
        orangeCar = p1
        blueCar = p2
        ball = ballstate

# All player controls
class PlayerControls():
    def __init__(self, players, frameStart, scenarioLength):
        self.frameStart = frameStart
        self.scenarioLength = scenarioLength
        if(players[0].is_orange):
            self.orangeControls = players[0].controls.loc[frameStart:frameStart+scenarioLength]
            self.blueControls = players[1].controls.loc[frameStart:frameStart+scenarioLength]
        else:
            self.orangeControls = players[1].controls.loc[frameStart:frameStart+scenarioLength]
            self.blueControls = players[0].controls.loc[frameStart:frameStart+scenarioLength]

class ScenarioCreator():
    def __init__(self):
        None
    # Get initial state, and get controls for each player, return for use with RLBot
    def   (self, rrd: RawReplayData, frameStart: int, scenarioLength: int):
        self.frameStart = frameStart
        self.scenarioLength = scenarioLength
        object_keys = rrd.game.columns.levels[0] # Get the objects in game
        data_keys = rrd.game.columns.levels[1] # Get the data keys from game
        game_data_at_frame = rrd.game.loc[frameStart] # get the dataframe at frameStart
        p1 = PlayerState(game_data_at_frame.loc[object_keys[0]]) # Own state
        p2 = PlayerState(game_data_at_frame.loc[object_keys[1]]) # Opponent state
        ball = BallState(game_data_at_frame.loc[object_keys[2]]) # Ball State
        # Get the initial game state to set in RLBot
        self.initalState = InitialGameState(p1, p2, ball)
        # Get control data frame subset, send to PlayerControls class, save in self.
        self.playerControls = PlayerControls(rrd.players, frameStart, scenarioLength)
        # Get the rest of the scenario data for the same subset for comparison etc...
        self.scenarioData = rrd.game.loc[frameStart:frameStart+scenarioLength]

if __name__ == '__main__':
    rrd = RawReplayData.load()
    print(rrd)
    sc = ScenarioCreator()
    sc.get_scenario_data(rrd, 100, 60)
    print('end')