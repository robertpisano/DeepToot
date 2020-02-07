from ParserLibrary import ReplayConverter
import carball

json_file = carball.decompile_replay("D:/Documents/RL Replays/69DA1AB64F613BE0CD5F3A94EDD7BC24.replay",
                    output_path = "D:/Documents/RL Replays",
                    overwrite=True)
self.game.initialize(loaded_json = json_file)
