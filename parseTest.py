from ParserLibrary import ReplayConverter
import glob

rc = ReplayConverter()
print(rc.file_path)
print(rc.replays_path)
rc.convert_and_save_replays()

# replays = glob.glob(rc.file_path + '/*.replay')
# print(replays)

print(rc.replays_path)
print(rc.output_path)
# self.file_path + '/' + str(i) + '.json'
