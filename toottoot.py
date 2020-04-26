
print('workeds')
import sys
print(sys.argv)

#Commands for CLI

try:
    # Download .replay files
    if sys.argv[1] == 'download':
        from RLBOT.src.GameData import BallChasingInteractor
        BallChasingInteractor().download_all()

    # Convert .replay --> rawdataframes and save
    if sys.argv[1] == 'convert':
        from RLBOT.src.ParserLibrary import ReplayConverter
        rc = ReplayConverter()
        rc.convert_and_save_replays() 

    # Find training data by using hits, takes two args inputlength and outputlength
    if sys.argv[1] == 'generate_basic':
        import RLBOT.src.NeuralNetworkDataGenerator as nndg
        try:
            input_length = int(sys.argv[2]) # First argument passed in is input length for LSTM
            output_length = int(sys.argv[3]) # Second argument passed is output length for LSTM
            inBatch, outBatch = nndg.generate_nn_data_from_saved_data_frames(input_length, output_length) # Run through all replays in a folder you will select and return batch
            # TODO: save inBatch, outBatch in some class called TrainingData or something to be loaded later for either filtering or NN training
            a = inBatch.to_numpy()
            print(a.shape)
        except Exception as e:
            print(e)
            try:
                sys.argv[2]
            except:
                print('generate_basic takes two arguments, inLength, outLength')
            # print('Input arguments for generate_basic cannot be cast to an integer')
        import code
        code.interact(local=locals())
    
    # Plot generated_data
    if sys.argv[1] == 'plot':
        try:
            from RLBOT.src.NeuralNetworkDataGenerator import DataAnalyzer
            analyzer = DataAnalyzer()
            analyzer.full_analysis()
        except Exception as e:
            print(e)
            

    # Data Filtering Standardization

    # Train Network, validate and save

    # Test Saved Network (plot NN trajectory and who hits ball first), Add some analysis

except Exception as e:
    print(e)


# Strictly so i can run breakpoints for debugging. while in CLI it doesn't seem to run breakpoints
if __name__ == "__main__":
    try:
        from RLBOT.src.NeuralNetworkDataGenerator import DataAnalyzer
        analyzer = DataAnalyzer()
        analyzer.full_analysis()
    except Exception as e:
        print(e)