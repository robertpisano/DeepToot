
try:
    from ParserLibrary import ReplayConverter, factory
    rc = factory()
    if(rc.saveStateExists == False):
        rc.convert_and_save_replays()
        rc.save_current_state()
    controls = ReplayConverter.get_controls_from_replay(rc.game)
    print(controls)
except Exception as e:
    # For debugging
    print(e)
import code
code.interact(local=locals())
