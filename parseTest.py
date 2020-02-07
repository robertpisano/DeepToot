
try:
    from ParserLibrary import ReplayConverter, factory
    rc = factory()
    if(rc.saveStateExists == False):
        rc.convert_and_save_replays()
        rc.save_current_state()
    cc = ReplayConverter.get_controls_from_replay(rc.game)
    p = cc.players
    # gameWithControls = ReplayConverter.append_control_data(rc.gameDataList[0], cc)
except Exception as e:
    # For debugging
    print(e)
import code
code.interact(local=locals())
