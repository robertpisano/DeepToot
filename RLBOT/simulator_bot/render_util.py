from rlbot.agents.base_agent import BaseAgent


def basic_render(bot, packet):
    # start rendering
    bot.renderer.begin_rendering()

    # coord system render
    render_intertial_coordinate_system(bot)

    # bot reference frame coordinate system
    render_reference_coordinate_system(bot, packet)

    # finish rendering
    bot.renderer.end_rendering()

def render_intertial_coordinate_system(bot: BaseAgent):

    
    # define some important coordinate features
    origin = [0,0,0]
    x = [500, 0, 0]
    y = [0, 500, 0]
    z = [0, 0, 500]

    # draw x axis in red
    bot.renderer.draw_line_3d(origin, x, bot.renderer.red())

    # draw y axis in green
    bot.renderer.draw_line_3d(origin, y, bot.renderer.green())

    # draw z axis in blue
    bot.renderer.draw_line_3d(origin, z, bot.renderer.blue())

def render_reference_coordinate_system(bot, packet):
    None