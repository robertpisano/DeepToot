DeepToot- toot the ball deep into the goal3


simulator_bot:
simulator_bot.py and simulator_utilities.py are a cleaned up version of the "Lotus" bot I had previously.
I decided to make it much simpler. There are a lot of functions still in the GUI() class, which means they'll be another code purge in the future as I want to make this more user friendly, and modular. But to do so I need to reorganize the way I have decided to run the optimization algorithm, and before I clean the code I have more simulation prototyping to do. (I need to add "impulse" usage aka the jump/dash and double jump to the model, as well as make the model able to model a trajectory that drives and then goes into an aerial. I need to figure out how to do all this mathematically first before I can make a nice modular code base to make optimizer options and other features part of the AerialOptimizer() class)

To run, run simulator_bot.py.
When running the file directly (instead of through RLBot), the get_output() function is called in a while loop
just how it would be when RLBot is running. Except, I believe get_output() gets called much faster in my case.
Anyways this is just so I can develop the GUI without needing to run RLBot.

The GUI works as follows:
1. A small window pops up with 5 buttons. Currently the top two buttons do nothing yet
2. Press "show sim params" button to open up a new window
3. You can load a saved sim_params, or edit the entry boxes yourself to make a sim_params set.
4. You can save the current values by using the "save sim params" button
5. Once loaded, go back to the main window (with the 5 buttons)
6. Click "generate sim state" which will take the sim_params, and run it through the AerialOptimizer algorithm
7. Click "Plot Sim" to plot all the relevant trajectory data of the simulation.

TODO: I will be added the ability to choose a "controller" type (a controller is the algorithm that determines how to control the RLBot dependant on the trajectory sent to it and the current state. This could be FeedBack, FeedForward or whatever other crap i come up with)
TODO: Add functionality to the "load sim state" and "save sim state" buttons, to save the full SimulationState.
SimulationState has all the relevant info to run the simulation in RLBot. (Trajectory, Initial States, Controller Type, other necessary simulation/interface parameters)
TODO: Add a "run" button and interface with RLBot.
