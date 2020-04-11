DeepToot- toot the ball deep into the goal3

Basics on how to run and use

Two portions:
First portion is the data generator portion where a human plays the game to generate data
Second portion is NN training, data analysis etc... using the CLI we created.

First portion:
Run run-gui.bat
Once RLBot gui is runnint make sure there is 1 human player and one bot.py player
Press RUN
Cmd Prompt will display the current time for 3 seconds then start recording the human input packets
Lastly it will export the data into the 'generated_data' folder inside the 'src' folder in RLBot where we have all our code
That's it for this, the data will be saved and now can be used in the CLI

CLI Interface:
In cmd prompt call [pipenv run toot <argument>]
Right now there are only some preliminary arguments
To plot the saved human data use [pipenv run toot plot]
Follow command line instructions and plot will show the position trajectory