# Control Strategies for Transport Network Optimisatioin
## RL to minimise and recover from Bus Bunching

This repository contains all the code, trained models, and results corresponding to a study that made use of Reinforcement Learning to tackle the Bus Bunching phenomenon. This work used the PPO and TRPO algorithms from Stable-Baselines3 on a bus route simulated using the SUMO traffic simulator. The Python code makes use of the TraCI API which makes it possible to alter the course of the simulation by applying the actions chosen by the algorithms as the episode progresses. 

The following are the available control strategies:
* Hold - The bus remains at the stop for some time after letting people board and alight
* Skip - The bus skips the upcoming stop
* Proceed Normally - The bus departs from the stop immediately after letting people board and alight

## To run

The user must have the SUMO simulator installed in order to be able to run the simulations using Python. A custom Gym environment is found in **env.py** which is then used by the RL algorithms in the files mentioned below. When creating the <em>SumoEnv</em> object the user can set the episode length, whether or not to include traffic, and whether or not the buses start bunched or equally spaced.

### usingPPO.py
This runs the simulation using the model trained with the PPO algorithm. 

### usingTRPO.py
This runs the simulation using the model trained with the TRPO algorithm. 


