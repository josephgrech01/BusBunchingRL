# Control Strategies for Transport Network Optimisatioin
## RL to minimise and prevent Bus Bunching

This repository contains all the code, trained models, and results corresponding to a study that made use of Reinforcement Learning to tackle the Bus Bunching phenomenon. This work used the PPO and TRPO algorithms from Stable-Baselines3 on a bus route simulated using the SUMO traffic simulator. The Python code makes use of the TraCI API which makes it possible to alter the course of the simulation by applying the actions chosen by the algorithms as the episode progresses. 

The following are the available actions:
* Hold - The bus remains at the stop for some time after letting people board and alight
* Skip - The bus skips the upcoming stop
* Proceed - The bus departs from the stop immediately after letting people board and alight (as it would normally do)


