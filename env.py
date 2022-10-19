import optparse
import gym
from gym.spaces import Discrete, Box
import os
import sys
import optparse

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME")

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                        default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

from sumolib import checkBinary
import traci


class SumoEnv(gym.Env):
    def __init__(self):
        options = get_options()
        if options.nogui:
            self._sumoBinary = checkBinary('sumo')
        else:
            self._sumoBinary = checkBinary('sumo-gui')

        self.sumoCmd = [self._sumoBinary, "-c", "demo.sumocfg", "--tripinfo-output", "tripinfo.xml", "--no-internal-links", "false"]

        self.gymStep = 0

        traci.start(self.sumoCmd)


    def step(self, action):

        self.gymStep += 1

        #####################
        #   APPLY ACTION    #
        #####################

        ########################################
        #   FAST FORWARD TO NEXT DECISION STEP #
        ########################################
        self.sumoStep()
        while len(self.stoppedBuses()) < 1:
            self.sumoStep()


        ###############################################
        #   GET NEW OBSERVATION AND CALCULATE REWARD  #
        ###############################################

        state = {}

        reward = 0

        if self.gymStep > 10:
            done = True
            print(self.gymStep)
        else:
            done = False

        info = {}

        return state, reward, done, info


    def reset(self):
        traci.close()
        print("first")
        traci.start(self.sumoCmd)
        print("second")
    def close(self):
        traci.close()

    def stoppedBuses(self):
        stopped = dict()
        for stop in ["stop1", "stop2"]:
            buses = traci.busstop.getVehicleIDs(stop)
            for bus in buses:
                stopped[bus] = stop
        return stopped

    def sumoStep(self):
        traci.simulationStep()


env = SumoEnv()
episodes = 5
for episode in range(1, episodes + 1):

    state = env.reset()

    done = False
    score = 0

    while not done:
        state, reward, done, info = env.step(0)
        score += reward

    print("Episode: {} Score: {}".format(episode, score))


    