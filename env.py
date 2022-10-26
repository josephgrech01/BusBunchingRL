import optparse
from statistics import variance
from cv2 import getNumberOfCPUs
import gym
from gym.spaces import Discrete, Box
import os
import sys
import optparse
import numpy as np
# from torch import float32

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
        self.busStops = ["stop1", "stop2"]
        print(self.busStops)
       
        self.stoppedBuses = [None, None]
        self.decisionBus = ["bus.0", "stop1"]

        traci.start(self.sumoCmd)

        self.busStops = list(traci.simulation.getBusStopIDList())
        print(self.busStops)

        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

        # self.busCapacity = traci.vehicle.getPersonCapacity(self.decisionBus[0])

        self.action_space = Discrete(3)

        # DEPEND ON THE NETWORK
        # 2 instead of len(self.buses)
        low = np.array([0 for _ in range(len(self.busStops))] + [0 for _ in range(2)] +  [0, 0] +  [0 for _ in range(len(self.busStops))] +  [0 for _ in range(len(self.busStops))] + [0, 0, 0], dtype='float32')
        high = np.array([1 for _ in range(len(self.busStops))] + [1 for _ in range(2)] + [196, 196] + [float('inf') for _ in self.busStops] + [2000 for _ in self.busStops] + [4, 4, 4], dtype='float32')

        self.observation_space = Box(low, high, dtype='float32')

        self.reward_range = (float('-inf'), 0)


    def step(self, action):

        self.gymStep += 1
        print("new gym step ", self.gymStep)

        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

        

        #####################
        #   APPLY ACTION    #
        #####################
        if traci.simulation.getTime() > 1: #the first bus leaves after the first simulation step
            if action == 0: # hold the bus
                stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
                traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=70)
                print("holding {} at {}".format(self.decisionBus[0], stopData[0].stoppingPlaceID))
            elif action == 1: # skip the stop
                stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
                traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=0)
            #else action == 2, no action taken and bus behaves normally


        ########################################
        #   FAST FORWARD TO NEXT DECISION STEP #
        ########################################

        # self.sumoStep()
        # while len(self.stoppedBuses()) < 1:
        #     self.sumoStep()


        # self.sumoStep() # CHECK ############
        reachedStopBuses = self.reachedStop()
        while len(reachedStopBuses) < 1:
            self.sumoStep()
            reachedStopBuses = self.reachedStop()

        ###### UPDATE DECISION BUS #######
        # self.decisionBus = [str(list(reachedStopBuses.keys())[0]), str(list(reachedStopBuses.items())[0])]
        self.decisionBus = [reachedStopBuses[0][0], reachedStopBuses[0][1]]


        ###############################################
        #   GET NEW OBSERVATION AND CALCULATE REWARD  #
        ###############################################

        state = {}
        if self.gymStep == 5:
            state = self.computeState()
            print(state)

        reward = self.computeReward()

        if self.gymStep > 10:
            print(self.gymStep)
            done = True
            
        else:
            done = False

        info = {}

        return state, reward, done, info


    def reset(self):
        traci.close()
        traci.start(self.sumoCmd)
        self.gymStep = 0
        self.stoppedBuses = [None, None]
        self.decisionBus = ["bus.0", "stop1"]
        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

    def close(self):
        traci.close()

    def stoppedBuses(self):
        stopped = dict()
        for stop in ["stop1", "stop2"]:
            buses = traci.busstop.getVehicleIDs(stop)
            for bus in buses:
                stopped[bus] = stop
        return stopped

    def newStoppedBus(self):   
        stopped = dict() 
        for vehicle in traci.vehicle.getIDList():
            if vehicle[0:3] == "bus":
                if traci.vehicle.isAtBusStop(vehicle):
                    if self.stoppedBuses[int(vehicle[-1])] == None:
                        print(vehicle)
                        #get stop id and update stoppedBuses list
                        for stop in self.busStops:
                            buses = traci.busstop.getVehicleIDs(stop)
                            if vehicle in buses:
                                self.stoppedBuses[int(vehicle[-1])] = stop
                                stopped[vehicle] = stop

                else:
                    if self.stoppedBuses[int(vehicle[-1])] != None:
                        self.stoppedBuses[int(vehicle[-1])] = None

        # for vehicle in traci.simulation.getStopStartingVehiclesIDList():
        #     if vehicle[0:3] == "bus":
        #         for stop in self.busStops:
        #             buses = traci.busstop.getVehicleIDs(stop)
        #             if vehicle in buses:
        #                 self.stoppedBuses[int(vehicle[-1])] = stop
        #                 stopped[vehicle] = stop
        # for vehicle in traci.simulation.getStopEndingVehiclesIDList():
        #     if vehicle[0:3] == "bus":
        #         self.stoppedBuses[int(vehicle[-1])] = None
        return stopped

    def reachedStop(self):
        # reached = dict()
        reached = []
        for vehicle in traci.vehicle.getIDList():
            if vehicle[0:3] == "bus":
                for stop in self.busStops:
                    if traci.busstop.getLaneID(stop) == traci.vehicle.getLaneID(vehicle):
                        if (traci.vehicle.getLanePosition(vehicle) >= (traci.busstop.getStartPos(stop) - 5)) and (traci.vehicle.getLanePosition(vehicle) <= (traci.busstop.getEndPos(stop) + 1)):
                            if self.stoppedBuses[int(vehicle[-1])] == None:
                                # get stop id and update stopped bused list
                                self.stoppedBuses[int(vehicle[-1])] = stop
                                # reached[vehicle] = stop
                                reached.append([vehicle, stop])
                        else:
                            if self.stoppedBuses[int(vehicle[-1])] != None:
                                self.stoppedBuses[int(vehicle[-1])] = None
        
        return reached


    def sumoStep(self):
        traci.simulationStep()

    def computeState(self):
        stop = self.oneHotEncode(self.busStops, self.decisionBus[1])
        bus = self.oneHotEncode(self.buses, self.decisionBus[0])
        headways = self.getHeadways()
        
        print("forward headway from decision {} = {}".format(self.decisionBus[0], headways[0]))
        print("backward headway from decision {} = {}".format(self.decisionBus[0], headways[1]))
        
        waitingPersons = self.getPersonsOnStops()

        print("no of waiting persons: ", waitingPersons)

        maxWaitTimes = self.getMaxWaitTimeOnStops()

        print("max wait times: ", maxWaitTimes)

        numPassengers = self.getNumPassengers()

        state = [stop] + [bus] + [headways] + [waitingPersons] + [maxWaitTimes] + [numPassengers]

        print("state: ", state)
        # return np.array(state, dtype='float32')
        return state

    def oneHotEncode(self, list, item):
        return [1 if i == item else 0 for i in list]

    def getHeadway(self, leader, follower):
        h = traci.lane.getLength(traci.vehicle.getLaneID(follower)) - traci.vehicle.getLanePosition(follower)
    
        repeats = abs(int(traci.vehicle.getRoadID(leader)) - int(traci.vehicle.getRoadID(follower))) - 1
        
        for i in range(repeats):
            print(type(str((int(traci.vehicle.getRoadID(follower))+i+1)%6)))
            h += traci.lane.getLength(str((int(traci.vehicle.getRoadID(follower))+i+1)%6)+"_0")


        h += traci.vehicle.getLanePosition(leader) 

        return h

    def getFollowerLeader(self):
        if int(self.decisionBus[0][-1]) + 1 == len(self.buses):
            follower = "bus.0"
        else:
            follower = "bus." + str(int(self.decisionBus[0][-1]) + 1)

        if int(self.decisionBus[0][-1]) == 0:
            leader = "bus." + str(len(self.buses) - 1)
        else:
            leader = "bus." + str(int(self.decisionBus[0][-1]) - 1)

        return follower, leader

    def getHeadways(self):
        if len(self.buses) > 1:
            # if int(self.decisionBus[0][-1]) + 1 == len(self.buses):
            #     follower = "bus.0"
            # else:
            #     follower = "bus." + str(int(self.decisionBus[0][-1]) + 1)

            # if int(self.decisionBus[0][-1]) == 0:
            #     leader = "bus." + str(len(self.buses) - 1)
            # else:
            #     leader = "bus." + str(int(self.decisionBus[0][-1]) - 1)

            follower, leader = self.getFollowerLeader()

            forwardHeadway = self.getHeadway(leader, self.decisionBus[0])
            backwardHeadway = self.getHeadway(self.decisionBus[0], follower)

            return [forwardHeadway, backwardHeadway]
        else:
            return [0, 0]

    def getPersonsOnStops(self):
        # persons = []
        # for stop in self.busStops:
        #     persons.append(traci.busstop.getPersonCount(stop))
        #     print(stop, ": ", traci.busstop.getPersonCount(stop))
        persons = [traci.busstop.getPersonCount(stop) for stop in self.busStops]

        return persons

    def getMaxWaitTimeOnStops(self):
        maxWaitTimes = []
        for stop in self.busStops:
            personsOnStop = traci.busstop.getPersonIDs(stop)
            waitTimes = [traci.person.getWaitingTime(person) for person in personsOnStop]
            if len(waitTimes) > 0:
                maxWaitTimes.append(max(waitTimes))
            else:
                maxWaitTimes.append(0)

        return maxWaitTimes

    def getNumPassengers(self):
        follower, leader = self.getFollowerLeader()

        numPassengers = [traci.vehicle.getPersonNumber(leader), traci.vehicle.getPersonNumber(self.decisionBus[0]), traci.vehicle.getPersonNumber(follower)]
        return numPassengers


    def computeReward(self):
        reward = 0
        headways = self.getHeadways()

        reward = -1 * abs(headways[0] - headways[1])



        return reward

    def getWaitingTimeVariance(self):
        meanSquares = []
        for stop in self.busStops:
            waitTime = 0
            
            totalPersons = traci.busstop.getPersonCount(stop)
            if totalPersons > 0:
                personsOnStop = traci.busstop.getPersonIDs(stop)
                for person in personsOnStop:
                    waitTime += (traci.person.getWaitingTime(person) ** 2)

                meanSquares.append(waitTime/totalPersons)

        average = sum(meanSquares)/len(meanSquares)

        # sum = 0
        # for ms in meanSquares:
        #     sum += ((ms - average)**2)
        deviations = [(ms - average) ** 2 for ms in meanSquares]

        waitingTimeVariance = sum(deviations) / len(meanSquares) 

        return waitingTimeVariance

    
        



env = SumoEnv()
episodes = 3
for episode in range(1, episodes + 1):

    state = env.reset()

    done = False
    score = 0

    while not done:
        state, reward, done, info = env.step(10)
        score += reward

    print("Episode: {} Score: {}".format(episode, score))

env.close()


    