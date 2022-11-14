import gym
from gym.spaces import Discrete, Box
import os
import sys
import numpy as np
import math
import pandas as pd
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME")

from sumolib import checkBinary
import traci

numBuses = 2

class SumoEnv(gym.Env):
    def __init__(self, gui=False, noWarnings=False):
        if gui:
            self._sumoBinary = checkBinary('sumo-gui')
        else:
            self._sumoBinary = checkBinary('sumo')

        self.sumoCmd = [self._sumoBinary, "-c", "demo.sumocfg", "--tripinfo-output", "tripinfo.xml", "--no-internal-links", "false"]
        if noWarnings:
            self.sumoCmd.append("--no-warnings")

        self.gymStep = 0
       
        self.stoppedBuses = [None for _ in range(numBuses)] #[None, None, None, None] # depends on number of buses
        self.decisionBus = ["bus.0", "stop1"]

        traci.start(self.sumoCmd)

        self.busStops = list(traci.simulation.getBusStopIDList())
        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

        # self.busCapacity = traci.vehicle.getPersonCapacity(self.decisionBus[0])
        self.busCapacity = 85

        # self.personsWithStop = []
        self.personsWithStop = dict()

        self.action_space = Discrete(3)

        # DEPEND ON THE NETWORK
        # 2 instead of len(self.buses)
        # person capacity must be changed from 4 to ?
        self.low = np.array([0 for _ in range(len(self.busStops))] + [0 for _ in range(numBuses)] +  [0, 0] +  [0 for _ in range(len(self.busStops))] +  [0 for _ in range(len(self.busStops))] + [0, 0, 0], dtype='float32')
        self.high = np.array([1 for _ in range(len(self.busStops))] + [1 for _ in range(numBuses)] + [230, 230] + [float('inf') for _ in self.busStops] + [2000 for _ in self.busStops] + [85, 85, 85], dtype='float32')
        # [[1,0,0],[1,0]]


        self.observation_space = Box(self.low, self.high, dtype='float32')

        self.reward_range = (float('-inf'), 0)

        self.sd = 0
        self.df = pd.DataFrame(columns=['SD', 'Reward', 'Action'])


    def step(self, action):

        self.gymStep += 1
        # self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]
        # self.updatePersonStop()
        

        #####################
        #   APPLY ACTION    #
        #####################
        
        if action == 0: # hold the bus
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=15)
            # print("holding {} at {}".format(self.decisionBus[0], stopData[0].stoppingPlaceID))
        elif action == 1: # skip the stop
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=0)
            # print("ACTION1")
        #else action == 2, no action taken and bus behaves normally
        else:
            # print("NO ACTION")
            pass


        ########################################
        #   FAST FORWARD TO NEXT DECISION STEP #
        ########################################

        reachedStopBuses = self.reachedStop()
        while len(reachedStopBuses) < 1:
            if traci.simulation.getTime() == 28:
                print(reachedStopBuses)
            self.sumoStep()
            reachedStopBuses = self.reachedStop()
            if traci.simulation.getTime() == 29:
                print(reachedStopBuses)

        ###### UPDATE DECISION BUS #######
        self.decisionBus = [reachedStopBuses[0][0], reachedStopBuses[0][1]]


        ###############################################
        #   GET NEW OBSERVATION AND CALCULATE REWARD  #
        ###############################################

        state = self.computeState()

        reward = self.computeReward("sd", 0.6, 0.4)

        # self.df = self.df.append({'SD':self.sd, 'Reward':reward, 'Action':action}, ignore_index=True)
        self.df = pd.concat([self.df, pd.DataFrame.from_records([{'SD':self.sd, 'Reward':reward, 'Action':action, 'Max Speed':traci.vehicle.getMaxSpeed("bus.0"), 'Speed': traci.vehicle.getSpeed("bus.0")}])], ignore_index=True)

        if self.gymStep > 50:
            print("DONE")
            print(self.decisionBus)
            print("PERSONS WITH STOP: ", self.personsWithStop)
            done = True
            self.df.to_csv('log.csv')
            
        else:
            done = False

        info = {}

        return state, reward, done, info


    def reset(self):
        traci.close()
        traci.start(self.sumoCmd)
        self.gymStep = 0
        self.stoppedBuses = [None for _ in range(numBuses)] #[None, None, None, None]
        self.decisionBus = ["bus.0", "stop1"]
        # self.personsWithStop = []
        self.personsWithStop = dict()

        self.sd = 0

        # sumo step until all buses are in the simulation
        while len(traci.vehicle.getIDList()) < numBuses: #DEPENDS ON THE NUMBER OF BUSES
            self.sumoStep()


        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

        state = self.computeState()
        return state

    def close(self):
        traci.close()

    def stoppedBuses(self):
        stopped = dict()
        for stop in ["stop1", "stop2", "stop3"]:
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
                        # print(vehicle)
                        #get stop id and update stoppedBuses list
                        for stop in self.busStops:
                            buses = traci.busstop.getVehicleIDs(stop)
                            if vehicle in buses:
                                self.stoppedBuses[int(vehicle[-1])] = stop
                                stopped[vehicle] = stop

                else:
                    if self.stoppedBuses[int(vehicle[-1])] != None:
                        self.stoppedBuses[int(vehicle[-1])] = None
        return stopped

    def reachedStop(self):
        reached = []
        for vehicle in traci.vehicle.getIDList():
            if vehicle[0:3] == "bus":
                for stop in self.busStops:
                    if traci.busstop.getLaneID(stop) == traci.vehicle.getLaneID(vehicle):
                        if (traci.vehicle.getLanePosition(vehicle) >= (traci.busstop.getStartPos(stop) - 5)) and (traci.vehicle.getLanePosition(vehicle) <= (traci.busstop.getEndPos(stop) + 1)):
                            if self.stoppedBuses[int(vehicle[-1])] == None:
                                # get stop id and update stopped bused list
                                self.stoppedBuses[int(vehicle[-1])] = stop
                                reached.append([vehicle, stop])
                                # break not sure
                        else:
                            if self.stoppedBuses[int(vehicle[-1])] != None:
                                self.stoppedBuses[int(vehicle[-1])] = None
        
        return reached


    def sumoStep(self):
        traci.simulationStep()
        self.updatePersonStop()

    def computeState(self):
        stop = self.oneHotEncode(self.busStops, self.decisionBus[1])
        bus = self.oneHotEncode(self.buses, self.decisionBus[0])
        headways = self.getHeadways()
        
        # print("forward headway from decision {} = {}".format(self.decisionBus[0], headways[0]))
        # print("backward headway from decision {} = {}".format(self.decisionBus[0], headways[1]))
        
        waitingPersons = self.getPersonsOnStops()

        # print("no of waiting persons: ", waitingPersons)

        maxWaitTimes = self.getMaxWaitTimeOnStops()

        # print("max wait times: ", maxWaitTimes)

        numPassengers = self.getNumPassengers()

        # state = [stop] + [bus] + [headways] + [waitingPersons] + [maxWaitTimes] + [numPassengers]
        state = stop + bus + headways + waitingPersons + maxWaitTimes + numPassengers

        # print("state: ", state)
        # return np.array(state, dtype='float32')
        return state

    def oneHotEncode(self, list, item):
        return [1 if i == item else 0 for i in list]

    def getHeadway(self, leader, follower): # first edge id must be 0, % depends on number of edges #gives forward headway of follower?
        # forward headway is wrong
        h = traci.lane.getLength(traci.vehicle.getLaneID(follower)) - traci.vehicle.getLanePosition(follower)
        # print(traci.lane.getLength(traci.vehicle.getLaneID(follower)) - traci.vehicle.getLanePosition(follower))
        repeats = abs(int(traci.vehicle.getRoadID(leader)) - int(traci.vehicle.getRoadID(follower))) - 1
        # print("repeats: ", repeats)
        # print("leader road: ", int(traci.vehicle.getRoadID(leader)))
        # print("follower road: ", int(traci.vehicle.getRoadID(follower)))
        for i in range(repeats):
            h += traci.lane.getLength(str((int(traci.vehicle.getRoadID(follower))+i+1)%6)+"_0")


        h += traci.vehicle.getLanePosition(leader) 

        return h

    def getForwardHeadway(self, leader, follower):
        numEdges = 6
        leaderRoad = int(traci.vehicle.getRoadID(leader))
        followerRoad = int(traci.vehicle.getRoadID(follower))

        # print("leader road: ", leaderRoad)
        # print("follower road: ", followerRoad)

        if leaderRoad == followerRoad:
            if traci.vehicle.getLanePosition(leader) - traci.vehicle.getLanePosition(follower) > 0:
                return traci.vehicle.getLanePosition(leader) - traci.vehicle.getLanePosition(follower)
        
        h = traci.lane.getLength(traci.vehicle.getLaneID(follower)) - traci.vehicle.getLanePosition(follower)
        if leaderRoad == followerRoad:
            repeats = numEdges - 1
        elif leaderRoad > followerRoad:
            repeats = leaderRoad - followerRoad - 1
        else:
            repeats = (numEdges - (abs(leaderRoad - followerRoad))) - 1
        
        # print("REPEATS: ", repeats)
        for i in range(repeats):
            lane = int(traci.vehicle.getRoadID(follower)) + i + 1
            if lane >= numEdges:
                lane = lane % numEdges
            # print("ROAD ID: ", lane)
            h += traci.lane.getLength(str(lane)+"_0")

        h+= traci.vehicle.getLanePosition(leader)

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

        # print("buses: ", self.buses)
        # print("decision bus: ", self.decisionBus[0])
        # print("follower: ", follower)
        # print("leader: ", leader)

        return follower, leader

    def getHeadways(self):
        if len(self.buses) > 1:
            follower, leader = self.getFollowerLeader()

            # print("FOLLOWER: ", follower)
            # print("LEADER: ", leader)
            # print("BUS: ", self.decisionBus)
            
            
            #forwardHeadway = self.getHeadway(leader, self.decisionBus[0])
            forwardHeadway = self.getForwardHeadway(leader, self.decisionBus[0])
            # print("FORWARD: ", forwardHeadway)
            #backwardHeadway = self.getHeadway(self.decisionBus[0], follower)
            backwardHeadway = self.getForwardHeadway(self.decisionBus[0], follower)
            # print("BACKWARD: ", backwardHeadway)
            

            return [forwardHeadway, backwardHeadway]
        else:
            return [0, 0]

    def getPersonsOnStops(self):
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


    def computeReward(self, s, alpha, beta):
        reward = 0
        headways = self.getHeadways()

        reward = -alpha * abs(headways[0] - headways[1])

        # print("VARIANCE: ", self.getWaitingTimeVariance())

        if s == "variance":
            reward += -beta * self.getWaitingTimeVariance()
        elif s == "sd":
            reward += -beta * self.getWaitStandardDevUsingMax()

        return reward
        

    def getWaitingTimeVariance(self):
        meanSquares = []
        for stop in self.busStops:
            waitTime = 0
            
            totalPersons = traci.busstop.getPersonCount(stop)
            if totalPersons > 0:
                personsOnStop = traci.busstop.getPersonIDs(stop)
                for person in personsOnStop:
                    waitTime += (traci.person.getWaitingTime(person)) #** 2)
                    #maximum instead of total

                meanSquares.append(waitTime/totalPersons)

        if len(meanSquares) > 0:
            average = sum(meanSquares)/len(meanSquares)
            deviations = [((ms - average) ** 2) for ms in meanSquares]
            waitingTimeVariance = sum(deviations) / len(meanSquares)

            # print("AVERAGE: ", average)
            # print("LEN MEAN SQAURES: ", len(meanSquares))
            # print("DEVIATIONS: ", deviations)
            # print("MEAN SQUARES: ", meanSquares) 
        else:
            # print("ZERO MEAN SQUARES")
            waitingTimeVariance = 0

        # print("WAIT VARIANCE", waitingTimeVariance)
        return waitingTimeVariance    

    def getWaitStandardDevUsingMax(self):
        maximums = [m**2 for m in self.getMaxWaitTimeOnStops()]

        average = sum(maximums)/len(maximums)
        deviations = [((m - average) ** 2) for m in maximums]
        variance = sum(deviations)/len(maximums)
        self.sd = math.sqrt(variance)
        return math.sqrt(variance)

    def updatePersonStop(self):
        persons = traci.person.getIDList()
        # personsWithoutStop = [person for person in persons if person not in self.personsWithStop]
        personsWithoutStop = [person for person in persons if person not in self.personsWithStop]
        for person in personsWithoutStop:
            num = random.randint(4,5) #needs to be fixed when using the proper circuit
            if num==4:
                s = "stop2"
            else:
                s = "stop3"
            traci.person.appendDrivingStage(person, str(num), "line1", stopID=s)
            traci.person.appendWalkingStage(person, [str(num)], 30)
            # self.personsWithStop.append(person)
            self.personsWithStop[person] = s
            print("PERSON {} TO STOP {}\n".format(person, s))




        
