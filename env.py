import gym
from gym.spaces import Discrete, Box
import os
import sys
import numpy as np
import math
import pandas as pd
import random
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME")

from sumolib import checkBinary
import traci

numBuses = 6

class SumoEnv(gym.Env):
    def __init__(self, gui=False, noWarnings=False):
        if gui:
            self._sumoBinary = checkBinary('sumo-gui')
        else:
            self._sumoBinary = checkBinary('sumo')

        self.sumoCmd = [self._sumoBinary, "-c", "ring.sumocfg", "--tripinfo-output", "tripinfo.xml", "--no-internal-links", "false"]
        if noWarnings:
            self.sumoCmd.append("--no-warnings")

        self.gymStep = 0
       
        self.stoppedBuses = [None for _ in range(numBuses)] #[None, None, None, None] # depends on number of buses
        self.decisionBus = ["bus.0", "stop1", 0]

        traci.start(self.sumoCmd)

        self.busStops = list(traci.simulation.getBusStopIDList())
        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

        # self.busCapacity = traci.vehicle.getPersonCapacity(self.decisionBus[0])
        self.busCapacity = 85

        # self.personsWithStop = []
        self.personsWithStop = dict()

        # self.peopleOnBuses = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]]
        self.peopleOnBuses = [[0]*12, [0]*12, [0]*12, [0]*12, [0]*12, [0]*12] # stores the people on each bus which will stop at each stop
        # print("PEOPLE ON BUSES: ", self.peopleOnBuses)

        self.action_space = Discrete(3)

        # DEPEND ON THE NETWORK
        # 2 instead of len(self.buses)
        # person capacity must be changed from 4 to ?
        # self.low = np.array([0 for _ in range(len(self.busStops))] + [0 for _ in range(numBuses)] +  [0, 0] +  [0 for _ in range(len(self.busStops))] + [0] + [0 for _ in range(len(self.busStops))] + [0, 0, 0], dtype='float32')
        # self.high = np.array([1 for _ in range(len(self.busStops))] + [1 for _ in range(numBuses)] + [5320, 5320] + [float('inf') for _ in self.busStops] + [float('inf')] + [200000 for _ in self.busStops] + [85, 85, 85], dtype='float32')
        # [[1,0,0],[1,0]]
        self.low = np.array([0 for _ in range(len(self.busStops))] + [0, 0] +  [0 for _ in range(len(self.busStops))] + [0] + [0 for _ in range(len(self.busStops))] + [0, 0, 0], dtype='float32')
        self.high = np.array([1 for _ in range(len(self.busStops))] + [5320, 5320] + [float('inf') for _ in self.busStops] + [float('inf')] + [200000 for _ in self.busStops] + [85, 85, 85], dtype='float32')


        self.observation_space = Box(self.low, self.high, dtype='float32')

        # self.reward_range = (float('-inf'), 0)
        self.reward_range = (0,250)

        self.sd = 0
        self.headwayReward = 0

        self.tempReward = 0


        self.stopTime = 0
        self.df = pd.DataFrame(columns=['HeadwayRew', 'SD', 'Reward', 'Action'])
        self.dfBunching = pd.DataFrame(columns=['Bus','Headway','Time'])
        # now = datetime.now()
        # time = now.strftime("%H:%M:%S")
        # self.dfBunching = pd.concat([self.dfBunching, pd.DataFrame.from_records([{'Bus':"NA", 'Headway':"NA", 'Time':time}])], ignore_index=True)


    def step(self, action):

        self.gymStep += 1
        print("GYM STEP: ", self.gymStep)
        # self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]
        

        #####################
        #   APPLY ACTION    #
        #####################
        
        if action == 0: # hold the bus
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=(self.decisionBus[2]+15))
            #UPDATE PEOPLE ON BUS
            personsOnStop = traci.busstop.getPersonIDs(self.decisionBus[1])
            # for key, value in self.personsWithStop:
            #boarding
            for person in personsOnStop:    
                self.personsWithStop[person][1] = self.decisionBus[0]
                self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] += 1


            #alighting
            personsOnBus = traci.vehicle.getPersonIDList(self.decisionBus[0])
            
            for person in personsOnBus:
                if self.personsWithStop[person][0] == self.decisionBus[1]:
                    self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] -= 1

            

            # self.peopleOnBuses

            # print("holding {} at {}".format(self.decisionBus[0], stopData[0].stoppingPlaceID))
        elif action == 1: # skip the stop
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=0)
            # print("ACTION1")
        #else action == 2, no action taken and bus behaves normally
        else:
            # print("NO ACTION")
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=self.decisionBus[2])
            # print("STOP: ", self.decisionBus[1])
            # print("DWELL TIME: ", self.decisionBus[2])

            #UPDATE PEOPLE ON BUS
            personsOnStop = traci.busstop.getPersonIDs(self.decisionBus[1])
            # for key, value in self.personsWithStop:
            #boarding
            for person in personsOnStop: 
                # print("PERSON ID: ", person)   
                self.personsWithStop[person][1] = self.decisionBus[0]
                self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] += 1


            #alighting
            personsOnBus = traci.vehicle.getPersonIDList(self.decisionBus[0])
            # print("PERSONS ON BUS: ", personsOnBus)
            for person in personsOnBus:
                if self.personsWithStop[person][0] == self.decisionBus[1]:
                    # print("DECREMENTING")
                    self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] -= 1


        ########################################
        #   FAST FORWARD TO NEXT DECISION STEP #
        ########################################

        reachedStopBuses = self.reachedStop()
        while len(reachedStopBuses) < 1:
            # if traci.simulation.getTime() == 28:
            #     print(reachedStopBuses)
            self.sumoStep()
            reachedStopBuses = self.reachedStop()
            # if traci.simulation.getTime() == 29:
            #     print(reachedStopBuses)

        ###### UPDATE DECISION BUS #######
        self.stopTime = self.getStopTime(reachedStopBuses[0][0], reachedStopBuses[0][1])
        self.decisionBus = [reachedStopBuses[0][0], reachedStopBuses[0][1], self.stopTime]


        ###############################################
        #   GET NEW OBSERVATION AND CALCULATE REWARD  #
        ###############################################

        # ADD DWELL TIME TO STATE!!!!
        state = self.computeState()

        reward = self.computeReward("sd", 1,0)# 0.6, 0.4)

        # self.df = self.df.append({'SD':self.sd, 'Reward':reward, 'Action':action}, ignore_index=True)
        self.df = pd.concat([self.df, pd.DataFrame.from_records([{'HeadwayRew':self.headwayReward, 'SD':self.sd, 'Reward':reward, 'Action':action}])], ignore_index=True)

        if self.gymStep > 250:#125:#1500
            print("DONE")
            # print(self.decisionBus)
            # print("PERSONS WITH STOP: ", self.personsWithStop)
            done = True
            # self.df.to_csv('logWithModel.csv')
            # self.df.to_csv('logWithModelNewHW.csv')
            # self.df.to_csv('logNewHW.csv')
            # self.dfBunching.to_csv('bunchingGUIHighSpeedNewModelNewHW.csv')
            
        else:
            done = False

        info = {}

        return state, reward, done, info


    def reset(self):
        traci.close()
        traci.start(self.sumoCmd)
        self.gymStep = 0
        self.stoppedBuses = [None for _ in range(numBuses)] #[None, None, None, None]
        self.decisionBus = ["bus.0", "stop1", 0]
        # self.personsWithStop = []
        self.personsWithStop = dict()
        self.peopleOnBuses = [[0]*12, [0]*12, [0]*12, [0]*12, [0]*12, [0]*12]

        self.sd = 0
        self.stopTime = 0

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
        self.updatePersonStop() #uncomment
        if len([bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]) == numBuses:
            self.updatePassengersOnBoard()

    def computeState(self):
        stop = self.oneHotEncode(self.busStops, self.decisionBus[1])
        bus = self.oneHotEncode(self.buses, self.decisionBus[0])

        headways = self.getHeadways()
        # if headways[0] < 100:
        #     now = datetime.now()

        #     time = now.strftime("%H:%M:%S")
        #     self.dfBunching = pd.concat([self.dfBunching, pd.DataFrame.from_records([{'Bus':self.decisionBus[0], 'Headway':headways[0], 'Time':time}])], ignore_index=True)
        # elif headways[1] < 100:
        #     now = datetime.now()

        #     time = now.strftime("%H:%M:%S")
        #     self.dfBunching = pd.concat([self.dfBunching, pd.DataFrame.from_records([{'Bus':self.decisionBus[0], 'Headway':headways[1], 'Time':time}])], ignore_index=True)

        
        # print("forward headway from decision {} = {}".format(self.decisionBus[0], headways[0]))
        # print("backward headway from decision {} = {}".format(self.decisionBus[0], headways[1]))
        
        waitingPersons = self.getPersonsOnStops()

        # print("no of waiting persons: ", waitingPersons)

        maxWaitTimes = self.getMaxWaitTimeOnStops()

        # print("max wait times: ", maxWaitTimes)

        numPassengers = self.getNumPassengers()

        # state = [stop] + [bus] + [headways] + [waitingPersons] + [maxWaitTimes] + [numPassengers]
        # state = stop + bus + headways + waitingPersons + [self.stopTime] + maxWaitTimes + numPassengers
        state = stop + headways + waitingPersons + [self.stopTime] + maxWaitTimes + numPassengers

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
        numEdges = 12
        leaderRoad = int(traci.vehicle.getRoadID(leader))
        followerRoad = int(traci.vehicle.getRoadID(follower))
        # print("leader: ", leader)
        # print("follower: ", follower)

        # print("leader road: ", leaderRoad)
        # print("follower road: ", followerRoad)

        if leaderRoad == followerRoad:
            if traci.vehicle.getLanePosition(leader) - traci.vehicle.getLanePosition(follower) > 0:
                # print("leader pos: ", traci.vehicle.getLanePosition(leader))
                # print("follower pos: ", traci.vehicle.getLanePosition(follower))
                # print("..........................................")
                return traci.vehicle.getLanePosition(leader) - traci.vehicle.getLanePosition(follower)
        
        h = traci.lane.getLength(traci.vehicle.getLaneID(follower)) - traci.vehicle.getLanePosition(follower)
        # print("Follower road length: ", traci.lane.getLength(traci.vehicle.getLaneID(follower)))
        # print("Follower pos: ", traci.vehicle.getLanePosition(follower))
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
            # print("ROAD LEN: ", traci.lane.getLength(str(lane)+"_0"))
            h += traci.lane.getLength(str(lane)+"_0")

        h+= traci.vehicle.getLanePosition(leader)
        # print("Leader pos: ", traci.vehicle.getLanePosition(leader))
        # print("...........................................")

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
            # print("############################################")
            

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

        # reward = -alpha * abs(headways[0] - headways[1])

        reward = -abs(headways[0] - headways[1])


        # self.headwayReward = -alpha * abs(headways[0] - headways[1])

        # reward function from paper using only its first term
        reward = math.exp(-abs(headways[0] - headways[1]))
        self.tempReward += reward
        # print('reward: ', reward)
        # print('forward: ', headways[0])
        # print('backward: ', headways[1])
        # print('total reward: ', self.tempReward)


        # reward = -(abs(886.67-headways[0]) + abs(886.67-headways[1]))/1773.34 #normalize between zero and one (incorrect?)
        # print(reward)

        # reward = -alpha * (abs(886.67-headways[0]) + abs(886.67-headways[1]))
        # self.headwayReward = -alpha * (abs(886.67-headways[0]) + abs(886.67-headways[1]))

        

        # print("VARIANCE: ", self.getWaitingTimeVariance())

        #TEST WITHOUT THE WAITING TIME IN REWARD
        # if s == "variance":
        #     reward += -beta * self.getWaitingTimeVariance()
        # elif s == "sd":
        #     reward += -beta * self.getWaitStandardDevUsingMax()
        #     self.sd = -beta * self.getWaitStandardDevUsingMax()

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
            #OLDEST
            # num = random.randint(4,5) #needs to be fixed when using the proper circuit
            # if num==4:
            #     s = "stop2"
            # else:
            #     s = "stop3"

            #RING
            num = random.randint(1,6)
            edge = traci.person.getRoadID(person)
            newEdge = (int(edge) + num) % 12
            newStop = newEdge + 1
            stop = "stop"+str(newStop)
            traci.person.appendDrivingStage(person, str(newEdge), "line1", stopID=stop) #RING
            traci.person.appendWalkingStage(person, [str(newEdge)], 250) #RING
            self.personsWithStop[person] = [stop, None]


            # traci.person.appendDrivingStage(person, str(num), "line1", stopID=s)#str(num), "line1", stopID=s) #OLDEST
            # traci.person.appendWalkingStage(person, [str(num)], 30) #OLDEST
            # self.personsWithStop[person] = [s, None] #OLDEST

            

            # traci.person.appendDrivingStage(person, 0, "line1", stopID="stop1") #WORKED
            # traci.person.appendWalkingStage(person, [str("0")], 230) #WORKED
            
            
            # self.personsWithStop[person] = ["stop1", None] #WORKED
            

    def getStopTime(self, bus, stop):
        boarding = traci.busstop.getPersonCount(stop)
        #update people on bus - NOT HERE, NEED TO BE SURE THAT IT WILL ACTUALLY STOP
        # alighting = sum([1 for key, value i])
        # print("bus[-1] {} (stop-1) {}".format(int(bus[-1]), (stop-1)))
        alighting = self.peopleOnBuses[int(bus[-1])][int(stop[-1])-1]

        # print("BOARDING: {} ALIGHTING: {}".format(boarding, alighting))
        # print("PEOPLE ON BUS: {}".format(self.peopleOnBuses))
        #work out dwell time
        time = max(math.ceil(boarding/3), math.ceil(abs(alighting)/1.8)) #boarding and alighting rate #abs is there just in case is falls below zero if a person should've left a bus but the simulation did not give them time
        return time

    def updatePassengersOnBoard(self): #For any passengers which board the bus during holding time and thus were not know beforehand that they would board
        for bus in self.buses:
            for person in traci.vehicle.getPersonIDList(bus):
                if self.personsWithStop[person][1] == None:
                    self.personsWithStop[person][1] = bus #WAS == BEFORE (A MISTAKE)
                    self.peopleOnBuses[int(bus[-1])][int(self.personsWithStop[person][0][-1])-1] += 1
                    # print("INCREMENTED {} STOP {}".format(bus, self.personsWithStop[person][0]))

        #not sure if i need to update any persons alighting here as well
        #If multiple buses stop at same time and this causes problems, remove alighting time altogether      


