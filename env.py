import gym
from gym.spaces import Discrete, Box
import os
import sys
import numpy as np
import math
import pandas as pd
import random
from datetime import datetime
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME")

from sumolib import checkBinary
import traci

numBuses = 6

class SumoEnv(gym.Env):
    # epLen: the number of RL steps before the episode terminates
    # traffic: set to zero for no traffic or else to the lowest traffic speed in km/h
    # bunched: True - buses start already bunched, False - buses start evenly spaced
    # mixedConfigs: Used during training to alternate between already bunched and evenly spaced scenarios
    def __init__(self, gui=False, noWarnings=False, epLen=250, traffic=0, bunched=False, mixedConfigs=False):
        if gui:
            self._sumoBinary = checkBinary('sumo-gui')
        else:
            self._sumoBinary = checkBinary('sumo')

        self.episodeNum = 0

        self.traffic = traffic
        self.mixedConfigs = mixedConfigs

        if bunched:
            self.config = 'sumo/bunched/ring.sumocfg'
        else:
            self.config = 'sumo/traffic/ring.sumocfg'

        if self.traffic == 0:
            self.config = 'sumo/noTraffic/ring.sumocfg'

        self.noWarnings = noWarnings
        self.sumoCmd = [self._sumoBinary, "-c", self.config, "--tripinfo-output", "tripinfo.xml", "--no-internal-links", "false", "--lanechange.overtake-right", "true"]
        if self.noWarnings:
            self.sumoCmd.append("--no-warnings")

        self.epLen = epLen

        if self.traffic == -1:
            self.lowestTrafficSpeed = random.randint(7,10)
        else:
            self.lowestTrafficSpeed = self.traffic

        self.gymStep = 0
       
        self.stoppedBuses = [None for _ in range(numBuses)]
        
        self.bunchingGraphData = {0:[[]], 1:[[]], 2:[[]], 3:[[]], 4:[[]], 5:[[]]}

        # Variable which contains the bus which has just reached a stop, the bus stop that it has reached, and the
        # stopping time required given the number of people alighting at this stop and those waiting to board
        self.decisionBus = ["bus.0", "stop1", 0]

        traci.start(self.sumoCmd)

        self.busStops = list(traci.simulation.getBusStopIDList()) # get the list of bus stops from the simulation
        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"] # get the list of buses from the simulation

        self.busCapacity = 85

        # dictionary containing those people who have a destination bus stop assigned
        self.personsWithStop = dict()

        self.stopTime = 0

        # stores the number of people on each bus which will stop at each stop
        self.peopleOnBuses = [[0]*12, [0]*12, [0]*12, [0]*12, [0]*12, [0]*12] 

        self.action_space = Discrete(3)

       
        # the observation space:
        # contains the stop which the bus has reached, the forward and backward headways of the bus, the number of persons waiting at each stop, 
        # the stopping time required according to the number of people boarding and alighting at this stop, the current maximum passenger waiting 
        # times at each bus stop, the numnber of passengers on the previous, current, and following buses, and the speed factors of the previous,
        # current and follwing buses
        if self.traffic != 0:
            self.low = np.array([0 for _ in range(len(self.busStops))] + [0, 0] +  [0 for _ in range(len(self.busStops))] + [0] + [0 for _ in range(len(self.busStops))] + [0, 0, 0] + [0, 0, 0], dtype='float32')
            self.high = np.array([1 for _ in range(len(self.busStops))] + [5320, 5320] + [float('inf') for _ in self.busStops] + [float('inf')] + [200000 for _ in self.busStops] + [85, 85, 85] + [1, 1, 1], dtype='float32')
        else:
            self.low = np.array([0 for _ in range(len(self.busStops))] + [0, 0] +  [0 for _ in range(len(self.busStops))] + [0] + [0 for _ in range(len(self.busStops))] + [0, 0, 0], dtype='float32')
            self.high = np.array([1 for _ in range(len(self.busStops))] + [5320, 5320] + [float('inf') for _ in self.busStops] + [float('inf')] + [200000 for _ in self.busStops] + [85, 85, 85], dtype='float32')

        self.observation_space = Box(self.low, self.high, dtype='float32')

        self.reward_range = (float('-inf'), 0)

        self.sdVal = 0
        
        self.dfLog = pd.DataFrame(columns=['meanWaitTime', 'action', 'dispersion', 'headwaySD'])

    # step function required by the gym environment
    # each each step signifies an arrival of a bus at a bus stop 
    def step(self, action):

        self.gymStep += 1
        print("GYM STEP: ", self.gymStep)
        
        self.logValues(action)

        #####################
        #   APPLY ACTION    #
        #####################
        
        # hold the bus
        if action == 0: 
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            # increase the stopping time of the vehicle by 15 seconds (hence holding the vehicle)
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=(self.decisionBus[2]+15))
            
            # UPDATE PEOPLE ON BUS

            # boarding
            personsOnStop = traci.busstop.getPersonIDs(self.decisionBus[1])
            # All persons on the stop can board the bus
            for person in personsOnStop:
                # set the decision bus as the bus which the person boarded 
                self.personsWithStop[person][1] = self.decisionBus[0] 
                # increment the number of passengers of the decision bus 
                self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] += 1 

            #alighting
            personsOnBus = traci.vehicle.getPersonIDList(self.decisionBus[0])
            # Not everyone on the bus may be alighting at this stop
            for person in personsOnBus:
                # check if passenger will alight at this stop
                if self.personsWithStop[person][0] == self.decisionBus[1]:
                    # decrement the number of passengers of the decision bus
                    self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] -= 1

        # skip the stop
        elif action == 1: 
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            # set the stopping duration to zero, hence skipping the stop
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=0)

        # else action == 2, no action taken and bus proceeds normally by letting passengers board and alight
        else:
            stopData = traci.vehicle.getStops(self.decisionBus[0], 1)
            # set the stopping time to the time required just to let passengers board and alight
            # notice that we do not increase the stopping duration as we did for the holding action 
            traci.vehicle.setBusStop(self.decisionBus[0], stopData[0].stoppingPlaceID, duration=self.decisionBus[2])

            #UPDATE PEOPLE ON BUS

            #boarding
            personsOnStop = traci.busstop.getPersonIDs(self.decisionBus[1])
            # All persons on the stop can board the bus
            for person in personsOnStop: 
                # set the decision bus as the bus which the person boards
                self.personsWithStop[person][1] = self.decisionBus[0]
                # increment the number of passengers of the decision bus
                self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] += 1

            #alighting
            personsOnBus = traci.vehicle.getPersonIDList(self.decisionBus[0])
            # Not everyone on the bus may be alighting at this stop
            for person in personsOnBus:
                # check if the passenger will alight at this stop
                if self.personsWithStop[person][0] == self.decisionBus[1]:
                    # decrement the number of passengers of the decision bus
                    self.peopleOnBuses[int(self.decisionBus[0][-1])][int(self.personsWithStop[person][0][-1])-1] -= 1


        ########################################
        #   FAST FORWARD TO NEXT DECISION STEP #
        ########################################

        # run the simulation until a bus has reached a stop.
        # the variable reachedStopBuses contains a list of all buses that have reached a stop at this
        # simulation step, with each element containing the bus and the stop it has reached.
        reachedStopBuses = self.reachedStop()
        while len(reachedStopBuses) < 1: # while no bus has reached a stop
            self.sumoStep()
            reachedStopBuses = self.reachedStop()


        ###### UPDATE DECISION BUS #######
        # we set the first bus in reachedStopBuses as the decision bus
        # calculate the stopping time required
        self.stopTime = self.getStopTime(reachedStopBuses[0][0], reachedStopBuses[0][1])
        self.decisionBus = [reachedStopBuses[0][0], reachedStopBuses[0][1], self.stopTime]



        ###############################################
        #   GET NEW OBSERVATION AND CALCULATE REWARD  #
        ###############################################

        state = self.computeState()

        reward = self.computeReward()

        
        # check if episode has terminated
        if self.gymStep > self.epLen:
            print("DONE, episode num: ", self.episodeNum)

            done = True

            # self.dfLog.to_csv('results/final/csvs/ppo/TrafficBunched.csv')


            # BUNCHING GRAPH
            colours = ['red', 'green', 'orange', 'blue', 'purple', 'black']
            labelled = [False for _ in range(6)]
            for y in range(0,6):
                for z in self.bunchingGraphData[y]:
                    x_values = []
                    y_values = []

                    for i in z:
                        x_values.append((i[0]*9)/60)
                        y_values.append(i[1])

                    if not labelled[y]:
                        plt.plot(x_values, y_values, color=colours[y], label='bus '+str(y))
                        labelled[y] = True
                    else:
                        plt.plot(x_values, y_values, color=colours[y])
            plt.yticks(range(1,13))
            plt.title("PPO with Traffic, Bunched")
            plt.xlabel('Time (mins)')
            plt.ylabel('Bus Stop')
            plt.legend(loc=4)
            # plt.savefig('results/final/bunching/ppo/TrafficBunchedBunching.jpg')
            plt.show()
            plt.clf()


            # PIE CHART showing the actions taken
            values = []
            labels = []
            actions = self.dfLog['action'].tolist()
            hold = actions.count('Hold') / len(actions)
            if hold != 0:
                values.append(hold)
                labels.append("Hold")
            skip = actions.count('Skip') / len(actions)
            if skip != 0:
                values.append(skip)
                labels.append("Skip")
            noAction = actions.count('No action') / len(actions)
            if noAction != 0:
                values.append(noAction)
                labels.append("No action")
    

            plt.pie(values, labels=labels, autopct='%1.1f%%')
            plt.title('Actions (PPO, Traffic, Bunched).jpg')
            # plt.savefig('results/final/actions/ppo/TrafficBunchedActions.jpg')
            plt.show()
            plt.clf()

        else:
            done = False

        info = {}

        return state, reward, done, info


    # reset function required by the gym environment 
    def reset(self):
        self.episodeNum += 1
        traci.close()
        
        if self.mixedConfigs: # choose the initial state of the environment (bunched or unbunched)
            if self.episodeNum % 2 == 0:
                self.config = 'sumo/traffic/ring.sumocfg'
            else:
                self.config = 'sumo/bunched/ring.sumocfg'

        self.sumoCmd = [self._sumoBinary, "-c", self.config, "--tripinfo-output", "tripinfo.xml", "--no-internal-links", "false", "--lanechange.overtake-right", "true"]
        if self.noWarnings:
            self.sumoCmd.append("--no-warnings")
        traci.start(self.sumoCmd)
        self.gymStep = 0
        self.stoppedBuses = [None for _ in range(numBuses)] 
        self.decisionBus = ["bus.0", "stop1", 0]
        self.personsWithStop = dict()
        self.peopleOnBuses = [[0]*12, [0]*12, [0]*12, [0]*12, [0]*12, [0]*12]

        self.stopTime = 0

        self.sdVal = 0

        self.bunchingGraphData = {0:[[]], 1:[[]], 2:[[]], 3:[[]], 4:[[]], 5:[[]]}

        if self.traffic == -1:
            self.lowestTrafficSpeed = random.randint(7,10)
        else:
            self.lowestTrafficSpeed = self.traffic

        # sumo step until all buses are in the simulation
        while len(traci.vehicle.getIDList()) < numBuses:
            self.sumoStep()

        self.buses = [bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]

        state = self.computeState()
        return state

    def close(self):
        traci.close()

    # function which returns a list of buses that have reached a stop
    def reachedStop(self):
        reached = []

        simTime = traci.simulation.getTime()

        for vehicle in traci.vehicle.getIDList():
            if vehicle[0:3] == "bus":
                for stop in self.busStops:
                    # if the bus is on the same lane as the stop
                    if traci.busstop.getLaneID(stop) == traci.vehicle.getLaneID(vehicle):
                        # check if the bus is within reach of the stop
                        if (traci.vehicle.getLanePosition(vehicle) >= (traci.busstop.getStartPos(stop) - 5)) and (traci.vehicle.getLanePosition(vehicle) <= (traci.busstop.getEndPos(stop) + 1)):
                            # the bus shouls be marked as a newly stopped bus only if it was not already marked as so in 
                            # the previous few time steps
                            if self.stoppedBuses[int(vehicle[-1])] == None:
                                # get stop id and update stopped bused list
                                self.stoppedBuses[int(vehicle[-1])] = stop
                                # add the bus to the list of newly stopped buses
                                reached.append([vehicle, stop])

                                # update bunching graph data
                                busNum = int(vehicle[-1])

                                if len(stop) == 5:
                                    s = int(stop[-1])
                                else:
                                    s = int(stop[-2:])
                                self.bunchingGraphData[busNum][-1].append((simTime, s))

                        else:
                            # update buses which have left a bus stop such that they are no longer marked as stopped
                            if self.stoppedBuses[int(vehicle[-1])] != None:
                                self.stoppedBuses[int(vehicle[-1])] = None
                                
                                # update bunching graoh data
                                busNum = int(vehicle[-1])
                    
                                if len(stop) == 5:
                                    s = int(stop[-1])
                                else:
                                    s = int(stop[-2:])   

                                self.bunchingGraphData[busNum][-1].append((simTime, s))

                                if s == 12:
                                    self.bunchingGraphData[busNum].append([])
    
        
        # calculate headway standard deviation
        if reached:
            headways = []
            for bus in traci.vehicle.getIDList():
                if bus[0:3] == "bus":
                    follower, leader = self.getFollowerLeader(bus=[bus])

                    forwardHeadway = self.getForwardHeadway(leader, bus)

                    backwardHeadway = self.getForwardHeadway(bus, follower)
                    headways.append(abs(forwardHeadway - backwardHeadway))

            average = sum(headways)/len(headways)
            deviations = [((headway - average)**2) for headway in headways]
            variance = sum(deviations) / len(headways)
            sd = math.sqrt(variance)

            self.sdVal = sd
        
        return reached


    def sumoStep(self):
        traci.simulationStep() # run the simulation for 1 step
        self.updatePersonStop() # update the stops corresponding to each person 
        # update the passengers on board only if all buses are currently in the simulation
        if len([bus for bus in traci.vehicle.getIDList() if bus[0:3] == "bus"]) == numBuses:
            self.updatePassengersOnBoard()

        simTime = traci.simulation.getTime()

        if self.traffic != 0:
            if simTime % 15 == 0: # add a new car in the simulation
                traci.vehicle.add('car'+str(simTime), 'traffic', typeID='traffic')
                repeats = random.randint(1,3) # randomly choose the number of times the new car will loop around the bus corridor
                # # build the new route of the car according to the number of repeats
                newRoute = ['E0']
                for _ in range(repeats):
                    newRoute.extend(['5','6','7','8','9','10','11','0','1','2','3','4'])
                newRoute.extend(['5','6','7','8','9','E1'])
                traci.vehicle.setRoute('car'+str(simTime), newRoute)

                # randomly set the speed of the new car
                speeds = [self.lowestTrafficSpeed, 20, 30, 50]
                speed = random.randint(0,3)
                traci.vehicle.setSpeed('car'+str(simTime), speeds[speed])

        # traci.vehicle.highlight('car1', color=(255,0,255), size=30)


    # function which computes the state required by the gym environment
    # The state that is returned contains the stop which the bus has reached, the forward and backward headways, the number of persons waiting at each stop,
    # the stopping time required according to the number of people boarding and alighting at this stop, the current maximum passenger waiting
    # times at each bus stop, and the number of passengers on the previous, currentm and following buses
    def computeState(self):
        stop = self.oneHotEncode(self.busStops, self.decisionBus[1])
        bus = self.oneHotEncode(self.buses, self.decisionBus[0])

        headways = self.getHeadways()
        
        waitingPersons = self.getPersonsOnStops()

        maxWaitTimes = self.getMaxWaitTimeOnStops()

        numPassengers = self.getNumPassengers()

        speedFactors = self.getSpeedFactors()
        
        if self.traffic != 0:
            state = stop + headways + waitingPersons + [self.stopTime] + maxWaitTimes + numPassengers + speedFactors
        else:
            state = stop + headways + waitingPersons + [self.stopTime] + maxWaitTimes + numPassengers

        return state

    def oneHotEncode(self, list, item):
        return [1 if i == item else 0 for i in list]

    # function which returns the forward headway of a given bus (follower)
    def getForwardHeadway(self, leader, follower):
        # number of edges in the ring network simulation
        numEdges = 12
        leaderRoad = int(traci.vehicle.getRoadID(leader))
        followerRoad = int(traci.vehicle.getRoadID(follower))

        # both buses are on the same edge and the leader is in front of the follower.
        # just return the distance between the position of both buses
        if leaderRoad == followerRoad: 
            if traci.vehicle.getLanePosition(leader) - traci.vehicle.getLanePosition(follower) > 0:
                return traci.vehicle.getLanePosition(leader) - traci.vehicle.getLanePosition(follower)
        

        # otherwise, we must compute the length of all lanes between the two buses.
       
        # first find the remaining distance of the lane on which the follower currently is 
        h = traci.lane.getLength(traci.vehicle.getLaneID(follower)) - traci.vehicle.getLanePosition(follower)

        # calculate the number of edges between the those on which the two buses are
        if leaderRoad == followerRoad:
            repeats = numEdges - 1
        elif leaderRoad > followerRoad:
            repeats = leaderRoad - followerRoad - 1
        else:
            repeats = (numEdges - (abs(leaderRoad - followerRoad))) - 1
        
        # add the length of each edge in between the edges on which the two buses currently are
        for i in range(repeats):
            lane = int(traci.vehicle.getRoadID(follower)) + i + 1
            if lane >= numEdges:
                lane = lane % numEdges

            h += traci.lane.getLength(str(lane)+"_0")

        # finally, add the portion of the leader's lane already driven  
        h+= traci.vehicle.getLanePosition(leader)

        return h
            
    # function which returns the id of the leader and follower buses of the decision bus
    def getFollowerLeader(self, bus=[]):
        if bus:
            b = bus[0][-1]
        else:
            b = self.decisionBus[0][-1]
        
        # if the decision bus is the last bus, then the follower is the first bus, hence it is set to zero
        if int(b) + 1 == len(self.buses):
            follower = "bus.0"
        # otherwise just increment the bus number
        else:
            follower = "bus." + str(int(b) + 1)

        # if the decision bus is the first bus, then the leader is the last bus, hence set to the number of buses minus 1
        if int(b) == 0:
            leader = "bus." + str(len(self.buses) - 1)
        # otherwise just decrement the bus number
        else:
            leader = "bus." + str(int(b) - 1)

        return follower, leader

    # function which returns the forward and backward headways of the decision bus
    def getHeadways(self):
        if len(self.buses) > 1:
            # get the follower and leader of the decision bus
            follower, leader = self.getFollowerLeader()
            
            # get the forward headway of the decision bus
            forwardHeadway = self.getForwardHeadway(leader, self.decisionBus[0])

            # get the backward headway of the decision bus.
            # in this case, we are in reality finding the forward headway of the follower to the decision bus which
            # is the same as the backward headway of the decision bus to its follower
            backwardHeadway = self.getForwardHeadway(self.decisionBus[0], follower)
         
            return [forwardHeadway, backwardHeadway]

        else:
            return [0, 0]

    # function which returns the number of people waiting on each stop in the network
    def getPersonsOnStops(self):
        persons = [traci.busstop.getPersonCount(stop) for stop in self.busStops]
        return persons

    # function which returns the maximum passenger waiting time of each stop in the network
    def getMaxWaitTimeOnStops(self):
        maxWaitTimes = []
        for stop in self.busStops:
            personsOnStop = traci.busstop.getPersonIDs(stop)
            waitTimes = [traci.person.getWaitingTime(person) for person in personsOnStop]
            # check if there are actually people waiting on the stop
            if len(waitTimes) > 0:
                maxWaitTimes.append(max(waitTimes))
            # if no people are waiting on the stop, then the max wait time of this stop is set to zero
            else:
                maxWaitTimes.append(0)

        return maxWaitTimes

    # function which returns the number of passengers on the leader bus, decision bus, and follower bus
    def getNumPassengers(self):
        follower, leader = self.getFollowerLeader()

        numPassengers = [traci.vehicle.getPersonNumber(leader), traci.vehicle.getPersonNumber(self.decisionBus[0]), traci.vehicle.getPersonNumber(follower)]
        return numPassengers

    # function which computes the reward required by the gym environment and rl algorithm
    def computeReward(self):

        headways = self.getHeadways()

        forward = headways[0]
        backward = headways[1]

        reward = -abs(forward - backward)

        return reward   


    # function which randomly assigns a destination bus stop to persons yet without a destination
    def updatePersonStop(self):
        persons = traci.person.getIDList()
        # get list of persons curently without destination
        personsWithoutStop = [person for person in persons if person not in self.personsWithStop]
        for person in personsWithoutStop:            
            # assign a random bus stop from the following six bus stops as the destination (as is done in the paper by Wang and Sun 2020)
            num = random.randint(1,6)
            edge = traci.person.getRoadID(person)
            newEdge = (int(edge) + num) % 12
            newStop = newEdge + 1
            stop = "stop"+str(newStop)
            traci.person.appendDrivingStage(person, str(newEdge), "line1", stopID=stop) 
            traci.person.appendWalkingStage(person, [str(newEdge)], 250) 
            # add the person to the persons with an assigned stop
            self.personsWithStop[person] = [stop, None]
            
    # function which determines the dwell time of a bus at a stop based on the number of passengers boarding and alighting, using the boarding and alighting rates
    def getStopTime(self, bus, stop):

        # the number of people on the bus stop waiting to board the bus
        boarding = traci.busstop.getPersonCount(stop)
        
        # the number of passengers on this bus that will alight at this stop
        alighting = self.peopleOnBuses[int(bus[-1])][int(stop[-1])-1]

        # calculate dwell time according to the boarding and alighting rates in the paper by Wang and Sun 2020
        time = max(math.ceil(boarding/3), math.ceil(abs(alighting)/1.8)) #abs is there just in case is falls below zero if a person should've left a bus but the simulation did not give them time
        
        return time

    # For any passengers which board the bus during holding time and thus were not know beforehand that they would board
    def updatePassengersOnBoard(self): 
        for bus in self.buses:
            for person in traci.vehicle.getPersonIDList(bus):
                # check if passenger does not yet have a bus assigned to them
                if self.personsWithStop[person][1] == None:
                    # assign the bus to the passenger
                    self.personsWithStop[person][1] = bus 
                    # increment number of passengers of the particular bus
                    self.peopleOnBuses[int(bus[-1])][int(self.personsWithStop[person][0][-1])-1] += 1

    # function which returns the speed factor (speed of bus/speed without traffic) of the leader bus, decision bus, and follower bus
    def getSpeedFactors(self):
        follower, leader = self.getFollowerLeader()

        speedFactors = [traci.vehicle.getSpeed(leader)/20, traci.vehicle.getSpeed(self.decisionBus[0])/20, traci.vehicle.getSpeed(follower)/20]
        return speedFactors

    # logging the values in the dataframe
    def logValues(self, action):
        maxWaitTimes = self.getMaxWaitTimeOnStops()    
        mean = sum(maxWaitTimes)/len(maxWaitTimes)
        
        actions = ['Hold', 'Skip', 'No action']

        occDisp = self.occupancyDispersion()

        self.dfLog = pd.concat([self.dfLog, pd.DataFrame.from_records([{'meanWaitTime':mean, 'action':actions[action], 'dispersion':occDisp, 'headwaySD':self.sdVal}])], ignore_index=True)

    # occupancy dispersion as calculated in Wang and Sun (2020), using a variance to mean ratio
    def occupancyDispersion(self):
        passengers = []
        for bus in self.buses:
            passengers.append(traci.vehicle.getPersonNumber(bus))

        average = sum(passengers)/len(passengers)
        if average == 0:
            return 0

        deviations = [((p - average)**2) for p in passengers]
        variance = sum(deviations) / len(passengers)

        occDisp = variance / average

        return occDisp
