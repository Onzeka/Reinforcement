#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: enzo
"""
from Mdp import Mdp
import math
import random
from copy import deepcopy
import sys
from scipy.special import gammaincc 
import numpy as np
import time

class EnergyHarvestingMdp(Mdp):
    """
    Mdp modelling of the Energy harvesting issue
    
    Inner classes
    -------------
    Action:
        represents an action of the MDP
    State:
        represents a state of the MDP
    
    Methods
    -------------
    __init__:
        object constructor
    states_generation:
        compute all the MDP states
    compute_probability_matrix:
        compute the probability matrix
    process:
        return the reward and the next state when an action a is applied in the current_state
    buffersgen:
        return a list of all possible buffer states
    transmission_power:
        compute the transmission power considering the current state an action and a canal state
    energy_consumed:
        compute the energy consumed by an action 
    is_action_valid:
        return if an action is possible or not
    e_arrival:
        modelize an energy arrival
    p_arrival:
        modelize a packet arrival
    Q:
        gamma function
    transition_probability:
        compute the transition probability considering the current state the action taken and a next state
    
    Attributes
    -----------
    states_attributes: list of strings
        states attributes
    actions_attributes: list of strings
        action attributes
    action_set: list of Mdp.Action
        list of action 
    states: list of Mdp.State
        list of states
    probability_matrix: 3D np array
        3D matrix probability_matrix[j][i][k] corresponds to the probability to
        arrive in state j from state i using action k
    buffer_capacity : int
        buffer capacity, how many packets can be stored simultaneously
    battery_capacity: int
        how many energy units can be stored 
    maximum_delay: int
        how long a packet can stay in the buffer
    canal_states : list of float
        define the different canal capacity
    ul_bandwith:
        uplink bandwith
    dl_bandwith:
        downlink bandwith
    packet_size_ul:
        uplink packets' size
    packet_size_dl:
        downling packets'size
    time_slot:
        How long last one iteration of our mdp
    time_station:
        processing time of base station
    power_local:
        power used for local processing
    power_mobile_station:
        transmission power of the mobile device
    power_station:
        power used by the base station to send the result
    power_max:
        max power consumable for one iteration 
    power_transmitter:
        power of the transmitter
    noise_spectral_density:
        noise spectral density 
    epsilone_u:
    lambdae : 
        energy arrival rate 
    lambdad :
        data arrival rate
    
    """
    
    def __init__(self, parameters):
        self.max_packets_local = 2 # à changer
        self.max_packets_offload = 4 # à changer 
        states_attributes = ['buffer','battery','canal_state']
        actions_attributes = ['action_type','processed_packets']
        actions_set = []
        actions_set += [('idle',0)]
        for i in range(1,self.max_packets_local+1):
            actions_set += [('local',i)]
        for i in range(1,self.max_packets_offload+1):
            actions_set += [('offload',i)]
        
        super().__init__(states_attributes,actions_attributes, actions_set, buffer_capacity = parameters['buffer_capacity'],
                       battery_capacity = parameters['battery_capacity'], maximum_delay = parameters['maximum_delay'],
                       canal_states = parameters['canal_states'], ul_bandwith = parameters['ul_bandwith'],
                       dl_bandwith = parameters['dl_bandwith'], packets_size_ul = parameters['packets_size_ul'],
                       packets_size_dl = parameters['packets_size_dl'], time_slot = parameters['time_slot'],
                       time_station = parameters['time_station'], power_local = parameters['power_local'],
                       power_mobile_device = parameters['power_mobile_device'],
                       power_station = parameters['power_station'], power_max = parameters['power_max'],
                       power_transmitter = parameters['power_transmitter'],
                       noise_spectral_density = parameters['noise_spectral_density'], epsilone_u = parameters['epsilone_u'],
                       lambdae = parameters['lambdae'], lambdad = parameters['lambdad'])
        
    class State(Mdp.State):
        def __init__(self,outter_instance,buffer, battery, canal_state):
            super().__init__(outter_instance, buffer, battery, canal_state)
            self.queue = 0
            self.critical_packets = 0
            
            for packet in self.buffer :
                if packet == -1 :
                    break
                else:
                    if packet == self.outter_instance.maximum_delay:
                        self.critical_packets += 1
                    self.queue += 1
            
        def __str__(self):
            return ("buffer: " + str(self.buffer) + "\nbatterie: " + "["+"*"*self.battery +"-"*
                    (self.outter_instance.battery_capacity - self.battery)+"]"+
                    "\n canal_state: " + str(self.canal_state) )
            
            
        
    class Action(Mdp.Action) :
        def __str__(self):
            if self.action_type == "idle" :
                string = self.action_type
            else :
                string = self.action_type +" "+ str(self.processed_packets) + " packets"
            return string
        
    

    def buffersgen (maximum_delay,buffer_capacity):
    
        for i in range(-1,maximum_delay+1):
            if buffer_capacity == 1 :
                yield [i]
            else:
                for j in EnergyHarvestingMdp.buffersgen(i,buffer_capacity-1):
                    yield [i]+j
                
    def states_generation(self):
        print('computing states')
        self._states = []
        maximum_delay = self.maximum_delay
        buffer_capacity = self.buffer_capacity
        for k in EnergyHarvestingMdp.buffersgen (maximum_delay,buffer_capacity):
          
            for x in range(len(self.canal_states)) :
                for b in range(self.battery_capacity+1):
                    self._states+=[self.State(self,k,b,self.canal_states[x])]
        print('done')
        
    
    def transmission_power(self, action, state = None, canal_state = None):
        if state != None:
            canal_state = state.canal_state
        transmission_power = 0
        if action.action_type == 'offload':
            x = canal_state
            u = action.processed_packets
            L = self.packets_size_ul
            Wul = self.ul_bandwith
            Ts = self.time_slot
            Tw = self.time_station
            Ldl = self.packets_size_dl
            Wdl = self.dl_bandwith
            Ps = self.power_station
            N0 = self.noise_spectral_density
            A =(L/Wul)/((Ts/u)-(Tw+Ldl/(Wdl*math.log(1+(Ps*x)/(Wdl*N0),2))))
            transmission_power = ((2**A)-1)*(Wul*N0/x)  
        return transmission_power
    
    def energy_consumed(self, action, transmission_power, state = None, canal_state = None):
        if state != None:
            canal_state = state.canal_state
        action_type = action.action_type
        if action_type == 'idle':
            energy_consumed = 0
        elif action_type == 'local':    
            u = action.processed_packets
            Pl = self.power_local
            Ts = self.time_slot
            EU = self.epsilone_u
            energy_consumed = math.ceil(u*Pl*Ts/EU)
        elif action_type == 'offload':
            x = canal_state
            u = action.processed_packets
            L = self.packets_size_ul
            Wul = self.ul_bandwith
            Tw = self.time_station
            Ldl = self.packets_size_dl
            Wdl = self.dl_bandwith
            Ps = self.power_station
            N0 = self.noise_spectral_density
            Pt = transmission_power
            Pr = self.power_mobile_device
            EU = self.epsilone_u
            Pw = self.power_transmitter
            A = L*Pt/(Wul*math.log(1+((Pt*x)/(Wul*N0)),2))
            B = Ldl*Pr/(Wdl*math.log(1+((Ps*x)/(Wdl*N0)),2))
            energy_consumed = math.ceil((u/EU)*(A+Tw*Pw+B))
        return energy_consumed
    
    
    def is_action_valid(self, current_state, action):
        action_type = action.action_type
        processed_packets = action.processed_packets
        if action_type == 'idle':
            valid_action = True
        elif processed_packets > current_state.queue: # peut etre endroit à modifier
                valid_action = False
        else :
                transmission_power = self.transmission_power(action, state = current_state)
                energy_consumed = self.energy_consumed(action,transmission_power,state = current_state)
                #print("energyconsumed: ",energy_consumed)
                if energy_consumed > current_state.battery:
                    valid_action = False
                elif transmission_power > self.power_max:#peut etre à modifier
                    valid_action = False
                else:
                    valid_action = True
        return valid_action
    
    def e_arrival(self):
        lambda_arrival = self.lambdae
        p_cumul = 0
        u = random.uniform(0,1)
        result = 0
        while True:
            p_cumul += math.exp(-lambda_arrival)*((lambda_arrival**result)/math.factorial(result)) 
            if u <= p_cumul :
                break
            result += 1
        return result
    def p_arrival(self):
        lambda_arrival = self.lambdad
        p_cumul = 0
        u = random.uniform(0,1)
        result = 0
        while True:
            p_cumul += math.exp(-lambda_arrival)*((lambda_arrival**result)/math.factorial(result)) 
            if u <= p_cumul :
                break
            result += 1
        return result
                                    
    def process(self,current_state, action):
        valid_action = self.is_action_valid(current_state, action)
        if valid_action:
            processed_packets = action.processed_packets
        
            next_state = deepcopy(current_state)
            energy_arrival = self.e_arrival()
            incoming_packets = self.p_arrival()
        
            delay_discarded_packet = max(0, current_state.critical_packets - processed_packets)
            leaving_packets = max(processed_packets,current_state.critical_packets)#wn
            buffer_overflow_discarded_packets = max(0,incoming_packets + current_state.queue - leaving_packets -
                                                    self.buffer_capacity)
            packets_arrival = incoming_packets - buffer_overflow_discarded_packets#an+1
            
            next_state.critical_packets = 0
            for i in range(current_state.queue - leaving_packets):
                next_state.buffer[i] = current_state.buffer[i+leaving_packets]+1
                if next_state.buffer[i] == self.maximum_delay:
                    next_state.critical_packets += 1
            for i in range(current_state.queue - leaving_packets, current_state.queue - leaving_packets + packets_arrival):
                next_state.buffer[i] = 0
            for i in range(current_state.queue - leaving_packets + packets_arrival, self.buffer_capacity):
                next_state.buffer[i] = -1
            
            energy_consumed = self.energy_consumed(action,self.transmission_power(action, state = current_state), state = current_state)
        
            next_state.canal_state = self.canal_states[random.randint(0,len(self.canal_states)-1)] #à modifier peut etre
            next_state.queue += packets_arrival - leaving_packets
            next_state.battery = min(self.battery_capacity, current_state.battery - energy_consumed + energy_arrival)
            discarded_packets = delay_discarded_packet + buffer_overflow_discarded_packets
        
        else :
            next_state = current_state
            discarded_packets = sys.maxsize
        return discarded_packets, next_state, valid_action
     
        
    def Q(a,b):
        r = 0
        for i in range(a):
            r+= (b**i)/(math.factorial(i))
        return math.exp(-b)*r
        
    def transition_probability(self,next_state,current_state,action, energy_tab = None, power_tab = None):
        probability = 0
        if power_tab != None:
            transmission_power = power_tab[action][current_state.canal_state]
        else:
            transmission_power = self.transmission_power(action,state = current_state)
        
        if energy_tab != None:
            energy_consumed = energy_tab[action][current_state.canal_state]
        else:
            energy_consumed = self.energy_consumed(action,transmission_power, state = current_state)
       
        leaving_packets = max(action.processed_packets,current_state.critical_packets)
        
        if (energy_consumed > current_state.battery or next_state.battery < current_state.battery - energy_consumed
            or action.processed_packets > current_state.queue or next_state.queue < current_state.queue - leaving_packets
            or transmission_power > self.power_max):
            probability = 0
        else:
            indice = True
            for i in range(self.buffer_capacity):
                if next_state.buffer[i] > current_state.buffer[i]+1:
                    indice = False
                    break
                elif i + leaving_packets < self.buffer_capacity:
                    if ((next_state.buffer[i] != current_state.buffer[i+leaving_packets]+1 and 
                         current_state.buffer[i+leaving_packets] != -1) or (next_state.buffer[i]>0 and 
                         current_state.buffer[i+leaving_packets] == -1 )):
                        indice = False
                        break
                elif (current_state.queue == self.buffer_capacity and i >= current_state.queue - leaving_packets and
                      leaving_packets != 0 and next_state.buffer[i] > 0):
                    indice = False
                    break
            if indice :
                if next_state.queue < self.buffer_capacity :
                    lambdad = self.lambdad
                    q2 = next_state.queue
                    q1 = current_state.queue
                    w = leaving_packets
                    p_k = math.exp(-lambdad)*(lambdad**(q2-q1 + w))/math.factorial(q2-q1+ w)
                else : 
                    lambdad = self.lambdad
                    q1 = current_state.queue
                    w = leaving_packets
                    p_k = (1 - EnergyHarvestingMdp.Q(self.buffer_capacity - q1 + w,lambdad))
                if next_state.battery < self.battery_capacity:
                    lambdae = self.lambdae
                    b2 = next_state.battery
                    b1 = current_state.battery
                    p_b = math.exp(-lambdae)*(lambdae**(b2-b1 + energy_consumed))/math.factorial(b2-b1 + energy_consumed)
                else :
                    b1 = current_state.battery
                    lambdae = self.lambdae
                    p_b = (1 - EnergyHarvestingMdp.Q(self.battery_capacity - b1 + energy_consumed,lambdae))
                probability = p_b*p_k*1/len(self.canal_states)# à modifier
            else:
                probability = 0
                
        return probability
    
    def compute_probability_matrix(self):
        print('computing probability_matrix')
        power_tab = {}
        energy_tab = {}
        for action in self.actions_set:
            power_tab[action] = {}
            energy_tab[action] = {}
            for x in self.canal_states:
                power_tab[action][x] = self.transmission_power(action,canal_state = x)
                energy_tab[action][x] = self.energy_consumed(action,power_tab[action][x],canal_state = x)
        states = self.states
        self._probability_matrix = np.zeros((len(states),len(states),len(self.actions_set)))
        for j,next_state in enumerate(states) :
            for i, current_state in enumerate(states):
                for k,action in enumerate(self.actions_set): 
                    self._probability_matrix[j][i][k] = self.transition_probability(next_state,current_state,
                                            action,power_tab = power_tab, energy_tab = energy_tab)
        
        
            
    def mean_action_reward(self,state,action):
        ed = 0
        eo = 0
        if state.critical_packets != 0 and state.critical_packets > action.processed_packets:
            ed = state.critical_packets - action.processed_packets
        else :
            ed = 0
        
        qn = state.queue
        lambdad = self.lambdad
        wn = max(state.critical_packets,action.processed_packets)
        Bd = self.buffer_capacity
        eo = lambdad*(1- gammaincc(Bd - qn + wn ,lambdad)) + (qn - wn - Bd)*(1-gammaincc(Bd - qn + wn +1,lambdad))     
        return eo + ed
                
            
    