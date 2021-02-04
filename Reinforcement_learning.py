#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: enzo
"""
import random
from termcolor import colored
import numpy as np
import sys

class Reinforcement_learning:
    
    def __init__(self, mdp):
        self.mdp = mdp
        
        #R-learning variable:
        self.r_tab = {}
        self.r_mean_reward = None
    
    def e_greedy(self,current_state,tab,i):
        e = 0.9999**i
        u = random.uniform(0,1)
        if u <= e :
            optimal = False #indique si l'on choisit une action optimale selon notre estimation, vaut True si l'on explore
            action = random.choice(self.mdp.actions_set)
        else :
            optimal = True
            action = min(tab[current_state], key = tab[current_state].get)
        return action, optimal
    
    def init_zeros(self,tab):
        for state in self.mdp.states:
            tab[state] = {}
            for action in self.mdp.actions_set:
                tab[state][action] = 0
    
    def alpha(n):
        return 1/n
    
    def beta(n):
        return 1/n
    
    #à optimiser en utilisant des np arrays peut etre ? 
    def r_learning(self,total_iteration, starting_states, choose_action = e_greedy,
                   init = init_zeros, min_or_max = min, alpha = alpha):
        init(self,self.r_tab)
        self.r_mean_reward = 0
        alpha_nsa_dict = {}
        a = 0
        for current_state in starting_states :
            a +=1
            print(a)
            init(self,alpha_nsa_dict)
            beta_count = 0
            print(current_state)
            iteration = 0
            while iteration < total_iteration:
                action,optimal = choose_action(self,current_state,self.r_tab,iteration)
                alpha_nsa_dict[current_state][action] += 1
                
                reward, next_state, valid_action = self.mdp.process(current_state,action)
                if not valid_action:
                    print(colored(action,'red'))
                if valid_action:
                    print('packet_loss: ', reward )
                    print(action)
                    
                    print("==============================================================")
                    print(next_state)
                    print(next_state.critical_packets)
                    
                    dt  = (reward - self.r_mean_reward + min_or_max(self.r_tab[next_state].values()) - 
                           self.r_tab[current_state][action] )
                    self.r_tab[current_state][action] += alpha(alpha_nsa_dict[current_state][action])*dt
                    if optimal:
                        beta_count += 1
                        self.r_mean_reward +=  Reinforcement_learning.beta(beta_count)*dt
                    iteration += 1
                    current_state = next_state
                else: 
                    self.r_tab[current_state][action] = reward # sur de ça ?
                    
                    
    # Policy_iteration funchions
    def get_transition_matrix(self,policy):
        states = self.mdp.states
        transition_matrix = np.zeros((len(states),len(states)))
        for i in range(len(states)):
            for j in range(len(states)):
                for k in range(len(self.mdp.actions_set)):
                    transition_matrix[i][j] += self.mdp.probability_matrix[j][i][k]*policy[i][k]
        return transition_matrix
    
    def cost(self,policy):
        states = self.mdp.states
        cost_vector = np.zeros(len(states))
        for i,state in enumerate(states):
            for k,action in enumerate(self.mdp.actions_set):
                cost_vector[i] += policy[i][k]* self.mdp.mean_action_reward(state,action)
        return cost_vector
    
    def random_policy(self):
        policy = np.zeros((len(self.mdp.states),len(self.mdp.actions_set)))
        for i in range(len(self.mdp.states)):
            policy[i][random.randint(0,len(self.mdp.actions_set)-1)] = 1
        return policy
    
    def solve(transition_matrix,cost_vector):
        # on se ramène à un systeme matriciel
        M = np.identity(len(cost_vector))-transition_matrix
        M = np.insert(M,0,1,axis = 1)
        tmp = np.ones(len(cost_vector))
        tmp = np.insert(tmp,0,0)
        M = np.insert(M,len(M),tmp,axis = 0)
        b = np.insert(cost_vector,len(cost_vector),0)
        #on résoud maintenant un systeme de la forme My=b
        Y = np.linalg.solve(M,b)
        return Y[1],Y[1:]
    
            
    def policy_iteration(self,e): #critère moyen
        #1
        policy = self.random_policy()
        
        beta1 = 0
        while True:
            #2
            print('transition_matrix_computation...')
            transition_matrix = self.get_transition_matrix(policy)
            print("done")
            cost_vector = self.cost(policy)
            #3
            print('solving bellman equation...')
            beta2,val = Reinforcement_learning.solve(transition_matrix,cost_vector) # partie à modifier pour ajouter d'autres critères
            policy2 = np.zeros((len(self.mdp.states),len(self.mdp.actions_set)))
            print(beta2)
            for i, current_state in enumerate(self.mdp.states):
                cmin = sys.maxsize
                kmin = 0
                for k,action in enumerate(self.mdp.actions_set):
                    if self.mdp.probability_matrix[:,i,k].any() !=0:
                        c = self.mdp.mean_action_reward(current_state,action)
                        for j, next_state in enumerate(self.mdp.states):
                            c+= self.mdp.probability_matrix[j][i][k]*val[j]
                        if c < cmin :
                            cmin = c
                            kmin = k
                policy2[i][kmin] = 1
            print(abs(beta2-beta1))
            if abs(beta2-beta1) < e:    
                break
            if (policy == policy2).all():
               policy = policy2
               break
            policy = policy2
            beta1 = beta2
        #policy_dict = {state:{action: policy[i][k] for k,action in enumerate(self.mdp.actions_set)} for i,state in enumerate(self.mdp.states)}  
        return policy  
                    
                    