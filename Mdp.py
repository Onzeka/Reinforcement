#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: enzo
"""


import abc
from copy import deepcopy


class Mdp(metaclass=abc.ABCMeta):
    
    """
    
    A Mdp represents a markov decision process. 
    Mdp is a general abstract superclass for specific MDP (see EnergyHarvestingMdp)
    
    Inner classes
    -------------
    Action : 
        represents an action of the MDP
    State : 
        represents a state of the MDP
    
    Methods
    ------------
    __init__:
        object constructor
    states_generation:
        compute all the MDP states
    compute_probability_matrix:
        compute the probability matrix
    process:
        return the reward and the next state when an action a is applied in the current_state
    
    Attributes
    ----------
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
            
    """
    
    def __init__(self, states_attributes, actions_attributes, actions_set, **mdp_parameters):
        """ 
        object constructor
        
        Parameters
        ----------
        states_attributes : list of strings
            Name of the states attributes.
        actions_attributes : list of strings
            Name of actions attributes.
        actions_set : list of tuple
            Each tuple in the list corresponds to one action.
            Each value in a tuple corresponds to an attribute value of the considered action
            following the same order than action_attribute
        **mdp_parameters : *
            additionnal variables for computation.

        Returns
        -------
        None.

        """
        self.states_attributes = states_attributes
        self.actions_attributes = actions_attributes
        self.actions_set = []
        for action in actions_set:
            self.actions_set += [self.Action(self,*action)]
        for attr_name, attr_value in mdp_parameters.items():
            setattr(self, attr_name, attr_value)
        self._states = None
        self._probability_matrix = None
        self._is_set = False
    
        
    class Action:
        """
        Action represent an Action of the mdp
        
        Methods
        -------------
        __init__:
            object contructor
        compute_hashcode:
            compute the hashcode of the object
        """
            
        def __init__(self,outter_instance, *args):
            """
            object constructor

            Parameters
            ----------
            outter_instance : Mdp
                MDP of which this action is a part of
            *args : *
                values of this action attributes

            Returns
            -------
            None.

            """
            self.outter_instance = outter_instance
            for attr_name,attr_value in zip(outter_instance.actions_attributes,args):
                setattr(self, attr_name, attr_value)
            self.hash_code = self.compute_hashcode()
            
        
        def compute_hashcode(self):
            return hash(str(self))
        
        def __hash__(self):
            return self.hash_code
        
        def __eq__(self,other):
            return hash(self) == hash(other)
        

    class State:
        """
        State represent a state of the mdp
        
        Methods
        -------------
        __init__:
            object contructor
        compute_hashcode:
            compute the hashcode of the object
        """
        def __init__(self,outter_instance, *args):
            self.outter_instance = outter_instance
            for attr_name,attr_value in zip(outter_instance.states_attributes,args):
                setattr(self, attr_name, attr_value)
            self.hash_code = self.compute_hashcode()
            
        def compute_hashcode(self):
            return hash(str(self))
                        
        def __hash__(self):
            return self.hash_code
        
        def __eq__(self,other):
            return hash(self) == hash(other)
        
        def __deepcopy__(self, memo):
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                if k != 'hash_code':
                    setattr(result, k, deepcopy(v, memo))
            setattr(result,'hash_code',self.compute_hashcode())
            return result
        
    @property
    def states(self):
        if self._states == None:
            self.states_generation()
        return self._states

    @property
    def probability_matrix(self):
        if  not self._is_set:
            self.compute_probability_matrix()
            self._is_set = True
        return self._probability_matrix
    
    
    @abc.abstractmethod
    def process(self, current_state,action):
        """
        return the reward and the next state when an action a is applied in the current_state 

        Parameters
        ----------
        current_state : State
            current state of the MDP
        action : Action
            Action selected at this state.

        Raises
        ------
        NotImplementedError

        Returns
        -------
        reward,next_state
        *,State

        """
        raise NotImplementedError
    @abc.abstractmethod
    def states_generation(self):
        """
        return all the states of the Mdp

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        list of State

        """
        raise NotImplementedError
    @abc.abstractmethod
    def compute_probability_matrix(self):
        """
        compute all the probability transition of the MDP

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        3D numpy array probability_matrix[j][i][k] corresponds to the probability to
        arrive in state j from state i using action k

        """    
        raise NotImplementedError
    @abc.abstractmethod
    def mean_action_reward(self,state,action): 
        """
        return the mean reward of an action considering the current state

        Parameters
        ----------
        state : State
            current state
        action : Action
            action

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        mean_reward: *

        """        
        raise NotImplementedError
                
        
    
    