# Q_pandas_utilities.py # helper functions for Qtables

import sys
sys.path.append('/home/pi/.local/lib/python3.6/site-packages/pandas')
import pandas as pd
import itertools
import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pickle


def find_nearest(array, value):
    #print(array,value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def random_state(states):
    return random.choice(states[:]) 

def random_reward():
    return random.uniform(0, 100)
  
def three_states_three_actions(state1,state2,state3,action1,action2,action3):
    states = list(itertools.product(state1,state2,state3)) # Create a list of every state combination
    actions = list(itertools.product(action1,action2,action3)) # Create a list of every action combination
    return states,actions


def create_zeros_Qtable(states,actions):
     return (pd.DataFrame(np.zeros((len(states), len(actions))), index=states, columns=actions, dtype=float))
     
def create_random_Qtable(states,actions,lowerbound,upperbound,round_to):
    return (pd.DataFrame(np.random.uniform(lowerbound, upperbound, size=(len(states), len(actions))).round(round_to), index=states, columns=actions, dtype=float))

def save_Qtable(Qtable,path,filename):
    Qtable.to_pickle(path+filename)
    print('Qfunction is saved')
    return()

def load_Qtable(path,filename):
    print('Qfunction is loaded')
    return pd.read_pickle(path+filename)

def exponential_decay(start,end,maxsteps):
    return math.exp(math.log(end/start)/maxsteps)
    
def Update_Qtable(Qtable,actions,state,action,new_state,reward,alpha,gamma):
    Vnext = Qtable[actions].loc[[(new_state)]][:].max() # Max Q at new state
    Vmax=max(Vnext)
    Q=Qtable[action].loc[[(state)]][:].max()  # Past Qvalue of state-action pair
    Qtable[action].loc[[state]] = (1-alpha)*Q+alpha*(reward+Vmax*gamma)
    return(Qtable)

def choose_action(Qtable,state,epsilion,actions):
    if random.uniform(0, 1) > epsilion:
        action = Qtable.idxmax(axis=1).loc[[state]][0]
    else: 
        action = random.choice(actions)
    return(action)
  
def get_Qtable_policy(Qtable):
   Qtable['policy'] = Qtable.idxmax(axis=1)
   policy = Qtable['policy']
   return(policy)

def get_policy_action(what_policy,current_state):
    policy_action = what_policy[current_state]
    return(policy_action)

    
