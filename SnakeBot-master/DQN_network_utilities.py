#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io, os, sys, types
#from IPython import get_ipython
#from nbformat import read
#from IPython.core.interactiveshell import InteractiveShell

#import import_ipynb # pip install import_ipynb


########################################################################################################################
#---------------------------------------- DQN_network_utilities -----------------------------------------------------#
########################################################################################################################

#########################################
#---- TO DO LIST & MODULE UPDATES-------#
#########################################

#- Verify all function gradient decent step and experience_replay, also check loss function and update equation
#- Add training function that accepts experience history data from Q_learning models to pretrain model
#- add a sorting function to experince replay that prioritizes high reward experiences
#- add functions for saving/loading models and weights 
#- add more input parameters to generate_networks function for diffrent initalizations, optimizers, 


###################################
#---- DQN_network_utilities-------#
###################################
 
#import DQN_network_utilities as NN  # import DQN_network_utilities module and rename it NN
#from NN import*                     # import all functions 

# Generate and initialize Target_network & Q_network
#Target_network, Q_network = NN.generate_networks(state, actions, learning_rate, model_architecture)

# Get selected action from NN and output index of Q(s,a) 
#action,Q_index = NN.choose_action(network, state, actions, epsilion = 1)

# Preform single gradient decent step and update network
#Q_network, loss = NN.gradient_step(Q_network, Target_network, experience, learning_rate, discount, Q_index)

# reshape states as vector for input capdability to model
#state,newstate = NN.reshape_states(state,newstate) 

# preform experience replay using minibatch of experiences, fitnetwork and return updated model and losses
#Q_network, batch_loss, batch_mean, batch_std = NN.experience_replay(Q_network, Target_network, D,  
                                                                 #learning_rate, discountfactor, 
                                                                 #batch_size, actions)

###################################
#----- import libraries ----------#
###################################

import traceback

#import matplotlib
#matplotlib.use('Agg')
import datetime
import random
import numpy as np
import csv, json
from copy import deepcopy
from pprint import pprint
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
import pandas as pd
import itertools
import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random
import statistics
from statistics import mean
import collections
from collections import deque
from keras import optimizers


#######################################################################################
#----------- Setting up input and output data structures -----------------------------#
#######################################################################################

def reshape_states(state,newstate):
    state = np.asarray(state).reshape(1, len(state))
    newstate = np.asarray(newstate).reshape(1, len(newstate))
    return state,newstate
    
def shape_experience(experience):
    state, action, newstate, reward = experience
    state = np.asarray(state).reshape(1, len(state))
    newstate = np.asarray(newstate).reshape(1, len(newstate))
    experience = state, action, newstate, reward 
    return experience



#######################################################################################
#---------------------------- Create Models ------------------------------------------#
#######################################################################################

def generate_networks(state, actions, learning_rate, model_architecture):
        
        layer1_neurons = model_architecture[0]
        layer2_neurons = model_architecture[1]
         
        model = Sequential()
        model.add(Dense(layer1_neurons, input_dim = len(state), activation='relu',kernel_initializer='zeros')) # try normal
        model.add(Dense(layer2_neurons, activation='relu',kernel_initializer='zeros'))
        #model.add(Dense(output_dim = len(actions), activation='linear',kernel_initializer='zeros'))
        model.add(Dense(output_dim = len(actions), activation='relu',kernel_initializer='zeros'))
        #sgd = optimizers.SGD(lr=learning_rate) #, decay=1e-6, momentum=0.5) #, nesterov=True)
        #model.compile(loss='mse', optimizer = sgd)
        model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate))
        model.summary()
        
        Target_network = model
        Q_network = Target_network
 
        return Target_network, Q_network
        
        
#######################################################################################
#---------------------------- Save and Load Models ------------------------ ----------#
#######################################################################################






######################################################################################
#---- Choose Action Based on Current Qnetwork Paramaters According to Epsilion-------#
######################################################################################

def choose_action(network, state, actions, epsilion = 1):
    
    state = np.asarray(state).reshape(1, len(state))
    
    Q = network.predict(state)
             
    if random.uniform(0, 1) > epsilion:
       # print('Argmax Action')
        Q_index = np.argmax(Q)
        Q_value = Q[0][Q_index]
        action = actions[Q_index]
             
    else:
       # print('Random Action')
        Q_index = random.randint(0,len(Q))
        Q_value = Q[0][Q_index]
        action = actions[Q_index]
             
    return(action,Q_index)


####################################################################################################
#------------- preform experience replay   --------------------------------#
####################################################################################################



def experience_replay(Q_network, Target_network, D, learning_rate, discountfactor, batch_size, actions):
    
    # experience relay consists of sampling a minibatch of experience from memory buffer D
    # after randomly selecting a set of experiences we train the current model using the
    # state from each experience as an input and the new estimate for Q(s,a) as the output 
    # the new estimate for Q(s,a,theta+) <--- Q(s,a,theta) + lr * [r(s,a,s') + gamma * max(Qhat(s',a',theta-) - Q(s,a,theta)]   
    
    # yi =  [r(s,a,s') + gamma * max(Qhat(s',a',theta-) <-- where Qhat is the target network
    # Q = 
    
    Loss = []
    Loss_total = []
    Loss_mean = []
    Loss_std = []
    
    minibatch = random.sample(D,batch_size)
    
    for i in range(len(minibatch)): # loop through all experiences currently in the memory buffer D
        
        state,action,newstate,reward = minibatch[i] # unpack s,a,s',r from expiernces 
        state,newstate = reshape_states(state,newstate) # pre-format s and s' to vector so they can be passed to network 
        
        current_Qs = Q_network.predict(state) # get current estimate for state s, "contains all Q(s,A) outputs"
        action_index = actions.index(action) # determine the index of Q(s,a) in current_Qs list 
        current_Q = current_Qs[0][action_index] # get current estimate of Q(s,a)
        Q = current_Q
        
        
        target_Qs = Target_network.predict(newstate) # get target estimates for newstate s',"contains all Qhat(s',A) outputs"
        target_index = np.argmax(Target_network.predict(newstate)) # determine the index of maxQhat(s',a') 
        target_Q = target_Qs[0][target_index] # set target_Q equal to maxQhat(s',a')  <-- target Q_function
        
        
        # compute value of new Q(s,a) using bellman equation, this is the Q estimate we will fit the network to
        new_Q = current_Q + learning_rate * (reward + discountfactor * target_Q - current_Q)
        yi = (reward + discountfactor * target_Q)
        #print('yi',yi)
        
        # set current networks estimate of Q(s,a) to the updated estimate new_Q calculated above 
        current_Qs[0][action_index] = new_Q 
        
        # create an copy of the current network output Q(s,A) which includes the new value estimate for Q(s,a) computed above
        updated_Qs = current_Qs # since only 1 value differs from the current network output for Q(s,A)
                                # when we preform gradient decent with respect to weights theta
                                # we only fit the weights to a single estimate Q(s,a) instead of Q(s,A), where A is all actions
        
        # preform graident decent "fit" the Q_network weight theta such that Q(s,a) = update_Q 
        Q_network.train_on_batch(state, updated_Qs)
        
        loss = (yi - Q)**2
        Loss.append(loss)
        
    Loss_total.append(sum(Loss))
    Loss_mean.append(sum(Loss)/len(Loss))
    Loss_std.append(np.std(Loss))
    #Loss_std.append( np.sqrt( (sum(Loss) - (sum(Loss)/len(Loss)))**2 / len(Loss)  ))
        
        

        
    return(Q_network, Loss_total, Loss_mean, Loss_std)



######################################################################################
#-------  gradient decent steps for single network updates --------------------------#
######################################################################################
    
    
def gradient_step(Q_network, Target_network, experience, learning_rate, discount, Q_index):
    state, action, newstate, reward = experience
    state, newstate = reshape_states(state,newstate)
    Q = Q_network.predict(state)
    #Q_update = Target_network.predict(newstate) # <-- use this <-- update according to partial bellman using target network
    Q_update = Q_network.predict(newstate)  # <-- update acorrding to partial bellman equation using Q network "seems to work better"
    #yi = reward + discount * max(Q_update[0])
    yi = reward + discount * max(Target_network.predict(newstate)[0])
    Q_current = Q[0][Q_index]
   # print('current Q', Q_current)
    loss = (yi - Q_current)**2
    #update_index = np.argmax(Target_network.predict(newstate)) #<--- use this <-- update according to partial bellman using target network 
    update_index = np.argmax(Q_network.predict(newstate)) # <-- update acorrding to partial bellman equation using Q network
    #Q[0][Q_index] = Q_update[0][update_index]
    
   # print('Q[0][Q_index]', Q[0][Q_index])
   # Q[0][Q_index] = 100
   # print('Q[0][Q_index]', Q[0][Q_index])
    
    Q[0][Q_index] =  Q_current + learning_rate * (reward  + discount * (yi - Q_current))
    #Q[0][Q_index] =  Q_current + learning_rate * (reward  + discount * Q_update[0][update_index] - Q_current)
    
   # print('new Q', Q)
    
    Q_network.train_on_batch(state, Q) # Q_network.fit(state, Q)
    return(Q_network, loss)

