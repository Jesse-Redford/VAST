

###################################
#----- import libraries ----------#
###################################

import traceback
import sys
import matplotlib
matplotlib.use('Agg')
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
    

###################################
#-------- MISC functions ----------#
###################################

def reset():
    # reset camera and phi
    # reset joint angles, theta1, theta2 to 90
    state = (90,90,0)
    return state

def Beta(start = 1,end = .1,maxsteps = 100):
    return math.exp(math.log(end/start)/maxsteps)

def list_actions(action1,action2,action3):
    actions = list(itertools.product(action1,action2,action3)) # Create a list of every action combination
    return actions

def shape_experience(experience):
    state, action, newstate, reward = experience
    state = np.asarray(state).reshape(1, len(state))
    newstate = np.asarray(newstate).reshape(1, len(newstate))
    experience = state, action, newstate, reward 
    return experience

###################################
#-------- Create Models ----------#
###################################


def generate_networks(state, actions, learning_rate, model_architecture):
        
        layer1_neurons = model_architecture[0]
        layer2_neurons = model_architecture[1]
        
        model = Sequential()
        model.add(Dense(layer1_neurons, input_dim = len(state), activation='relu'))
        model.add(Dense(layer2_neurons, activation='relu'))
        model.add(Dense(output_dim = len(actions), activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate))
        
        Target_network = model
        Q_network = Target_network
        
        print('Model Architecture')
        print('-------------------')
        print('Input Layer', len(state))
        print('Layer1_neurons', layer1_neurons)
        print('Layer2_neurons', layer2_neurons)
        print('Output Layer', len(actions))
        print('-------------------')
        
        return Target_network, Q_network

    
######################################################################################
#---- Choose Action Based on Current Qnetwork Paramaters According to Epsilion-------#
######################################################################################

def choose_action(network, state, actions, epsilion = 1):
    
    state = np.asarray(state).reshape(1, len(state))
    
    Q = network.predict(state)
             
    if random.uniform(0, 1) > epsilion:
        #print('Argmax Action')
        Q_index = np.argmax(Q)
        Q_value = Q[0][Q_index]
        action = actions[Q_index]
             
    else:
        #print('Random Action')
        Q_index = random.randint(0,len(Q))
        Q_value = Q[0][Q_index]
        action = actions[Q_index]
             
    return(action,Q_index)

#######################################################################################
#------------------------------Execute Selected Action--------------------------------#
#######################################################################################
     
def execute_action(state, action): # simualate or act real enviroment here 

    start = 0
    end = 2
    
    if state == (0,0,0) and action ==(0,0,0):
        reward = 10
        newstate = (0,0,1)
    
    elif state == (0,0,1) and action == (0,0,1):
        reward = 10
        newstate = (0,0,2)
    
    elif state == (0,0,2) and action == (0,0,2):
        reward = 10
        newstate = (0,0,3)
        
    elif state == (0,0,3) and action == (1,0,0):
        reward = 20
        newstate = (0,0,0)
        
    elif action == (1,1,1):
        reward = 1
        newstate = (0,0,0)
    
    else:
        reward = -1
        newstate = random.randint(start, end) , random.randint(start, end), random.randint(start, end)

    return newstate, reward
    
######################################################################################
#---- Fit Current Qnetwork Paramaters Using Partial of Bellman Eqution --------------#
######################################################################################
    
def reshape_states(state,newstate):
    state = np.asarray(state).reshape(1, len(state))
    newstate = np.asarray(newstate).reshape(1, len(newstate))
    return state,newstate
    
def gradient_step(Q_network, Target_network, experience, discount, Q_index):
    state, action, newstate, reward = experience
    state, newstate = reshape_states(state,newstate)
    Q = Q_network.predict(state)
    Q_update = Target_network.predict(newstate)
    #print('Qs',Q)
    #print('Qs Hat', Q_update)
    yi = reward + discount * max(Target_network.predict(newstate)[0])
    Q_current = Q[0][Q_index]
    #print('current Qsa', Q_current)
    #print('target Q', yi)
    loss = (yi - Q_current)**2
    #print('loss',loss)
    update_index = np.argmax(Target_network.predict(newstate))
    Q[0][Q_index] = Q_update[0][update_index]
    #print('updated Q', Q)
    Q_network.train_on_batch(state, Q) # Q_network.fit(state, Q)
    #print('verify fit', Q_network.predict(state))
    return(Q_network, loss)
    
####################################################################################################
#---- Plot Loss Function Results  --------------#
####################################################################################################

def plot_results(Losses, Avg_Loss):
    iterations = [i for i in range(len(Losses))]
    matplotlib.use('TkAgg')
    # Visualize loss history
    plt.plot(iterations, Avg_Loss, 'r--')
    plt.plot(iterations, Losses, 'b--')
    #plt.plot(epoch_count, test_loss, 'b-')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(['Losses', 'Avg_Loss']) # compare Losses during experience replay to step losses 
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.show()
    return()

def plot_rewards(Reward,Reward_Avg):
    iterations = [i for i in range(len(Reward))]
    matplotlib.use('TkAgg')
    # Visualize loss history
    plt.plot(iterations, Reward, 'r--')
    plt.plot(iterations, Reward_Avg, 'b--')
    #plt.plot(epoch_count, test_loss, 'b-')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(['reward', 'avg_reward']) # compare Losses during experience replay to step losses 
    plt.xlabel('iterations')
    plt.ylabel('x-displacment')
    plt.show()
    return()


####################################################################################################
#------------------------------------ Experience Replay ------------------------------------------#
####################################################################################################

def experience_replay(Q_network, Target_network, D, batch_size, learning_rate, actions):
    
    Batch_Losses = []
    Avg_Batch_Loss = []
    
    minibatch = random.sample(D,batch_size)
    for i in range(len(minibatch)):
        state,action,newstate,reward = minibatch[i]
        state,newstate = reshape_states(state,newstate)
        #print('state', state, 'action',action,'newstate',newstate,'reward')
        Q = Q_network.predict(state)
        index = actions.index(action)
        q_current = Q[0][index]
        #print('Q(s)' , Q)
        #print('actions',actions)
        #print('action', action)
        #print('action index', index)
        #print('q at action index', q_current)
        max_q_future = reward + learning_rate * max(Target_network.predict(newstate)[0])
        #print('max future Q', max_q_future)
        update_index = np.argmax(Target_network.predict(newstate))
        Q_update = Target_network.predict(newstate)
        Q[0][index] = Q_update[0][update_index] * learning_rate + reward
        #print('updated Q', Q)
        loss = Q_network.train_on_batch(state, Q) # Q not Q_update
        Batch_Losses.append(loss)
        Avg_Batch_Loss.append(sum(Batch_Losses)/len(Batch_Losses))
    #plot_results(Batch_Losses, Avg_Batch_Loss)
    return(Q_network, Target_network)

####################################################################################################
#------------------------------------ DQN Algorithm  ----------------------------------------------#
####################################################################################################

def DQN_RUNNER(example_state, actions, learning_rate, model_architecture, discountfactor, 
                iterations, episodes, buffer_size, batch_size, update_frequency, epsilion, epsilion_inital, epsilion_final):
    
    Losses = []
    Avg_Loss = []
    Batch_Loss = []
    Avg_Batch_Loss = []
    Reward = []
    Total_Reward = []
    Avg_Reward = []
    
    # Initialize replay memory D to capacity N
    D = collections.deque(maxlen=buffer_size)
    
    # Initialize action-value function Q with weights theta ---> Q_network
    Target_network, Q_network = generate_networks(example_state, actions, learning_rate, model_architecture)
    
    # Initialize target action-value function Qhat with weights theta - = theta --> Target_network
    Target_network.set_weights(Q_network.get_weights())
    
    step = 0
    
    for e in range(episodes):
        # initialize sequence and preprocess state
        state = reset()
        
        for i in range(iterations):
            
            epsilion = epsilion * Beta(epsilion_inital,epsilion_final,iterations*episodes) # decay epsilion 
            
            # with probability e select action a
            action, Q_index = choose_action(Q_network, state, actions, epsilion)
            
            # execute action and observe reward, and newstate
            newstate,reward = execute_action(state,action)
            
            # Store transition (s,a,s',r) in D
            experience = state,action,newstate,reward
            D.append(experience)
        
            if i > buffer_size or e > 1:
                # sample random minibatch of transistions (s,a,s',r) from D
                Q_network, Target_network = experience_replay(Q_network, Target_network, D, batch_size, learning_rate,actions)
                
            else:  
                # perform a gradient descent step on ( yi - lr*max_Qhat(s',a',theta -))^2 wrt to current Q_network weights 
                Q_network, loss = gradient_step(Q_network, Target_network, experience, discountfactor,Q_index)
                
                # ?? gradient decent on just Q(s,a) or Q(s,A), where A is all the networkout puts
                
            #print(loss)
            
            # Every C steps reset Qhat = Q
            if step == update_frequency:
                Target_network = Q_network
                step = 0                 
            else:
                step += 1
                
            state = newstate
            
            Reward.append(reward)
            Total_Reward.append(sum(Reward))
            Avg_Reward.append(sum(Total_Reward)/len(Total_Reward))
            Losses.append(loss)
            Avg_Loss.append(sum(Losses)/len(Losses))
    
    plot_rewards(Reward,Avg_Reward)
    plot_results(Losses, Avg_Loss)    
    print('done')
    return()
             
             
########################################################################################################################
#----------------------------------------  Setup and Run Test ---------------------------------------------------------#
########################################################################################################################

STATE = (0,0,0)
ACTIONS = list_actions(action1 = range(0,10,1), action2 = range(0,10,1), action3 = range(0,10,1))
MODEL_ARCHITECTURE = (int(len(ACTIONS)/len(STATE)**3), int((len(ACTIONS)/len(STATE)**3)/2))
LEARNING_RATE = .2 #2 # 0.02
DISCOUNT_FACTOR = .4
ITERATIONS = 100
EPISODES = 10
BUFFER_SIZE = 32 # 
BATCH_SIZE = 16
UPDATE_FREQUENCY = 10
EPSILION_INITAL = .1
EPSILION_FINAL = .1
   
DQN_RUNNER(example_state = STATE,
           actions = ACTIONS,
           learning_rate = LEARNING_RATE,
           model_architecture = MODEL_ARCHITECTURE,
           discountfactor = DISCOUNT_FACTOR, 
           iterations = ITERATIONS, 
           episodes = EPISODES,
           buffer_size = BUFFER_SIZE, 
           batch_size = BATCH_SIZE, 
           update_frequency = UPDATE_FREQUENCY,
           epsilion = EPSILION_INITAL,
           epsilion_inital = EPSILION_INITAL,
           epsilion_final = EPSILION_FINAL,
          )


    
    
    

    
    
