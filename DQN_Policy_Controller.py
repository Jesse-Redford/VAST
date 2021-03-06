
import DQN
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
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
import pyrealsense2 as rs


##########################################################
#--- Helper Functions -------#
##########################################################
import Q_pandas_utilities as QPU
from QPU import*
import numpy as np
#import import_ipynb # pip install import_ipynb

import DQN_plotting_utilities
from DQN_plotting_utilities import *
#from DQN_plotting_utilities import plot_results

import Snakebot_enviroments as SBE
from Snakebot_enviroments import *

import DQN_network_utilities as NN
from DQN_network_utilities import *


def GetLoc(pipe):
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    data = pose.get_pose_data()
    Q = [data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w]
    rotVecs = getRotVecs(Q)
    phi = rotVecs[2]
    x = round(-data.translation.z * 39.3701,2) # x inches
    y= round(-data.translation.x * 39.3701,2) # y inches
    #print('x', x, 'y', y, 'phi',phi)
    return[x,y,phi]
    
    
def system_reset(xtarget = 50, ytarget = 50):
    device.reset()
    pipe =     
    x,y,phi = GetLoc(pipe)
    kit.servo[1].angle = 90
    kit.servo[2].angle = 90
    system_state = x,y,phi,90,90,xtarget,ytarget
    return(system_state, pipe)


def PolicyTransition(current_state, selected_action, phi_states,pipe):
    
    a1,a2, phi = current_state
    a1_target, a2_target, t_delay = selected_action
    
    kit.servo[1].angle = a1
    kit.servo[2].angle = a2
    
    while a1 != a1_target or a2 != a2_target: 
        if a1 < a1_target:
            a1 += 1
        elif a1 > a1_target:
            a1 -= 1
                
        if a2 < a2_target:
                a2 += 1
        elif a2 > a2_target:
                a2 -= 1
                
        kit.servo[1].angle = a1
        kit.servo[2].angle = a2
            
        time.sleep(t_delay)
    
    assert a1 == a1_target and a2 == a2_target, "Problem with moving joint angles to target positions"
    
    xf,yf,phi_f = GetLoc(pipe)    
    disc_phi = find_nearest(phi_states, phi_f)
    newstate = a1, a2, disc_phi

    return(newstate)

def PolicyRollout(current_system_state, selected_policy_action, policy_steps, pipe):
    
    policy =  get_Qtable_policy(selected_policy_action[0])
    
     
    # Get low level representation
    phi_disc = find_nearest(selected_policy_action[2],current_system_state[2])
    low_level_state = (current_system_state[3],current_system_state[4], phi_disc) # (theta1,theta2,phi_disc)     
    state = low_level_state

    for p in range(policy_steps):  
        action = get_policy_action(policy,state)     
        check, new_state, reward = PolicyTransition(state,action,selected_policy_action[2],pipe)   
        state = new_state
    
    xf,yf,phi_f = GetLoc(pipe)
    
    theta1,theta2,phi = state
    
    new_system_state = xf,yf,phi_f,theta1,theta2,current_system_state[5],current_system_state[6]
    
    return new_system_state

def execute_policy(current_system_state, selected_policy_action, policy_steps = 10, pipe):

     # Executre policy for a fixed amout of steps
     new_system_state = PolicyRollout(current_system_state,  selected_policy_action, policy_steps, pipe)
     
     xtarget = current_system_state[5]
     ytarget = current_system_state[6]
     
     x = current_system_state[0]
     y = current_system_state[1]
     
     xf = new_system_state[0]
     yf = new_system_state[1]
     
     reward = np.sqrt( (xtarget-x) + (ytarget -y) ) - np.sqrt( (xtarget-xf) + (ytarget -yf) )
     
     return new_system_state,reward
   

def RunSession(example_state,actions, learning_rate = 0.0001, model_architecture = (50,10) , discountfactor = 0.9, 
                iterations = 10, episodes = 2, buffer_size = 100, batch_size = 20, update_frequency =20,
                  epsilion, epsilion_inital = 1, epsilion_final = 0.1):
    Step_loss =[]
    Step_total = []
    Step_mean = []
    Step_std = []
    
    Batch_loss = []
    Batch_mean = []
    Batch_std = []
    
    Reward = []
    Reward_total = []
    Reward_avg = []
    Reward_std = []
    Reward_samp = []
    
    X = []
    Y = []
    
    Step = []       
    step = 0
    
    #Initalize QNetwork
    Q_network, Target_network = NN.generate_networks(example_state, actions, learning_rate, model_architecture)
    
    # Initialize target action-value function Qhat with weights theta - = theta --> Target_network
    Target_network.set_weights(Q_network.get_weights())

    # Initalize Replay buffer
    D = collections.deque(maxlen=buffer_size)
    
     # initialize training             
    for e in range(episodes):
        
        # reset system and define a set of target coordinates for episode - system_state = x,y,phi,theta1,theta2,xtarget,ytarget 
        system_state,pipe = system_reset(xtarget = 50, ytarget = 50) 
        
        # reduce system state for new DQN observation - dqn_state = (x,y,phi,xtarget,ytarget)
        dqn_state = (system_state[0],system_state[1],system_state[2],system_state[5],system_state[6]) 
        
        for i in range(iterations):
            
            X.append(system_state[0]) # record current location
            Y.append(system_state[1]) # record current location
            
            # choose action(policy) according to epsilion
            action, Q_index = NN.choose_action(Q_network, dqn_state, actions, epsilion)
            
            # excute action(policy) for some amount of fixed policy steps
            new_system_state, reward = execute_policy(system_state, selected_policy_action, policy_steps = 10, pipe)
            
            # update new DQN state 
            dqn_newstate = (new_system_state[0],new_system_state[1],new_system_state[2],new_system_state[5],new_system_state[6])
            
            # record experience and add to replay buffer
            experience = (dqn_state, action, dqn_newstate, reward)
            D.append(experience)
            
            if i > buffer_size or e > 1:
                # sample random minibatch of transistions (s,a,s',r) from D
                Q_network, batch_loss, batch_mean, batch_std = NN.experience_replay(Q_network, Target_network, D, learning_rate, discountfactor, batch_size, actions)
                
                # store loss history 
                Batch_loss += batch_loss #.append(batch_loss)
                Batch_mean += batch_mean #.append(batch_mean)
                Batch_std += batch_std #.append(batch_std)
                
            
            # perform a gradient descent step on ( yi - lr*max_Qhat(s',a',theta -))^2 wrt to current Q_network weights 
            Q_network, loss = NN.gradient_step(Q_network, Target_network, experience, learning_rate, discountfactor,Q_index)
                
            # Every C steps reset Qhat = Q
            if step == update_frequency:
                Target_network = Q_network
                step = 0                 
            else:
                step += 1
     
            # update current state
            dqn_state = dqn_newstate
            
            Reward.append(reward)
            Reward_total.append(sum(Reward))
            Reward_avg.append(sum(Reward_total)/len(Reward))
            Reward_samp.append(sum(Reward)/len(Reward))
            Reward_std.append(np.std(Reward))
          
            Step_loss.append(loss)
            Step_total.append(sum(Step_loss))
            Step_mean.append(sum(Step_loss)/len(Step_loss))
            Step_std.append(np.std(Step_loss))
    
            
        pipe.stop() # stop data feed before reseting camera
    
    
    DQN_plotting_utilities.plot_results(Reward, Reward_total, Reward_avg, Reward_std,  Reward_samp,
                                        Step_loss, Step_total, Step_mean,Step_std,
                                       Batch_loss, Batch_mean, Batch_std)
        
        
    return()


######################################################
############## Setup for Intellegent Controller ######
######################################################

# import load parameters function for Qtables

def load_Qtable(path,filename):
    print('Qfunction is loaded')
    return pd.read_pickle(path+filename)
    
def GetFileParameters(path,filename):
    return states,actions
    
# Load Qpolicies and parameters to create Action Space
QX = (Qfx, states, phi_states) # or just load policy and phi_states
QY = (Qfy, states, phi_states)
action1 = QX
actions2 = QY
actions =  list_actions(QX,QY)

#Define Target Coordinates for Training session
Target_Coordinates = (xtarget = 20, ytarget = 20)

# Define a State
example_state = (1,2,3,4,5,6,7)

RunSession(example_state,actions, learning_rate, model_architecture, discountfactor, 
                iterations, episodes, buffer_size = 100, batch_size = 20, update_frequency =20,
                  epsilion, epsilion_inital = 1, epsilion_final = 0.1):



