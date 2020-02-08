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
#import seaborn as sns # for heatmap of Qtable



##########################################################
#--- Helper Functions From  Q_pandas_utilities.py -------#
##########################################################

# Add QPU infrom on functions to use in main script

import Q_pandas_utilities as QPU
from QPU import*

array[idx] = find_nearest(array, value)
states,actions = three_states_three_actions(state1,state2,state3,action1,action2,action3)
Zeros_Qtable = create_zeros_Qtable(states,actions)
Random_Qtable = create_random_Qtable(states,actions,lowerbound,upperbound,round_to)
save_Qtable(Qtable,path,filename)
load_Qtable(path,filename)
Qtable = Update_Qtable(Qtable,actions,state,action,new_state,reward,alpha,gamma)
action = choose_action(Qtable,state,epsilion,actions)
policy = get_Qtable_policy(Qtable)
policy_action = get_policy_action(what_policy,current_state)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
#----------   Physical Transition Enviroment ------------#
##########################################################

import pyrealsense2 as rs

def getRotVecs(quat):
    qHat = [quat[0],quat[1],quat[2]]
    qHatMag = (quat[0]**2+quat[1]**2+quat[2]**2)**1/2
    eHat = [qHat[0]/qHatMag,qHat[1]/qHatMag,qHat[2]/qHatMag]
    theta = 2*math.acos(quat[3])
    return [theta*eHat[0],theta*eHat[1],theta*eHat[2]]

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
    
    
def reset_state():
    print("Reseting State")
    kit.servo[1].angle = 90
    kit.servo[2].angle = 90
    print("Finished Resetting")
    return()


def non_simulated_transition(current_state, selected_action, phi_states,pipe):
    
    print("Current State = ", current_state)
    print("Selected Action = ", selected_action)
    
    xi,yi,phi_i = GetLoc(pipe)
    
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
    
    xf,yf,phi_f = GetLoc()
    disc_phi = find_nearest(phi_states, phi_f)

    newstate = a1, a2, disc_phi
    reward = xf - xi 

    print('newstate', newstate)
    print('reward', reward)
    print('x coordnate',xf)
    
    return('done',newstate,reward)


##########################################################
#----------   Simulated Transition Enviroment -----------#
##########################################################


def simulated_transition(current_state, selected_action, all_possible_states):
    
    state = current_state
    action = selected_action
    
    #print('state',state,'action',action)
    
    if state == (0,0,0) and action ==(0,0,1):
        reward = 1
        newstate = (0,0,1)
    
    elif state == (0,0,1) and action == (0,0,2):
        reward = 1
        newstate = (0,0,2)
    
    elif state == (0,0,2) and action == (0,0,3):
        reward = 1
        newstate = (0,0,3)
        
    elif state == (0,0,3) and action == (0,0,0):
        reward = 100
        newstate = (0,0,0)
        
    elif action == (1,1,1):
        reward = .5
        newstate = (0,0,0)
    
    else:
        reward = -1
        newstate = random_state(all_possible_states)

    return(newstate,reward)
    
##########################################################
#----------  Training Loop ----------------------------#
##########################################################


def Train(simulate,Qtable,states,phi_states,actions,initial_state,eps_initial,eps_final,alpha_initial,alpha_final,gamma_initial,gamma_final,training_steps):
    
    if simulate == False:
        reset_state()
        rs.device.hardware_reset
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        pipe.start(cfg)
      
    state = initial_state

    epsilion = eps_initial
    alpha = alpha_initial
    gamma = gamma_initial

    step = 0

    while step < training_steps:
    
        alpha = alpha * exponential_decay(alpha_initial,alpha_final,training_steps)
        gamma = gamma * exponential_decay(gamma_initial,gamma_final,training_steps)
        epsilion = epsilion * exponential_decay(eps_initial,eps_final,training_steps)
        
        action = choose_action(Qtable,state,epsilion,actions) # choose action with probability epsilion
    
        if simulate == True:
            new_state, reward = simulated_transition(state,action,states)
        else:
            check = 'not done'
            while True:
                check,new_state, reward = non_simulated_transition(state,action,phi_states,pipe)
                if check == 'done':
                    break
                else: sleep(0.01)
    
        Qtable = Update_Qtable(Qtable,actions,state,action,new_state,reward,alpha,gamma)
        state = new_state
        step = step + 1 
          
    pipe.stop()
    
    return(Qtable)

##########################################################
#----------  Policy Rollout Loop ------------------------#
##########################################################


def PolicyRollout(Qtable,simulate,initial_state,policy_steps,states,phi_states):
    
    if simulate == False:
        reset_state()
        rs.device.hardware_reset
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        pipe.start(cfg)
    
    
    policy =  get_Qtable_policy(Qtable)
    steps = 0
    state = initial_state
    iteration = []
    reward_step = []
    total_reward = []
    average_reward = []

    while steps < policy_steps:
        action = get_policy_action(policy,state)
        
        if simulate == True:
            new_state, reward = simulated_transition(state,action,states)
        elif simulate == False:
            check = 'not done'
            while True:
                check,new_state, reward = non_simulated_transition(state,action,phi_states,pipe)
                if check == 'done':
                    break
                else: sleep(0.01)
                
        state = new_state
        iteration.append(steps)
        reward_step.append(reward)
        total_reward.append(sum(reward_step))
        average_reward.append(sum(total_reward)/len(iteration))
        steps = steps + 1
    
    reward_plotter(iteration,reward_step,total_reward,average_reward)  
    pipe.stop()
    
    return()


def reward_plotter(iteration,reward_step,total_reward,average_reward):

    plt.figure(2)
    df_Preformance_Plots = pd.DataFrame({ 
            'Iteration': iteration, 
            'Reward':reward_step,
            'Total Reward':total_reward,
            'Average Reward':average_reward
            })
    ax_preformance_parameters = plt.gca()
    df_Preformance_Plots.plot(kind='line',x='Iteration',y='Reward',color='red',ax=ax_preformance_parameters)
    df_Preformance_Plots.plot(kind='line',x='Iteration',y='Total Reward',color='blue',ax=ax_preformance_parameters)
    df_Preformance_Plots.plot(kind='line',x='Iteration',y='Average Reward',color='green',ax=ax_preformance_parameters)
    plt.show()
    
    return()

  
##########################################################
#----------  Run Full Test Function ------------------------#
##########################################################



def RunTest(simulate, load_path, load_filename, save_path, save_filename, state1, state2, state3, action1, action2, action3, initial_state, eps_initial, eps_final, alpha_initial, alpha_final, gamma_initial, gamma_final, training_steps, policy_steps):
    
    if str(load_path) == str(load_filename):
    
        print("Defining State Action Space")
        states,actions = three_states_three_actions(state1,state2,state3,action1,action2,action3)
        Qtable_untrained = create_zeros_Qtable(states,actions)
        
        print("Saving Untrained Qtable")
        save_Qtable(Qtable_untrained,save_path,save_filename)
        
        print("Loading Qtable") 
        #Qtable_untrained.to_pickle(save_path+save_filename)  
        Qtable =load_Qtable(save_path,save_filename)
        
    else:
        print("Loading Qtable") 
        Qtable =load_Qtable(load_path,load_filename)
        states, actions = three_states_three_actions(state1,state2,state3,action1,action2,action3)
    
    print("Starting Training Session")
    trainedQtable = Train(simulate,Qtable,states,state3,actions,initial_state,eps_initial,eps_final,alpha_initial,alpha_final,gamma_initial,gamma_final,training_steps)
    print("Training Session is Complete")
    
    
    print("Saving Updates")
    trainedQtable.to_pickle(save_path+save_filename) # can also do JSON or other file types
    print('Finished Saving Qtable')
    
    #plt.figure()
    #sns.heatmap(trainedQtable, cmap='RdYlGn_r', annot=True)
    #plt.title("Trained Qtable")
    
    if policy_steps > 0:
        print("Begining PolicyRollout") 
        PolicyRollout(trainedQtable,simulate,initial_state, policy_steps,states,state3)
        print("PolicyRollout is Complete")
    else:
        print("Session is Complete")
            
    return()
     
