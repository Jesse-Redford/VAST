import sys

sys.path.append('/home/pi/.local/lib/python3.6/site-packages/pandas')

import pandas as pd

import itertools
import random
import numpy as np
import math
#from math import cos sin
import time
import matplotlib.pyplot as plt
import pickle
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
#import seaborn as sns # for heatmap of Qtable



##########################################################
#----------   Physical Transition Enviroment ------------#
##########################################################

import pyrealsense2 as rs

def yaw(Q):
  if Q[3]**2+Q[1] ==0:
      return 0
  else:
    mag = (Q[3]**2+Q[1]**2)**0.5
    ang = 2*math.acos(Q[3]/mag)
    if Q[1] < 0:
        ang = -ang
    if ang > math.pi:
        ang = ang-(2*math.pi)
    if ang<(-math.pi):
        ang = ang+(2*math.pi)
  return ang # angle between -pi & pi in radiatns


 def GetLoc(pipe):
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    data = pose.get_pose_data()
    Q = [data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w]
    phi = yaw(Q)
    x = round(-data.translation.z * 39.3701,2) # x inches
    y= round(-data.translation.x * 39.3701,2) # y inches
  return[x,y,phi]
    
    
def reset_state(phi_states):
    print("Reseting State")
    rs.device.hardware_reset
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)
    x,y,phi = GetLoc(pipe)
    kit.servo[1].angle = 90
    kit.servo[2].angle = 90
    disc_phi = find_nearest(phi_states, phi)
    state = (90,90,disc_phi)
    print("Finished Resetting")
    return()


def find_nearest(array, value):
    #print(array,value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
    

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
    
    xf,yf,phi_f = GetLoc(pipe)
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

def random_state(states):
    return random.choice(states[:]) 

def random_reward():
    return random.uniform(0, 100)


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
    

def Train(simulate,Qtable,states,phi_states,actions,initial_state,eps_initial,eps_final,alpha_initial,alpha_final,gamma_initial,gamma_final,training_steps):
   
    X = []
    Y = []
    PHI = []
    STEP = []
    
   # if simulate == False:
    #  state,pipe = reset_state(phi_states)
     
    epsilion = eps_initial
    alpha = alpha_initial
    gamma = gamma_initial

    step = 0
    
    for e in range(episodes):
    
      state, pipe = reset_state(phi_states) 
      
      for i in range(iterations):
        
        alpha = alpha * exponential_decay(alpha_initial,alpha_final,episodes*iterations)
        gamma = gamma * exponential_decay(gamma_initial,gamma_final,episodes*iterations)
        epsilion = epsilion * exponential_decay(eps_initial,eps_final,episodes*iterations)
        
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
        
        
        x,y,phi = GetLoc(pipe)
        X.append(x)
        Y.append(Y)
        PHI.append(phi)
        STEP.append(step)
        
          
    pipe.stop()
    
    return(Qtable)



def get_Qtable_policy(Qtable):
   Qtable['policy'] = Qtable.idxmax(axis=1)
   policy = Qtable['policy']
   return(policy)

def get_policy_action(what_policy,current_state):
    policy_action = what_policy[current_state]
    return(policy_action)

def PolicyRollout(Qtable,simulate,initial_state,policy_steps,states,phi_states):
    
    #if simulate == False:
    state,pipe = reset_state(phi_states)
    
    
    policy =  get_Qtable_policy(Qtable)
    steps = 0
    state = initial_state
    iteration = []
    reward_step = []
    total_reward = []
    average_reward = []
    X = []
    Y = []
    PHI = []
    
    for i in range(policy_steps)
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
        
        x,y,phi = GetLoc(pipe)
        X.append(x)
        Y.append(Y)
        PHI.append(phi)
   
    
    reward_plotter(iteration,reward_step,total_reward,average_reward)  
    coordinates_plotter(iteration,X,Y,PHI):
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
    
def coordinates_plotter(steps,xs,ys,phis):

    plt.figure(3)
    plt.scatter(xs,ys)
    plt.show()
    
    return()


def RunTest(simulate, load_path, load_filename, save_path, save_filename, state1, state2, state3, action1, action2, action3, initial_state, eps_initial, eps_final, alpha_initial, alpha_final, gamma_initial, gamma_final, iterations, episodes, policy_steps):
    
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
     
     

##########################################################
#--------------   Set up and Run Test -------------------#
##########################################################

SIMULATED = False

LOAD_PATH = r'/home/pi/Desktop/' , #'none'
LOAD_FILENAME = 'Qtables\Qtable_trained.pkl', #'none'
SAVE_PATH = r'/home/pi/Desktop/'
SAVE_FILENAME = 'Qtables\Qtable_trained.pkl'

THETA_LOWER = 90
THETA_UPPER = 90
THETA_STEPSIZE = 90
PHI_LOWER = -np.pi
PHI_UPPER = np.pi
PHI_INTERVALS = 10
T_DELAY_LOWER = 0.015
T_DELAY_UPPER = 0.025
T_DELAY_INTERVALS = 0.005 

EPSILION_INITAL = 1
EPSILION_FINAL = .1
ALPHA_INITAL = .9
ALPHA_FINAL = .8
GAMMA_INITAL = .8
GAMMA_FINAL = .9

TRAINING_STEPS = 25
POLICY_STEPS = 25



RunTest(simulate = SIMULATED, # False = run physical robot enviroment, True = run simulated  enviroment
        load_path = LOAD_PATH , #'none', # set load_path & load_filename to 'none' to begin training with empty Qtable
        load_filename = LOAD_FILENAME, #'none', # specify a previously trained Qtable.pkl file to run session with pre trained Q function 
        save_path = SAVE_PATH, # specifify folder path to save trained Qtable 
        save_filename = SAVE_FILENAME, # specify name of trained Qtable that will be saved at the end of the training session
        state1= range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), # pos servo 1
        state2= range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), #pos servo 2
        state3 = np.linspace(PHI_LOWER, PHI_UPPER, PHI_INTERVALS), # disccirtize phi state --> np.linspace(lower = -np.pi, upper = np.pi, phi_interval = 10)
        action1 = range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), #pos servo 2
        action2 = range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), # position servo 2
        action3 = np.arange(T_DELAY_LOWER, T_DELAY_UPPER, T_DELAY_INTERVALS), # t inteval between 1deg steps, in milisecons 
        initial_state = (90,90,0), # start with inital state 
        eps_initial = EPSILION_INITAL, 
        eps_final = EPSILION_FINAL, 
        alpha_initial = ALPHA_INITAL, 
        alpha_final = ALPHA_FINAL, 
        gamma_initial = GAMMA_INITAL,
        gamma_final = GAMMA_FINAL, 
        iterations = ITERATIONS,
        episodes = EPISODES,
        training_steps = TRAINING_STEPS, 
        policy_steps = POLICY_STEPS)
