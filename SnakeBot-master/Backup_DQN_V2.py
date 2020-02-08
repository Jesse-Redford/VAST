

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
from keras import optimizers
    
###################################
#-------- MISC functions ----------#
###################################

def reset():
    # reset camera and phi
    # reset joint angles, theta1, theta2 to 90
    state = (0,0,0)
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

#######################################################################################
#---------------------------- Create Models ------------------------------------------#
#######################################################################################

def generate_networks(state, actions, learning_rate, model_architecture):
        
        layer1_neurons = model_architecture[0]
        layer2_neurons = model_architecture[1]
         
        model = Sequential()
        model.add(Dense(layer1_neurons, input_dim = len(state), activation='relu',kernel_initializer='zeros')) # try normal
        model.add(Dense(layer2_neurons, activation='relu',kernel_initializer='zeros'))
        model.add(Dense(output_dim = len(actions), activation='linear',kernel_initializer='zeros'))
        #sgd = optimizers.SGD(lr=learning_rate) #, decay=1e-6, momentum=0.5) #, nesterov=True)
        #model.compile(loss='mse', optimizer = sgd)
        model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate))
        model.summary()
        
        Target_network = model
        Q_network = Target_network
 
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
    #end = 32 #6
    
    phi_disc = 2
    angle_disc = 5
    
    if state == (1,1,1) and action ==(1,1,1):
        reward = 5
        #reward = random.randint(1, 2)
        newstate = (2,2,2)
    
    elif state == (2,2,2) and action == (3,3,3):
        reward = 10
        #reward = random.randint(2, 3)
        newstate = (3,3,3)
    
    elif state == (3,3,3) and action == (4,4,4):
        reward = 15
        #reward = random.randint(3,5)
        newstate = (5,5,5)
        
    elif state == (5,5,5) and action == (6,6,6):
        reward = 50
        #reward = 50 #random.randint(5, 7)
        newstate = (1,1,1)
        
    elif action == (1,1,1):
        reward = -1
       # reward = random.randint(0, 1)
        newstate = (1,1,1)
    
    else:
        #reward = -1
        reward = -5 #random.randint(-5, 1)
        newstate = (1,1,1) #random.randint(start, angle_disc) , random.randint(start,angle_disc), random.randint(start, phi_disc)

    return newstate, reward
    
######################################################################################
#---- Fit Current Qnetwork Paramaters Using Partial of Bellman Eqution --------------#
######################################################################################
    
def reshape_states(state,newstate):
    state = np.asarray(state).reshape(1, len(state))
    newstate = np.asarray(newstate).reshape(1, len(newstate))
    return state,newstate
    
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

####################################################################################################
#-------------  compile training data from experience buffer --------------------------------------#
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



#####################################################################
#------------------------ Plot Loss Function Results  --------------#
#####################################################################


def plot_results(reward, reward_total, reward_avg, reward_std,  reward_samp,
                      step_loss, step_total, step_mean, step_std, 
                      batch_loss, batch_mean, batch_std):
    
    matplotlib.use('TkAgg')
    
    # Batch losses
    x1 = [i for i in range(len(batch_loss))]
    batch_loss = np.asarray(batch_loss)
    batch_mean = np.asarray(batch_mean)
    batch_std = np.asarray(batch_std)
    
    # Step losses
    x2 = [i for i in range(len(step_loss))]
    step_loss = np.asarray(step_loss)
    step_total = np.asarray(step_total)
    step_mean = np.asarray(step_mean)
    step_std = np.asarray(step_std)
    
    # rewards
    x3 = [i for i in range(len(reward))]
    reward = np.asarray(reward)
    reward_total = np.asarray(reward_total)
    reward_avg = np.asarray(reward_avg)
    reward_std = np.asarray(reward_std)

    fig = plt.figure()
    
    # Batch losses
    plt.subplot(3, 2, 3)
    plt.plot(x1,batch_loss, 'green') 
    plt.xlabel('Experience Replay')
    plt.ylabel('Loss')
    plt.legend(['batch loss'])
    
    plt.subplot(3, 2, 4)
    plt.plot(x1,batch_mean, 'green')
    plt.fill_between(x1, batch_mean+batch_std, batch_mean-batch_std, facecolor='green', alpha=.3, edgecolor='none')
    plt.xlabel('Experience Replay')
    plt.legend(['mean', 'std'])
    
    # Step Losses
    plt.subplot(3, 2, 5)
    plt.plot(x2,step_loss, 'b')    
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.legend(['step loss'])
    
    
    plt.subplot(3, 2, 6)
    plt.plot(x2,step_mean, 'b')
    plt.fill_between(x2, step_mean+step_std, step_mean-step_std, facecolor='b', alpha=.3, edgecolor='none')
    plt.xlabel('step')
    plt.legend(['mean', 'std'])
    
    
    # rewards 
    plt.subplot(3, 2, 1)
   # plt.plot(x2,reward_avg, 'r--')
    #plt.plot(x2,reward_samp, 'b--')
    plt.plot(x2,reward, 'brown') 
   # plt.fill_between(x2, reward_avg+reward_std, reward_avg-reward_std, facecolor='black', alpha=.3, edgecolor='none')
    plt.xlabel('step')
    plt.ylabel('x-displacment')
    plt.legend(['reward'])
    
    
    plt.subplot(3, 2, 2)
    plt.plot(x2,reward, 'brown')
    #plt.plot(x2,reward_samp, 'b--')
    plt.plot(x2,reward_total, 'r--')
    plt.plot(x2,reward_avg, 'r')
   # plt.fill_between(x2, reward_total, reward_total-reward_avg, facecolor='r', alpha=.3, edgecolor='none')
    plt.fill_between(x2, reward_avg+reward_std, reward_avg-reward_std, facecolor='red', alpha=.3, edgecolor='none')
  
    #plt.fill_between(x2, reward_total+reward_avg, reward_total-reward_avg, facecolor='y', alpha=.7, edgecolor='none')
    plt.xlabel('step')
    plt.legend(['reward','total reward', 'mean +- std '])
    
    plt.tight_layout()
    plt.show()


    
    return()


####################################################################################################
#---------------------------- Save Session Parameters and Results  --------------------------------#
####################################################################################################

def display_session_parameters(example_state, actions, learning_rate, model_architecture, discountfactor, 
                               iterations, episodes, buffer_size, batch_size, update_frequency,
                               epsilion, epsilion_inital, epsilion_final):
        
        PHI_DISC = 32
        STATE_RESOLUTION = len(actions) * PHI_DISC
        STATE_ACTION_SPACE = STATE_RESOLUTION * len(actions)
        ATTEMPTS = iterations * episodes
        epsilion = epsilion_inital
        
        fig = plt.figure()
        matplotlib.use('TkAgg')
        r = 0
        s = 1
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.text(r, s, '------------------------------') 
        plt.text(r, s - .05, 'Model Architecture', weight = 'bold')
        plt.text(r, s -.1, ' ----------------------------')
        
        plt.text(r, s-.15, 'Input Layer' + ' - ' + str(len(example_state)))
        plt.text(r, s-.2, 'Layer1_neurons' + ' - ' + str(model_architecture[0]))
        plt.text(r, s-.25, 'Layer2_neurons' + ' - ' + str(model_architecture[1]))
        plt.text(r, s-.3, 'Output Layer' + ' - ' + str(len(actions)))

        plt.text(r, s-.35, '----------------------------------') 
        plt.text(r, s - .4, 'Training Parameters', weight = 'bold')
        plt.text(r, s -.45, ' --------------------------------')
        plt.text(r, s-.5, 'iterations' + '=' + str(iterations))
        plt.text(r, s-.55, 'episdoes' + '=' + str(episodes))
        plt.text(r, s-.6, 'buffer size' + '=' + str(buffer_size))
        plt.text(r, s-.65, 'batch size' + '=' + str(batch_size))
        plt.text(r, s-.7, 'update frequency' + '=' + str(update_frequency))
        plt.text(r, s-.75, 'learning rate' + '=' + str(learning_rate))
        plt.text(r, s-.8, 'discount factor' + '=' + str(discountfactor))
        plt.text(r, s-.85, 'epsilion inital' + '=' + str(epsilion))
        plt.text(r, s-.9, 'epsilion final' + '=' + str(epsilion_final))
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(r, s, '----------------------------------------------') 
        plt.text(r, s - .05, 'Agent & Enviroment Details', weight = 'bold')
        plt.text(r, s -.1, ' --------------------------------------------')
        plt.text(r, s-.15, 'State Space' + ' - ' + 'discrete')
        plt.text(r, s-.2, 'Action Space' + ' - ' + 'discrete')
        plt.text(r, s-.25, 'DOF for State' + ' - ' + str(len(example_state)))
        plt.text(r, s-.3, 'DOF for Action ' + ' - ' + str(len(actions[0])))
        plt.text(r, s-.35, 'Markov Comlexity ' + ' - ' + str(STATE_ACTION_SPACE))
        plt.text(r, s-.4, 'Proprioception Resolution ' + '-' + str(STATE_RESOLUTION))
        plt.text(r, s-.45, 'Reflex Resolution' + ' - ' + str(len(actions[0])))
 
        plt.show()

    
        print('----------------------')
        print('| Model Architecture |')
        print('----------------------')
        print('Input Layer', len(example_state))
        print('Layer1_neurons', model_architecture[0])
        print('Layer2_neurons', model_architecture[1])
        print('Output Layer', len(actions))
        
        print('-----------------------')
        print('| Training Parameters |')
        print('-----------------------')
        print('iterations', iterations)
        print('episodes', episodes)
        print('buffer size', buffer_size)
        print('batch size', batch_size)
        print('update frequency', update_frequency)
        print('learning rate', learning_rate)
        print('discount factor', discountfactor)
        print('epsilion inital', epsilion)
        print('epsilion final', epsilion_final)    
        
        print('------------------------------')
        print('| Agent & Enviroment Details |')
        print('------------------------------')
        
        print('State Space = ', 'Discrete') # Continuous
        print('Action Space =', 'Discrete')
        
        print('DOF for State =', len(example_state))
        print('DOF for Action =', len(actions[0]))
       
        print('Proprioception Resolution  =', STATE_RESOLUTION) # sense of self-movement and body positon "kinaesthesia "
        print('Reflex Resolution =', len(actions)) #https://en.wikipedia.org/wiki/Proprioception
        
        print('Markov Complexity = ', STATE_ACTION_SPACE)  # Markov space
        print('Enviroment Interactions = ', ATTEMPTS)
        
        print('Problem Difficulty = ',( (STATE_ACTION_SPACE / ATTEMPTS) / (STATE_RESOLUTION + len(actions)) ) )#print('Markov Difficulty Equation = State-Action Space / Number of Attemps')

        #print('-------------------')

        return() # prompt user to initialize session with current model architecture and training parameters

    



####################################################################################################
#------------------------------------ DQN Algorithm  ----------------------------------------------#
####################################################################################################

def DQN_RUNNER(example_state, actions, learning_rate, model_architecture, discountfactor, 
                iterations, episodes, buffer_size, batch_size, update_frequency,
                  epsilion, epsilion_inital, epsilion_final):
    
    step = 0
  
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
 
    # Initialize replay memory D to capacity N
    D = collections.deque(maxlen=buffer_size)
    
    # Initialize action-value function Q with weights theta ---> Q_network
    Target_network, Q_network = generate_networks(example_state, actions, learning_rate, model_architecture)
    
    # Initialize target action-value function Qhat with weights theta - = theta --> Target_network
    Target_network.set_weights(Q_network.get_weights())
    
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
                Q_network, batch_loss, batch_mean, batch_std = experience_replay(Q_network, Target_network, D, learning_rate, discountfactor, batch_size, actions)
                
                # store loss history 
                Batch_loss += batch_loss #.append(batch_loss)
                Batch_mean += batch_mean #.append(batch_mean)
                Batch_std += batch_std #.append(batch_std)
                
            
            # perform a gradient descent step on ( yi - lr*max_Qhat(s',a',theta -))^2 wrt to current Q_network weights 
            Q_network, loss = gradient_step(Q_network, Target_network, experience, learning_rate, discountfactor,Q_index)
                
            # Every C steps reset Qhat = Q
            if step == update_frequency:
                Target_network = Q_network
                step = 0                 
            else:
                step += 1
            
            # update state
            state = newstate
            
            Reward.append(reward)
            Reward_total.append(sum(Reward))
            Reward_avg.append(sum(Reward_total)/len(Reward))
            Reward_samp.append(sum(Reward)/len(Reward))
            Reward_std.append(np.std(Reward))
           
            
            Step_loss.append(loss)
            Step_total.append(sum(Step_loss))
            Step_mean.append(sum(Step_loss)/len(Step_loss))
            Step_std.append(np.std(Step_loss))
    
    plot_results(Reward, Reward_total, Reward_avg, Reward_std,  Reward_samp,
                      Step_loss, Step_total, Step_mean,Step_std,
                      Batch_loss, Batch_mean, Batch_std)
    
    # Display and Check inital parameters before starting training session
    display_session_parameters(example_state, actions, learning_rate, model_architecture, discountfactor, 
                               iterations, episodes, buffer_size, batch_size, update_frequency,
                               epsilion, epsilion_inital, epsilion_final)
    
    print('done')
    
    return()
             
########################################################################################################################
#----------------------------------------  Setup and Run Test ---------------------------------------------------------#
########################################################################################################################

STATE = (0,0,0)
ACTIONS = list_actions(action1 = range(0,10,1), action2 = range(0,10,1), action3 = range(0,10,1))
MODEL_ARCHITECTURE = (50,10)
LEARNING_RATE = .02 # .002 #.2 #2 # 0.02
DISCOUNT_FACTOR = .1
ITERATIONS = 25
EPISODES =  50
BUFFER_SIZE = 32
BATCH_SIZE =16
UPDATE_FREQUENCY = 32
EPSILION_INITAL = 2
EPSILION_FINAL = .01


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
