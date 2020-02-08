####################################################################################################
#--------------- Set up example to run test SnakeBot_DQN_agent -----------------------------------#
####################################################################################################

#import Snakebot_DQN_agent
#from Snakebot_DQN_agent import RunTest

# add parameters and updates cross reference  SnakeBot_DQN_runner 
#RunTest(example_state, actions, learning_rate, model_architecture, discountfactor, 
 #               iterations, episodes, buffer_size, batch_size, update_frequency,
  #                epsilion, epsilion_inital, epsilion_final):




####################################################################################################
#------------------------------------ External Utilities -----------------------------------------#
####################################################################################################

#import io, os, sys, types
#from IPython import get_ipython
#from nbformat import read
#from IPython.core.interactiveshell import InteractiveShell
import numpy as np
#import import_ipynb # pip install import_ipynb

import DQN_plotting_utilities
from DQN_plotting_utilities import *
#from DQN_plotting_utilities import plot_results

import Snakebot_enviroments as SBE
from Snakebot_enviroments import *

import DQN_network_utilities as NN
from DQN_network_utilities import *

####################################################################################################
#--------------- Set up functions for simulated and physical enviroments --------------------------#
####################################################################################################

def reset():
    # reset camera and phi
    # reset joint angles, theta1, theta2 to 90
    state = (1.,.2,.2)
    return state

#def execute_action(state, action): # simualate or act real enviroment here 
 #   # add transition sequence for physical bot
  #  newstate = (0,0,0)
   # reward = 1
    #return newstate, reward


###################################
#-------- MISC functions ----------#
###################################

def Beta(start = 1,end = .1,maxsteps = 100):
    return math.exp(math.log(end/start)/maxsteps)

def list_actions(action1,action2,action3):
    actions = list(itertools.product(action1,action2,action3)) # Create a list of every action combination
    return actions

###################################################################################
#------------------------------------ DQN Algorithm  ----------------------------------------------#
####################################################################################################

def Run(example_state, action1,action2,action3, learning_rate, model_architecture, discountfactor, 
                iterations, episodes, buffer_size, batch_size, update_frequency,
                  epsilion, epsilion_inital, epsilion_final):
    
    actions = list_actions(action1,action2,action3)
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
    
    Theta1 = []
    Theta2 = []
    Phi = []
    Step = []
 
    # Initialize replay memory D to capacity N
    D = collections.deque(maxlen=buffer_size)
    
    # Initialize action-value function Q with weights theta ---> Q_network
    Target_network, Q_network = NN.generate_networks(example_state, actions, learning_rate, model_architecture) # add load,save, and itializer options
    
    # Initialize target action-value function Qhat with weights theta - = theta --> Target_network
    Target_network.set_weights(Q_network.get_weights())
    
    for e in range(episodes):
        
        # initialize sequence and preprocess state
        state = reset()
        
        for i in range(iterations):
            
            epsilion = epsilion * Beta(epsilion_inital,epsilion_final,iterations*episodes) # decay epsilion 
            
            # with probability e select action a
            action, Q_index = NN.choose_action(Q_network, state, actions, epsilion)
            
            # execute action and observe reward, and newstate
            newstate,reward = SBE.execute_action(state,action)
            
            # Store transition (s,a,s',r) in D
            experience = state,action,newstate,reward
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
            
            # update state
            state = newstate
            
            Theta1.append(state[0])
            Theta2.append(state[1])
            Phi.append(state[2])
            Step.append(i) 
            
            Reward.append(reward)
            Reward_total.append(sum(Reward))
            Reward_avg.append(sum(Reward_total)/len(Reward))
            Reward_samp.append(sum(Reward)/len(Reward))
            Reward_std.append(np.std(Reward))
          
            Step_loss.append(loss)
            Step_total.append(sum(Step_loss))
            Step_mean.append(sum(Step_loss)/len(Step_loss))
            Step_std.append(np.std(Step_loss))
    
    DQN_plotting_utilities.plot_results(Reward, Reward_total, Reward_avg, Reward_std,  Reward_samp,
                                        Step_loss, Step_total, Step_mean,Step_std,
                                       Batch_loss, Batch_mean, Batch_std)
    
     # Display and Check inital parameters before starting training session
    DQN_plotting_utilities.display_session_parameters(example_state, actions, learning_rate, model_architecture, discountfactor, 
                              iterations, episodes, buffer_size, batch_size, update_frequency,
                               epsilion, epsilion_inital, epsilion_final)
    
    DQN_plotting_utilities.VAST_VISUAL(Theta1,Theta2,Reward,Step)
    
    print('done')
    
    return()

