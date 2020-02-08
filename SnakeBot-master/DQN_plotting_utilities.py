#import io, os, sys, types
#from IPython import get_ipython
#from nbformat import read
#from IPython.core.interactiveshell import InteractiveShell

#import import_ipynb # pip install import_ipynb


########################################
#-----DQN plotting utilities ----------#
########################################

# revisions needed 
#- add inputs(path,directory,foldername, figure_titles = True_or_False)
#- update display_session_parameters equations and include more session parameters 

#import DQN_plotting_utilities         # load module into script
#from DQN_plotting_utilities import*   # this will import all functions from plotting utilities

#DQN_plotting_utilities.plot_results(reward, reward_total, reward_avg, reward_std, reward_samp,  
   #                                 step_loss, step_total, step_mean, step_std, 
    #                                batch_loss, batch_mean, batch_std)
                      
                      
#DQN_plotting_utilities.display_session_parameters(example_state, actions, learning_rate, model_architecture, discountfactor,   
 #                                                 iterations, episodes, buffer_size, batch_size, update_frequency,
  #                                                  epsilion, epsilion_inital, epsilion_final)


#DQN_plotting_utilities.VAST_VISUAL(theta1,theta2,x_displacment,step_number)


###################################
#----- import libraries ----------#
###################################

import io, os, sys, types
#from IPython import get_ipython
#from nbformat import read
#from IPython.core.interactiveshell import InteractiveShell

#import import_ipynb # pip install import_ipynb

#import traceback
#import sys
import matplotlib
matplotlib.use('Agg')
import datetime
import random
import numpy as np
import csv, json
from copy import deepcopy
from pprint import pprint
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from statistics import mean



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

    fig = plt.figure(1)
    
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
    #plt.close()


    
    return()


####################################################################################################
#-------------- View Session Parameters and save Results as Figure --------------------------------#
####################################################################################################

def display_session_parameters(example_state, actions, learning_rate, model_architecture, discountfactor, 
                               iterations, episodes, buffer_size, batch_size, update_frequency,
                               epsilion, epsilion_inital, epsilion_final):
        
        PHI_DISC = 32
        STATE_RESOLUTION = len(actions) * PHI_DISC
        STATE_ACTION_SPACE = STATE_RESOLUTION * len(actions)
        ATTEMPTS = iterations * episodes
        epsilion = epsilion_inital
        
        fig = plt.figure(2)
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
#-------------- 3d Visual of policy plot --------------------------------#
####################################################################################################



#get_ipython().run_line_magic('matplotlib', 'notebook')

import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.axes3d import *
from matplotlib import cm

def VAST_VISUAL(theta1,theta2,x_displacment,Step):
    
    
    
    fig = plt.figure(4)
    
    # Batch losses
    plt.subplot(1, 2, 1)
    plt.plot(Step,theta1, 'b--') 
    plt.plot(Step,theta2, 'r--') 
    plt.xlabel('step')
    plt.ylabel('theta')
    plt.legend(['theta1','theta2'])
    
        # Batch losses
    plt.subplot(1, 2, 2)
    plt.plot(Step,theta2, 'green') 
    plt.xlabel('step')
    plt.ylabel('theta')
    plt.legend(['theta2'])
    plt.show()

   # plt.style.use('classic') # Styles - classic - seaborn-whitegrid - Solarize_Light2
   # fig = plt.figure(3)
   # ax = fig.add_subplot(111, projection='3d')
    #x = theta1 #* 180/np.pi
    #y = theta2
    #z =   x_displacment #* 180/np.pi
    #a = []
    #b = []
   # c = []
   ## for item in x:
    #    a.append(float(item))
    #for item in y:
    #    b.append(float(item))
    #for item in z:
   #     c.append(float(item))
   # r = np.array(a)
#    s = np.array(b)
   # t = np.array(c)

    #ax.scatter(r,s,zs = t, s=200, c = Step, cmap='hot',label='True Position')
    #ax.plot3D(r,s,z)
   # ax.plot(r,s,z)
   # ax.set_xlabel('theta1')
  #  ax.set_ylabel('theta2')
   # ax.set_zlabel('x displacment')
   # p = ax.scatter(r,s,zs = t, s=5, c = Step, cmap='hot')
   # fig.colorbar(p)
   # plt.show()
    return()

