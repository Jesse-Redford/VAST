
########################################################################################################################
#---------------------------------------- SnakeBot_DQN_runner -----------------------------------------------------#
########################################################################################################################

#from IPython import get_ipython
#from nbformat import read
#from IPython.core.interactiveshell import InteractiveShell
#import import_ipynb # pip install import_ipynb
import io, os, sys, types
import numpy as np
import DQN_agent
from DQN_agent import Run


#----Add extra input parameters---------# 
LOAD_PATH = r'/home/pi/Desktop/' , #'none'
LOAD_FILENAME = 'Qtables\Qtable_trained.pkl', #'none'
SAVE_PATH = r'/home/pi/Desktop/'
SAVE_FILENAME = 'Qtables\Qtable_trained.pkl'
POLICY_STEPS = 25
SIMULATED = False

PHI_LOWER = -np.pi
PHI_UPPER = np.pi
PHI_INTERVALS = 10
T_DELAY_LOWER = 0.015
T_DELAY_UPPER = 0.025
T_DELAY_INTERVALS = 0.005 

THETA_LOWER = 90
THETA_UPPER = 90
THETA_STEPSIZE = 90
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^#



# Setup Parameters for Session

THETA_DOT_1 = range(-2,3,1) # np.linspace(-2.0,3.0, 2)
THETA_DOT_2 = range(-2,3,1)
T_INTERVAL = range(0,501,250) 
EXAMPLE_STATE = (0,0,0)

MODEL_ARCHITECTURE = (50,10)
BUFFER_SIZE = 10
BATCH_SIZE = 5
UPDATE_FREQUENCY = 5

LEARNING_RATE = .02
DISCOUNT_FACTOR = .7
EPSILION_INITAL = 1
EPSILION_FINAL = .1

ITERATIONS = 250
EPISODES = 2

   
DQN_agent.Run(example_state =EXAMPLE_STATE,
              action1 = THETA_DOT_1,
              action2 = THETA_DOT_2,
              action3 = T_INTERVAL ,
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


# Qrunner example
#RunTest(simulate = SIMULATED, # False = run physical robot enviroment, True = run simulated  enviroment
 #       load_path = LOAD_PATH , #'none', # set load_path & load_filename to 'none' to begin training with empty Qtable
  #      load_filename = LOAD_FILENAME, #'none', # specify a previously trained Qtable.pkl file to run session with pre trained Q function 
   #    save_path = SAVE_PATH, # specifify folder path to save trained Qtable 
    #     save_filename = SAVE_FILENAME, # specify name of trained Qtable that will be saved at the end of the training session
    #    state1= range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), # pos servo 1
     #   state2= range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), #pos servo 2
      #  state3 = np.linspace(PHI_LOWER, PHI_UPPER, PHI_INTERVALS), # disccirtize phi state --> np.linspace(lower = -np.pi, upper = np.pi, phi_interval = 10)
       # action1 = range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), #pos servo 2
        #action2 = range(THETA_LOWER,THETA_UPPER+1,THETA_STEPSIZE), # position servo 2
        #action3 = np.arange(T_DELAY_LOWER, T_DELAY_UPPER, T_DELAY_INTERVALS), # t inteval between 1deg steps, in milisecons 
        #initial_state = (90,90,0), # start with inital state 
 #       eps_initial = EPSILION_INITAL, 
  #      eps_final = EPSILION_FINAL, 
   #     alpha_initial = ALPHA_INITAL, 
    #    alpha_final = ALPHA_FINAL, 
     #   gamma_initial = GAMMA_INITAL,
      #  gamma_final = GAMMA_FINAL, 
      #  training_steps = TRAINING_STEPS, 
       # policy_steps = POLICY_STEPS)

    
    
