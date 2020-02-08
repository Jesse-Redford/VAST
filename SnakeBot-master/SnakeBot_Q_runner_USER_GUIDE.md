# SnakeBot_Q_runner_USER_GUIDE.md
This guide includes a complete walk through of how to run test using the SnakeBot_Q_runner.py file.
Before procceding make sure that you have completed the system setup tutorial included in the SnakeBot_SETUP_GUIDE.md

# Q Learning Algorithm 

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)

Q-learning is a model-free reinforcement learning algorithm. The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances. It does not require a model (hence the connotation "model-free") of the environment, and it can handle problems with stochastic transitions and rewards, without requiring adaptations.

For any finite Markov decision process (FMDP), Q-learning finds a policy that is optimal in the sense that it maximizes the expected value of the total reward over any and all successive steps, starting from the current state.[1] Q-learning can identify an optimal action-selection policy for any given FMDP, given infinite exploration time and a partly-random policy.[1] "Q" names the function that returns the reward used to provide the reinforcement and can be said to stand for the "quality" of an action taken in a given state.
https://en.wikipedia.org/wiki/Q-learning

# Required Files
- SnakeBot_Q_runner.py    # initializing and running test
- Q_Snakebot_Agent.py     # Qlearning algorithm and enviromentsetup
- Q_pandas_utilities.py   # Data handeling functions for Pandas 
- Q_plotting_utilities.py # plotting functions 

# Setup Check list
- Q Learning Overview
- Defining a State-Action Space
- State Space Defintion
- Action Space Defintion
- Creating an Enviroment 
- Defining a Reward Function
- Qlearning with Pandas 
- Inializing a Training Session
- Experimental Results Benchmarks

# Defining a State-Action Space
Classical Qlearning can only operate on what is known as discrete state-action space. There is an advatage to this however, as we can explopite the fact that we can ensure control over the robots policys in a heriarchal nature. 

We need to consider, constrain, and define all the possible combinations of "actions" in which the Qagent can choose from, and also all the possible "states" the Qagent may find itself in. We record these interactions the agent has with its physical enviroment in the form of an experience, defined as a experince = (state,action,newstate,reward). Future implentations can be applied to DQN network, dicuessed later.

# State Space Defintion

- Consider 

For this example of our snakebot, the robot's "state" defintion will be 3-dimensional and defined by the variables theta1, theta2 and phi. Where theta1, theta2 are the robots joint angles which reflect the current angular positions of the servos installed at each joint, and Phi is the relative "heading" or angular oritentation of the robots labratory frame of provided by the Intelrealsense camera. 

- Constrain

Given the physical constrains of the servos, theta1 and theta2 can only range from 0 to 180 degrees with a 1 degree resolution.
The third variable of our state defintion "phi" is ultimatley constrained by resolution of the intel sensor. However, to simplify the learning enviroment for the agent we can approximate the sensors measurments to a finite space with a specifed amount of resolution.
This discritization of our "state", in units of degrees, is then created using the example below. 

- Define

        theta1 = theta2 = range(lower_limit = 0 ,upper_limit = 180,step_size = 1) 
        phi = range(lower_limit = 0 ,upper_limit = 180, resolution = 1) 
        states = list(itertools.product(theta1,theta2,phi))

# Action Space Defintion
Next we will create an "action" defintion which in the case will also be three-dimensional, consisting of the variables v1,v2, and T.
where v1, v2 are the potential angular velosities we can send to the servos and T is the duration of time in which these velosities will be executed for. 

- Consider 

Given that we are using positional servos which can only move in 1degree increments, we created an alogirthm allows for contunious like behavoior by adding time delays between positonal movments. 

-Define 

    v1 = v2 = (lower_bound = -10, upper_bound = 10, step_size = 1) # in units of deg/sec
    t = range(0,1,.1) # units of sec
    actions = list(itertools.product(v1,v2,T))


-Create the action function which will be used in the non_simulated_transition function later 

    def Snake_Bot_Action(current_state, selected_action)
      theta1, theta2, phi = current_state
      v1, v2, T = selected_action
  
       while True:
        theta_new = v * T
        if theta =! theta_new && theta < theta_new:
       servo.write(theta + 1)
        sleep(
      
      new_state = theta1_new, theta2_new
      
      return(new_state)
      
# Determing the size of your state action space

    print("Total States = ", len(states), "Total Actions", len(actions), "State-Action Space = " , len(states) * len(actions))
   
# Creating your Enviroment and a Reward Function
In order to apply Qlearning we need to create an enviroment in which the agent can interact and learn from.
Your enviroment will consist of primarly of two things a transition function and a reward function.
The transition function should accept a current state,action pair and return a newstate coupled by a reward. 
Sudo code for creating the Snake_bot_Environment, (see module for complete details). example below For our purposes the transition function will abide by the following simulated and nonsimulated need to be added to SnakeBot module


# Physical Enviroment

    def non_simulated_transition(current_state, selected_action, all_possible_states)
  
      theta1_old, theta2_old, phi_old = current_state 
      thetadot1, thetadot2, t_interval = selected_action
  
      x_old, y_old = GetLoc() # sample current x,y location 
  
      new_state  = Snake_Bot_Action(current_state,selected_action) # function should return, new_state =  theta1_new, theta2_new,     phi_new
 
    new_state = find_nearest(new_state,all_possible_states)  
  
    x_new, y_new = GetLoc() # sample new x,y location after completeing a selected action
  
    reward = x_new - x_old
 
    return(new_state,reward)
    
    
    
# Simulated Enviroment
    
    def simulated_transition(current_state, selected_action, all_possible_states)
    
        reward = x_new - x_old
        return(new_state,reward)


# Qlearning with Pandas 
Next we will setup a Qlearning algorithm with a pandas data struture. In doing so we can mantain a consistant method of intizalizing, training, and managing a varity of simulated and non-simulated enviroments. Lets start by setting up a working directory and an untrained Qtable.
    
    import Snakebot
    
    path =r'C:\Users\Jesse\Desktop\VAST\Qtables' 
    filename = '\Qtable_untrained.pkl'

    Qtable_untrained = Snakebot.create_zeros_Qtable(states,actions)
    
    Qtable_untrained.to_pickle(path+filename) 
    
    
     

# Inializing a Training Session


     InitializeRun = SnakeBot.RunTest(

                      simulate =  True, # Set to false for physical testing
                      
                      load_path = r'C:\Users\Jesse\Desktop\VASt\Qtables', # path where saving data, update as 
                      
                      load_filename = '\Qtable_untrained.pkl' , # make zeros file, or load pre trained
                      
                      save_path = load_path, # where to save results
                      
                      save_filename = '\Qtable_trained.pkl' ,# name of new Qfunction
                      
                      theta1 = range(0,1,1), # lowerbound, upperbound, stepsize
                      
                      theta2 = range(0,1,1), # lowerbound, upperbound, stepsize
                      
                      phi = range(0,1,1), # lowerbound, upperbound, stepsize
                      
                      v1 = range(0,1,1),  # velosity bounds
                      
                      v2 = range(0,1,1), # velosity bounds
                      
                      t = range(0,1,1) # dt - time duration of action
                      
                      initial_state = (90,90,90), # intialized state at beging of training or policy rollout
                      
                      eps_initial = 1, # eps initial for training
                      
                      eps_final = .1, # eps initial for training
                      
                      alpha_initial = .7,  # alpha initial for training
                      
                      alpha_final = .8, # alpha initial for training
                      
                      gamma_initial = .9, # alpha initial for training
                      
                      gamma_final = .8, # alpha initial for training
                      
                      training_steps = 100, # number of steps to train
                      
                      policy_steps = 100, # number of steps to rollout policy, set training steps to 0 to just run policy
                      
                      )



# SnakeBot Module

Libraries


        import pandas as pd
        import itertools
        import random
        import numpy as np
        import math
        import time
        import matplotlib.pyplot as plt
        import pickle


Main loop for Testing

        def RunTest( simulate, load_path, load_filename, save_path, save_filename, state1, state2, state3, action1, action2,                                  action3, initial_state, eps_initial, eps_final,alpha_initial, alpha_final, gamma_initial, gamma_final,                                  training_steps, policy_steps)
                 
                 print("Loading Qtable") 
                 Qtable =load_Qtable(load_path,load_filename)
                 states,actions = three_states_three_actions(state1,state2,state3,action1,action2,action3)

                 trainedQtable =                                    Train(simulate,Qtable,states,actions,initial_state,eps_initial,eps_final,alpha_initial,alpha_final,gamma_initial,gamma_final,training_steps)
                  
                  
                  print("Saving Updates")
                  trainedQtable.to_pickle(save_path+filename) # can also do JSON or other file types
                  print('Finished Saving Qtable')
        
         
                  if policy_steps > 0:
                        print("Begining PolicyRollout") 
                        PolicyRollout(trainedQtable,simulate,initial_state = (90,90,90), policy_steps  = 25, states = states)
                        print("PolicyRollout is Complete")
                 else:
                         print("Training is Complete")
                
         return()
         
Simulated and non-simulated transition functions

        def simulate_transition(state,action,states)
                return(new_state,reward)
        
        
        def non_simulate_transition(state,action,states)
                return(new_state,reward)
        

Various Call Functions


        def three_states_three_actions(state1,state2,state3,action1,action2,action3):
                states = list(itertools.product(state1,state2,state3)) # Create a list of every state combination
                actions = list(itertools.product(action1,action2,action3)) # Create a list of every action combination
                return states,actions
    

        def create_zeros_Qtable(states,actions):
         return (pd.DataFrame(np.zeros((len(states), len(actions))), index=states, columns=actions, dtype=float))
         
        def create_random_Qtable(states,actions,lowerbound,upperbound,round_to):
        return (pd.DataFrame(np.random.uniform(lowerbound, upperbound, size=(len(states), len(actions))).round(round_to), index=states,         columns=actions, dtype=float))

        def save_Qtable(Qtable,path,filename):
                Qtable.to_pickle(path+filename)
                print('Qfunction is saved')
                return()

        def load_Qtable(path,filename):
                print('Qfunction is loaded')
                return pd.read_pickle(path+filename)
                
                
 Training Loop Function 
 
    def         Train(simulate,Qtable,states,actions,initial_state,eps_initial,eps_final,alpha_initial,alpha_final,gamma_initial,gamma_final,training_steps):
    
    state = initial_state
    
    epsilion = eps_inital
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
            new_state, reward = non_simulated_transition(state,action,states)
        
        Qtable = Update_Qtable(Qtable,actions,state,action,new_state,reward,alpha,gamma)
        state = new_state
        step = step + 1    
        
    return(Qtable)
    
    
    
policy rollout functions

    def get_Qtable_policy(Qtable):
       Qtable['policy'] = Qtable.idxmax(axis=1)
       policy = Qtable['policy']
       return(policy)
    
     def get_policy_action(what_policy,current_state):
        policy_action = what_policy[current_state]
        return(policy_action)
    
      def policy_rollout(simulate,what_policy,initial_state,policy_steps,states):
          steps = 0
          state = initial_state
          iteration = []
          reward_step = []
          total_reward = []
          average_reward = []
          while steps < policy_steps:
            action = get_policy_action(what_policy,state)
            
            if simulate == True:
               new_state, reward = simulated_transition(state,action,states)
             else:
               new_state, reward = non_simulated_transition(state,action,states)
        
             state = new_state
        
            iteration.append(steps)
            reward_step.append(reward)
            total_reward.append(sum(reward_step))
            average_reward.append(sum(total_reward)/len(iteration))
        
            steps = steps + 1
        
            reward_plotter(iteration,reward_step,total_reward,average_reward)  
    
            return()
    
    
    def reward_plotter(iteration,reward_step,total_reward,average_reward):

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


        def PolicyRollout(Qtable,simulate,initial_state,policy_steps,states)
                 policy = get_Qtable_policy(Qtable)
                policy_rollout(simulate,policy,initial_state,policy_steps,states)
        return()
    
    
    
    
    
    
