# SnakeBot_DQN_runner_USER_GUIDE.md
This guide includes a complete walk through of how to run test using the SnakeBot_DQN_runner.py file.
Before procceding make sure that you have completed the system setup tutorial included in the SnakeBot_SETUP_GUIDE.md



# Deep Q Learning


![alt text](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_236397%2Fproject_365994%2Fimages%2FDQN.png)

The DeepMind system used a deep convolutional neural network, with layers of tiled convolutional filters to mimic the effects of receptive fields. Reinforcement learning is unstable or divergent when a nonlinear function approximator such as a neural network is used to represent Q. This instability comes from the correlations present in the sequence of observations, the fact that small updates to Q may significantly change the policy and the data distribution, and the correlations between Q and the target values. The technique used experience replay, a biologically inspired mechanism that uses a random sample of prior actions instead of the most recent action to proceed.[2] This removes correlations in the observation sequence and smooths changes in the data distribution. Iterative updates adjust Q towards target values that are only periodically updated, further reducing correlations with the target.[17]

# Required Files
- SnakeBot_DQN_runner.py # setup and execute test
- DQN_Snakebot_Agent.py # algorithm and enviroment setup
- DQN_network_utilities.py # helper functions for DQN networks
- DQN_plotting_utilities.py # helper for saving session history

# Setup Check list
- DQN Learning Overview
- State Space Defintion
- Action Space Defintion
- Creating an Enviroment
- Defining a Reward Function
- Creating a Model in Keras
- DQN_network_utilities.py
- DQN_Snakebot_Agent.py
- Inializing a Training Session
- Experimental Results Benchmarks

# DQN-Algorithm

        def DQN_RUNNER(example_state, actions, learning_rate, model_architecture, discountfactor, 
                                iterations, episodes, buffer_size, batch_size, update_frequency,
                                        epsilion, epsilion_inital, epsilion_final):
    
    step = 0
    
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
                Q_network = experience_replay(Q_network, Target_network, D, learning_rate,                                                                                                   discountfactor, batch_size, actions)
                
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
            
    return()

# State Space Defintion
Unlike Qlearning, DQN can handel continous state spaces. However, we need to make sure the dimension of the state remains consant.
For our example our state will be a 3 dimensional vector consisting of the variables theta1,theta2, and phi.
where theta1 and theta2 are the joint angles of servoes and phi is the relative heading of the labratory frame provided by the 
intel real sense camera. Value of phi will range between pi and -pi. So an example of our state representation will be a tuple
S = (theta1,theta2,phi). To pass this state to the network simply reshape the state into a vector.

        def reshape_states(state,newstate):
                state = np.asarray(state).reshape(1, len(state))
                newstate = np.asarray(newstate).reshape(1, len(newstate))
          return state,newstate
    

# Action Space Defintion
Although we can have a continous input, the action space needs to be constrained. We will define an action as the 
combination of three variables thetadot1, thetadot2, and T, where thetadot12 are the velosity values we can
send to the servos and T is the time interval for which the servos will move. We will define our action asthis function will now create a list containing all possible combinations of actions in which the agent can select from.
This also defines the size of our output layer of the nerual net

        thetadot1 = thetadot2 = range(lowerbound, upper_bound, stepsize)

        T = range(lowerbound, upper_bound, stepsize)

        def list_actions(action1,action2,action3):
                actions = list(itertools.product(action1,action2,action3)) # Create a list of every action combination
                return actions
                
         
        actions = get_actions(thetadot1,thetadot2,T) # get list of all action combinations
        

# Creating A Simulated Enviroment
Just like we did in the Qlearning example, we need to determine a function which represent the agents experince after executing an action. The enviroment or transition function should accept the current state a selected action and return a newstate with a reward.
     
     # save this file as Snakebot_Enviroments.py
     # import in main script with the following
     import Snakebot_Enviroments.py
     from Snakebot_Enviroments import execute_action
     
     newstate,reward = Snakebot_Enviroments.execute_action(state,action,enviroment = simulated)
     
     
    def execute_action(state,action,enviroment):
        if enviroment == simulated:
                newstate,reward = execute_simulated_enviroment(state,action)
         else:
                 newstate,reward = execute_physical_enviroment(state,action)    
                 
          return(newstate,reward)
    
   
    #------------------------------Simulated enviroment--------------------------------#
    
    from scipy import integrate
    import math
    import numpy as np

    def reduceAngle(x):
    while (x>=2*math.pi):
        x=x-(2*math.pi)
    while (x<=0):
        x=x+(2*math.pi)
    return x

    def threeWheel(z,t,theta1,theta2,theta1Dot,theta2Dot,rho):
    x,y,phi=z
    
    xyNumerator=(1+math.cos(theta2))*theta1Dot+(1+math.cos(theta1))*theta2Dot
    denominator=math.sin(theta1)+math.sin(theta1-theta2)-math.sin(theta2)
    phiNumerator=math.sin(theta2)*theta1Dot+math.sin(theta1)*theta2Dot
    xDot=(rho*math.cos(phi))*xyNumerator/denominator
    yDot=(rho*math.sin(phi))*xyNumerator/denominator
    phiDot=phiNumerator/denominator
    
    dzdt=[xDot,yDot,phiDot]
    return dzdt
    
    def advanceState(dt,s,theta1Dot,theta2Dot,rho=1,n=100,sys=threeWheel):
    ##set state variables
    x0=0
    y0=0
    phi0=s[0]
    z0=[x0,y0,phi0]
    
    ##linearize time, thetaDots and thetas
    t=np.linspace(0,dt/1000,n)
    
    theta1Dot=np.linspace(theta1Dot,theta1Dot,n)
    theta2Dot=np.linspace(theta2Dot,theta2Dot,n)
    
    theta1=[s[1]]
    theta2=[s[2]]
    for i in range(len(t)):
        theta1=np.append(theta1,[theta1[i]+theta1Dot[i]*(dt/1000/n)],axis=0)
        theta2=np.append(theta2,[theta2[i]+theta2Dot[i]*(dt/1000/n)],axis=0)
    
    #initialize storage z
    zs=[z0]
    
    #integrate
    for i in range(len(t)-1):
        
        ts=[t[i],t[i+1]]
        z=integrate.solve_ivp(lambda t,z: threeWheel(z,t,theta1[i],theta2[i],theta1Dot[i],theta2Dot[i],rho), ts, z0)
        z0=[z.y[0,1],z.y[1,1],z.y[2,1]]
        zs=np.append(zs,[z0],axis=0)
        
    
    #calc displacement
    xdis=zs[len(zs)-1,0]
    ydis=zs[len(zs)-1,1]
    displacement=[xdis,ydis]
    
    #calc new state
    phiFinal=zs[len(zs)-1,2]
    phiFinal=reduceAngle(phiFinal)
    theta1Final=theta1[len(theta1)-1]
    theta2Final=theta2[len(theta2)-1]
    sNew=[phiFinal,theta1Final,theta2Final]
    
    return displacement, sNew


    def execute_simulated_enviroment(state,action):
    
    ###Convert Jesse's notation to Will's notation
    s=[state[2],state[0],state[1]]
    theta1Dot=action[0]
    theta2Dot=action[1]
    dt=action[2]
    
    #Feed variables to solver, get displacement and new state in Will's notation
    displacement, sNew = advanceState(dt=dt,s=s,theta1Dot=theta1Dot,theta2Dot=theta2Dot)
    
    #Convert state to Jesse's notation
    newState=(sNew[1],sNew[2],sNew[0])
    
    #Return new state and displacement in Jesse's notation
    return newState,displacement[0]

    #------------------------------physical enviroment--------------------------------#
      
      
     def execute_physical_enviroment(state,action):
            # access state action
            # sample current x
            # move servos to position
            # sample new x
            # calculate reward
            return newstate,reward

# Defining a Reward Function
In our case the reward function will be defined as the final - inital x coordinated sampled during a transition in the enviroment function

# Creating a Model in Keras
Since the DQN algorithm initializes both a Target Network (target action-value function) and a Q Network (action-value function)
we will create a function that accepts the parameters state = (n,n,.....,n) which defines the input dimension of the networks,
and the list of actions in which we defined earlier.  

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
        
                Target_network = model    
                Q_network = Target_network
 
                return Target_network, Q_network

# Selecting actions 
The nerual net does not select actions directly, instead the network will generate an array or values representing its current estimate of the action space based on the input state and the weights of the network. So to map the network outputs to executable actions we will use the the following choose action function. choose action will except the Qnetwork, the example state, and a list of the executable actions.

        def choose_action(network, state, actions, epsilion = 1):
    
                state = np.asarray(state).reshape(1, len(state))
    
                Q = network.predict(state)
             
                if random.uniform(0, 1) > epsilion:
                        print('Argmax Action')
                        Q_index = np.argmax(Q)
                        Q_value = Q[0][Q_index]
                        action = actions[Q_index]
             
                else:
                        #print('Random Action')
                        Q_index = random.randint(0,len(Q))
                        Q_value = Q[0][Q_index]
                        action = actions[Q_index]
                        
               return(action,Q_index)
           
# Executing actions 
To execute action we will draw on our enviroment functions created eariler. The function will execept the agents current state along with the selected action. The function will return the newstate of the robot and the reward encured over the duration of the transition. 
             
    def execute_action(state, action): # simualate or act real enviroment here 
    
        # Add control code here 
        
        return newstate, reward
    

# Network updates 
We consider tasks in which the agent interacts with an environment through a sequence of observations, actions and rewards. The goal of the agent is to select actions in a fashion that maximizes cumulative future reward. The underlying objective of DQN is to use the Qnetwork to approximate the optimal action-value function, which is the maximum sum of rewards (rt) discounted by (gamma) at each timestep (t), achievable by a behaviour policy pi = P(a|s) , after making an observation (s) and taking an action (a). 

Similar to Qlearning, the Qnetwork which I will call the "online network" is going to have its weights theta 
updated after each experience. We update the current estimate Q(s,a) of our online network using the function below

    def gradient_step(Q_network, Target_network, experience, learning_rate, discount, actions):
    
    # unpack experience based on transition
    state, action, newstate, reward = experience
    state, newstate = reshape_states(state,newstate)
    
    # get current estimate for state s, "contains all Q(s,A) outputs", based on Qnetwork weights theta
    current_Qs = Q_network.predict(state)
    
    # get Q(s,a) - Qnetworks current estimate of the state-action pair in our experience tuple
    current_Q = current_Qs[0][actions.index(action)] 
    
    # calculate the target Q value as the reward plus the discounted max estimate for Q(s',a') given by network weights theta-
    target_Q = reward + discount * max(Target_network.predict(newstate)[0])
    
    # compute loss
    loss = (target_Q - current_Q)**2 
    
    # replace current estimate Q(s,a) with the target value, only update weights for single state-action 
    current_Qs[0][actions.index(action)] = target_Q 
    
    # preform gradient decent step, fit weights theta to our new Q(s,a) estimate
    Q_network.train_on_batch(state, current_Qs) # or use --> Q_network.fit(state, current_Qs)
    
    return(Q_network, loss)
    
    
# Network updates contuinued...
I'd like to point out one import differitiation between Qlearning and DQN updates. 

Qlearning update -  the Q(s,a) estimate after an update will be exactly equal to the equation below

Q(s,a,new) = Q(s,a,old) + learning_rate * (r + discount * maxQ(s',a',old) - Q(s,a).

DQN update -  Q(s,a) estimate after an update WILL NOT BE EXACTLY WHAT WE CALCULATE in the equation below

Q(s,a,theta) =  reward + discount * max(Qhat(s',a',theta-)

For example let Q(s,a,theta-) represent the networks estimate for a state-action pair before 
an update, Q(s,a,theta) represent the calculated estimate for the state action pair,
and Q(s,a,theta+) represent the new network estimate after fitting Q(s,a,theta-) to Q(s,a,theta). 

So when choosing a learning rate, keep the following in mind

Q(s,a,theta+) ~ Q(s,a,theta-)--> fitted --> Q(s,a,theta) based on some learning rate lr

Q(s,a,theta+) << Q(s,a,theta) ; lr << .1

Q(s,a,theta+) ~ 1/2 Q(s,a,theta) ; lr ~ .8

Q(s,a,theta+) ~ Q(s,a,theta) ; lr ~ 1.6

Q(s,a,theta+) ~ 2 Q(s,a,theta) ; lr ~ 3.2

# Loss Function 
Based on the network update at iteration i used the following loss function
![alt text](https://adeshpande3.github.io/assets/IRL13.png)

where r + lr * maxQ(s',a',theta-) is the target value, or the reward plus the discounted estimate from our targetnetwork 
and Q(s,a,theta) is the estimate from our current network.

# Experience Replay 
Once we have accumilated enough experiences (s,a,s',r) in our replay buffer D, Experience relay consists of sampling a minibatch of experience from the replay buffer D at each step and using them to train the current Q network. This process works just like the network update discussed above ( see gradient_step() functon) expecpt instead of updating the network to a single experience, we will update on a batch of experiences.  

    def experience_replay(Q_network, Target_network, D, learning_rate, discountfactor, batch_size, actions):
    
    minibatch = random.sample(D,batch_size)
    
    for i in range(len(minibatch)): # loop through all experiences currently in the memory buffer D
        
        state,action,newstate,reward = minibatch[i] # unpack s,a,s',r from expiernces 
        state,newstate = reshape_states(state,newstate) # pre-format s and s' to vector so they can be passed to network 
       
        # get current estimate for state s, "contains all Q(s,A) outputs", based on Qnetwork weights theta
        current_Qs = Q_network.predict(state)
    
        # get Q(s,a) - Qnetworks current estimate of the state-action pair in our experience tuple
        current_Q = current_Qs[0][actions.index(action)] 
    
        # calculate the target Q value as the reward plus the discounted max estimate for Q(s',a') given by network weights theta-
        target_Q = reward + discount * max(Target_network.predict(newstate)[0])
    
         # compute loss
        loss = (target_Q - current_Q)**2 
    
        # replace current estimate Q(s,a) with the target value, only update weights for single state-action 
        current_Qs[0][actions.index(action)] = target_Q 
    
        # preform gradient decent step, fit weights theta to our new Q(s,a) estimate
        Q_network.train_on_batch(state, current_Qs) # or use --> Q_network.fit(state, current_Qs)
        
    return(Q_network, Loss_total, Loss_mean, Loss_std)
    

# Proposed revisions to DQN algorithm
- Use SARSA for on-policy updates 

Another way we can estimate the optimal action-value function, is by subsututing our off-policy target value:

target_Q = reward + discount * maxQ(s',a') 

with an on-policy update given by the SARSA algorithm given below. 

SARSA_Q = Q(s,a) + alpha ( r + gamma * maxQhat(s',a') - Q(s,a) ) , where maxQhat(s',a') is generated by the targetnetwork

In theory this should help the network stabilize and learn more efficent policies
for the sequenced locomotion problem at hand. 


# Setup and Run Test





# References and Resources
- https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
- https://www.coursera.org/lecture/machine-learning-duke/connecting-deep-q-learning-with-conventional-q-learning-oB6hj
- https://en.wikipedia.org/wiki/Q-learning
- https://www.tensorflow.org/guide/keras/train_and_evaluate
- https://www.tensorflow.org/tutorials/keras/save_and_load



