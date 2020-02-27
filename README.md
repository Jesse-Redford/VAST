# VAST - Valid Action State Transition Optimization

The idea of VAST is to provide all the resources you need to ML and RL running on your robotic projects. Each master folder contains a robot prototype. These folder contain setup intrucstions, wiring details, code examples, results and more.


 
# Snakebot - Master

- add folder with all Qlearning and DQN files and documentation once verified and completed, highlight plots, images and video results here

# Fishbot - Master

- add folder with all Qlearning and DQN files and documentation once verified and completed, highlight plots, images and video results here

# Intellegent Locomotion Controller - Master

- add folder containing all files and documentation for developing an Intellegent Locomotion Controller which uses 
- Qlearning to devleop low level control polcies which are then saved and used by a highlevel DQN network as actions
- in order to naviage to difference global coordinates highlight plots, images and video results here

# Setup for Intellegent Controller

Load Qpolicies 
QX = (Qfx, states, phi_states)
QY = (Qfy, states, phi_states)

Create Action Space
action1 = QX
actions2 = QY
actions =  list_actions(QX,QY)

Define Target Coordinates for Training
xtarget = 20
ytarget = 20

# Control Algorithm

Initalize QNetwork
Q_network = NN.generate_networks(example_state, actions, learning_rate, model_architecture)


for e in range(episodes):
  x,y,phi,theta1,theta2 = robot.reset()
  dqn_state = (x,y,phi,xtarget,ytarget)
  
  for i in range(iterations):
     action, Q_index = NN.choose_action(Q_network, dqn_state, actions, epsilion)
     
     # Execute action --> rollout policy
     Qfx, states, phi_states = action
     phi_disc = find_nearest(phi_states,phi)
     low_level_state = (theta1,theta2,phi_disc)
     
     systemstate,pipe = PolicyRollout(Qtable, low_level_state, states, phi_states, pipe, policy_steps = 10)
     
     # observe newstate returned from enviroment
     xf,yf,phif,theta1,theat2 = systemstate
     dqn_newstate = (xf,yf,phif,xtarget,ytarget)

     # Check if robot has reached target coordinates, if yes reward =100 and session restarts 
     if xf == xtarget and yf == ytarget:
         reward = 100
         break 
         
     else:  # calculate change in disance from target as reward and add expereince
         reward = np.sqrt( (xtarget-x) + (ytarget -y) ) - np.sqrt( (xtarget-xf) + (ytarget -yf) )
         experience = (dqn_state, action, dqn_newstate, reward)
         D.append(experience)
     
     # preform experiene replay and gradient decent step
     Q_network, batch_loss, batch_mean, batch_std = NN.experience_replay(Q_network, D, learning_rate, batch_size,actions) 
     
     # update current state
      dqn_state = dqn_newstate
     
    
    


# VAST App - Master

- add kivy documetation and files for creating a monioritng and deployment app

# ROS Developer Guide
- code details and setup for using ROS and online development studio
