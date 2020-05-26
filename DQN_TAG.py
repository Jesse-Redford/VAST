
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, SGD, Adadelta
import itertools
import numpy as np

from keras.models import load_model   
import turtle
import random
from turtle import Turtle, Screen



def generate_network(actions,player):
    # Define Example of State input the NN can expect, and a list of actions 
    S = (0,0,0,0,0,0) # (x1,y1,phi1,x2,y2,game_condition)
    #A = list(itertools.product(range(-1,2,1),range(-1,2,1)))
    
    if player == 1:
        # Build NN Model
        layer1_neurons = 60
        layer2_neurons = 30
        layer3_neurons = 30
        model = Sequential()
        model.add(Dense(layer1_neurons, input_dim = len(S), activation='relu',kernel_initializer='random_normal'))
        model.add(Dense(layer2_neurons, activation='relu',kernel_initializer='random_normal'))
       
        model.add(Dense(output_dim = len(actions), activation='linear',kernel_initializer='random_normal'))
        #model.compile(loss='mse', optimizer=RMSprop(lr=.0001))
        model.compile(loss='mse', optimizer=Adadelta(lr=0.01))
    if player == 2:
        # Build NN Model
        layer1_neurons = 30
        layer2_neurons = 15
   
        model = Sequential()
        model.add(Dense(layer1_neurons, input_dim = len(S), activation='relu',kernel_initializer='random_normal'))
        model.add(Dense(layer2_neurons, activation='relu',kernel_initializer='random_normal'))
       
        model.add(Dense(output_dim = len(actions), activation='linear',kernel_initializer='random_normal'))
        #model.compile(loss='mse', optimizer=RMSprop(lr=.0001))
        model.compile(loss='mse', optimizer=Adadelta(lr=0.01))
        
    return model

def experience_replay(Q_network, Target_network, D, learning_rate, discountfactor, batch_size, actions):
    minibatch = random.sample(D,batch_size)
    for i in range(len(minibatch)): # loop through all experiences currently in the memory buffer D
        
        state,action,newstate,reward = minibatch[i] # unpack s,a,s',r from expiernces 
        state = (np.asarray(state).reshape(1, len(state)))
        newstate = (np.asarray(newstate).reshape(1, len(newstate)))
        
      #  state,newstate = reshape_states(state,newstate,normalize = False) # pre-format s and s' to vector so they can be passed to network 
        
        current_Qs = Q_network.predict(state) # get current estimate for state s, "contains all Q(s,A) outputs"
        action_index = actions.index(action) # determine the index of Q(s,a) in current_Qs list 
        current_Q = current_Qs[0][action_index] # get current estimate of Q(s,a)
        Q = current_Q
        
        target_Qs = Target_network.predict(newstate) # get target estimates for newstate s',"contains all Qhat(s',A) outputs"
        target_index = np.argmax(Target_network.predict(newstate)) # determine the index of maxQhat(s',a') 
        target_Q = target_Qs[0][target_index] # set target_Q equal to maxQhat(s',a')  <-- target Q_function
        
        # compute value of new Q(s,a) using bellman equation, this is the Q estimate we will fit the network to
        new_Q = current_Q + learning_rate * (reward + discountfactor * target_Q - current_Q) # reward #(reward + discountfactor * target_Q) # current_Q + learning_rate * (reward + discountfactor * target_Q - current_Q)  #
        #new_Q = reward + discountfactor * max(Target_network.predict(newstate)[0])
        yi = new_Q #(reward + discountfactor * target_Q)
        #print('yi',yi)
        
        # set current networks estimate of Q(s,a) to the updated estimate new_Q calculated above 
        current_Qs[0][action_index] = new_Q 
        
        # create an copy of the current network output Q(s,A) which includes the new value estimate for Q(s,a) computed above
        updated_Qs = current_Qs # since only 1 value differs from the current network output for Q(s,A)
                                # when we preform gradient decent with respect to weights theta
                                # we only fit the weights to a single estimate Q(s,a) instead of Q(s,A), where A is all actions
        
        # preform graident decent "fit" the Q_network weight theta such that Q(s,a) = update_Q 
        Q_network.train_on_batch(state, updated_Qs) #orginial 
        #Q_network.fit(state, updated_Qs, epochs=1)
    return(Q_network)

    
def gradient_step(Q_network, Target_network, experience, learning_rate, discountfactor, actions, action):
    state, action, newstate, reward = experience
    state = (np.asarray(state).reshape(1, len(state)))
    newstate = (np.asarray(newstate).reshape(1, len(newstate)))
    
    current_Qs = Q_network.predict(state) # get current estimate for state s, "contains all Q(s,A) outputs"
    action_index = actions.index(action) # determine the index of Q(s,a) in current_Qs list 
    current_Q = current_Qs[0][action_index] # get current estimate of Q(s,a)
    Q = current_Q
        
    target_Qs = Target_network.predict(newstate) # get target estimates for newstate s',"contains all Qhat(s',A) outputs"
    target_index = np.argmax(Target_network.predict(newstate)) # determine the index of maxQhat(s',a') 
    target_Q = target_Qs[0][target_index] # set target_Q equal to maxQhat(s',a')  <-- target Q_function
        
    # compute value of new Q(s,a) using bellman equation, this is the Q estimate we will fit the network to
    new_Q = reward #current_Q + learning_rate * (reward + discountfactor * target_Q - current_Q)  #
       
    # set current networks estimate of Q(s,a) to the updated estimate new_Q calculated above 
    current_Qs[0][action_index] = new_Q 
        
    # create an copy of the current network output Q(s,A) which includes the new value estimate for Q(s,a) computed above
    updated_Qs = current_Qs # since only 1 value differs from the current network output for Q(s,A)
                                # when we preform gradient decent with respect to weights theta
                                # we only fit the weights to a single estimate Q(s,a) instead of Q(s,A), where A is all actions
        
    # preform graident decent "fit" the Q_network weight theta such that Q(s,a) = update_Q 
    #Q_network.train_on_batch(state, updated_Qs) #orginial 
    Q_network.fit(state, updated_Qs, epochs=2)
    return(Q_network)



def get_action(network, state, actions, epsilion ):
    #norm_state = state[0]/180, state[1]/180, state[2]
    #mod_state = np.asarray(norm_state).reshape(1, len(norm_state))
    #state = reshape_state(state, normalize = False)
    state = (np.asarray(state).reshape(1, len(state)))
    Q = network.predict(state)
    
    #state = np.asarray(state).reshape(1, len(state))
    #Q = network.predict(state)
    #print('Qs',Q)
    if random.uniform(0, 1) > epsilion:
        print('Argmax Action')
        Q_index = np.argmax(Q)
        Q_value = Q[0][Q_index]
        action = actions[Q_index]
        print('Action', action, 'Q_value', Q_value)
    else:
        print('Random Action')
        Q_index = random.randint(0,len(Q[0])-1)
        #print('random selected', Q_index)
        Q_value = Q[0][Q_index]
        action = actions[Q_index]
        print('Action', action, 'Q_value', Q_value)
    return(action) 


def Beta(start = 1,end = .1,maxsteps = 100):
    return math.exp(math.log(end/start)/maxsteps)
    
    
# Set up the screen to work on
screen = Screen()
screen.setup(500, 500)

player_one = turtle.Turtle()
player_one.color("blue") 
player_one.shape("turtle")
player_one.penup()
player_one.speed('slow')


player_two = turtle.Turtle()
player_two.color("red")
player_two.shape("turtle")
player_two.penup()
player_two.speed('slow')

headings = range(0,271,30)
speeds = range(20,31,10)
actions = list(itertools.product(headings,speeds))
print(actions)
player_one_network =load_model('player_one.h5') # generate_network(actions,1)
player_two_network =  load_model('player_two.h5') #generate_network(actions,2)
player_one_target_network = generate_network(actions,1)
player_two_target_network = generate_network(actions,2)


#player_one.setpos(random.randint(-250, 250),random.randint(-250, 250))
#player_two.setpos(random.randint(-250, 250),random.randint(-250, 250))

# -1 = it
# 1 = not it

j = 0
P1_D = []
P2_D = []

eps_i = 3
eps_f = 0.1
eps = eps_i
#player_one_state = (-1,player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
#player_two_state = (1,player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
player_one.setpos(random.randint(-200, 200),random.randint(-200, 200))
player_two.setpos(random.randint(-200, 200),random.randint(-200,200))
 
import math
episodes = 100
iterations = 100
ep = 1

R_player_one = []
R_player_two = []
for e in range(episodes):
    
    if e % 2 == 0:
        player_one_state = (-1,player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
        player_two_state = (1,player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
    else:
        player_one_state = (1,player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
        player_two_state = (-1,player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
   # eps = eps * Beta(eps_i,eps_f,iterations*episodes)
    
    #player_one.setpos(0,0)
    #player_two.setpos(0,0)
    ep = ep+1
        

    
    for i in range(iterations):
        eps =  math.cos(i) # eps * Beta(eps_i,eps_f,iterations*episodes)
        eps2 =  math.cos(i+1)
        print('iteration:', ep*i , 'of' ,iterations*episodes)
        
        #eps = eps * Beta(eps_i,eps_f,iterations*episodes)
        
        print('eps',eps)
        j= j+1
    
    
        """ Alternate turns between player one and player two """
        #if i % 2 == 0:
        player_one_action = get_action(player_one_network,player_one_state,actions,eps)
        player_one.setheading(player_one_action[0])
        player_one.forward(player_one_action[1])
        
       
        player_two_action = get_action(player_two_network,player_two_state,actions,eps2)
        player_two.setheading(player_two_action[0])
        player_two.forward(player_two_action[1])
        
        print('player one action',player_one_action,'player_two_action',player_two_action)
    
    
        """ Check if one player taged the other, and update behavior to 0  or 1, return state,action,newstate,reward """
        
        print(player_one.xcor(), player_two.xcor())
        print(player_one.ycor(), player_two.ycor())
        X_condition = False
        Y_condition = False
        
        if  player_two.xcor()-20 <= player_one.xcor() <= player_two.xcor()+20:
            X_condition = True #math.isclose(player_one.xcor(), player_two.xcor(), rel_tol=0.5, abs_tol=0.0)
        if  player_two.ycor()-20 <= player_one.ycor() <= player_two.ycor()+20:
            Y_condition = True #math.isclose(player_one.ycor(), player_two.ycor(), rel_tol=0.5, abs_tol=0.0)
            
        
            
        
        
        player_one_behavior = player_one_state[0]
        player_two_behavior = player_two_state[0]
        
        if player_one_state[0] == -1:
              turtle.title ('player one (blue) is IT and player two (red) is NOT IT' + '   ' +'p1:'+str(round(sum(R_player_one),1)) + 'p2:'+str(round(sum(R_player_two),1) ))
        if player_one_state[0] == 1:
            turtle.title ('player one (blue) is NOT IT and player two (red) is IT'  + '   ' + 'p1:'+str(round(sum(R_player_one),1))+ 'p2:'+str(round(sum(R_player_two),1)))
        
   
   
        player_one_new_state = (player_one_behavior,player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
        player_two_new_state = (player_two_behavior,player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
        player_one_reward = 0
        player_two_reward = 0
            
        if player_one_state[0] == -1 and X_condition == True and Y_condition == True: 
            player_one_new_state = (1,player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
            player_two_new_state = (-1,player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
            player_one_reward = 1
            player_two_reward = -2
      
        
        if player_two_state[0] == -1 and X_condition == True and Y_condition == True:
        #elif player_one_state[0] == 1 and player_one.xcor() == player_two.xcor() and player_one.ycor() == player_two.ycor():
            player_one_new_state = (-1,player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
            player_two_new_state = (1,player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
            player_two_reward =1
            player_one_reward = -2
            
        if player_one_state[0] == -1:
            player_one_reward = -1
        
        if player_two_state[0] == -1:
            player_two_reward = -1
            
        if player_one_state[0] == 1:
            player_one_reward = 1
        
        if player_two_state[0] == 1:
            player_two_reward = 1
            
        if player_one.xcor() < -100 or  player_one.xcor() > 100 or player_one.ycor() < -100 or player_one.ycor() > 100:
            player_one.setpos(random.randint(-200, 200),random.randint(-200, 200))
            player_one_reward = -10
            player_one_new_state = (player_one_state[0],player_one.xcor(), player_one.ycor() , player_one.heading()  , player_two.xcor(), player_two.ycor() )
           
            
        if player_two.xcor() < -100 or  player_two.xcor() > 100 or player_two.ycor() < -100 or player_two.ycor() > 100:
            player_two.setpos(random.randint(-200, 200),random.randint(-200,200))
            player_two_reward = -10
            player_two_new_state = (player_two_state[0],player_two.xcor(), player_two.ycor() , player_two.heading()  , player_one.xcor(), player_one.ycor() )
    
      
       
            
        R_player_one.append(player_one_reward)
        R_player_two.append(player_two_reward)
        
       
        
        print('player_one_state',player_one_state[0])
        print('player_two_state',player_two_state[0])
        
        print('player_one_reward',player_one_reward)
        print('player_two_reward',player_two_reward)
        
        player_one_experience = [player_one_state,player_one_action,player_one_new_state,player_one_reward]
        player_two_experience = [player_two_state,player_two_action,player_two_new_state,player_two_reward]
        
        """ Store Experience in replay buffers """
        P1_D.append([player_one_state,player_one_action,player_one_new_state,player_one_reward])
        P2_D.append([player_two_state,player_two_action,player_two_new_state,player_two_reward])
        
       # P2_D.append([player_one_state,player_one_action,player_one_new_state,player_one_reward])
       # P1_D.append([player_two_state,player_two_action,player_two_new_state,player_two_reward])
        
        
        player_one_Q_network = gradient_step(player_one_network, player_one_target_network, player_one_experience, .02, .9, actions, player_one_action)
        player_two_Q_network = gradient_step(player_two_network, player_two_target_network, player_two_experience, .02, .9, actions, player_two_action)
    
        player_one_state = player_one_new_state
        player_two_state = player_two_new_state
    
        """ preform experience replay """
        if e > 1:
            player_one_Q_network = experience_replay(player_one_network, player_one_target_network, P1_D, learning_rate = 0.5, discountfactor=.9, batch_size = 64, actions = actions)
            player_two_Q_network = experience_replay(player_two_network, player_two_target_network, P2_D, learning_rate = 0.5, discountfactor=.9, batch_size = 64, actions = actions)
    
        if j == 100:
            player_one_target_network = player_one_network 
            player_two_target_network = player_two_network 
            j = 0
       
    
        
player_one_network.save('player_one.h5')
player_two_network.save('player_two.h5')    
    
            