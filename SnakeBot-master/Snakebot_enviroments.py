###########################################################################
#--------------- Snakebot_enviroments ------------------------------------#
###########################################################################

# - Author: Jesse Redford
# - Date: 2/7/2020

import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
import import_ipynb # pip install import_ipynb

###########################################################################################
#--------------- Set up example Snakebot_enviroments  ------------------------------------#
###########################################################################################

#import Snakebot_enviroments as SBE
#from DQN_Snakebot_Agent import execute_action

# add parameters and updates cross reference  SnakeBot_DQN_runner 
#newstate,reward = execute_action(state,action, simulate = True) # set to false for physical
####################################################################################################
#--------------- Set up functions for simulated and physical enviroments --------------------------#
####################################################################################################

def reset():
    # reset camera and phi
    # reset joint angles, theta1, theta2 to 90
    state = (.1,.2,.1)
    return state

#def execute_action(state, action): # simualate or act real enviroment here 
    # add transition sequence for physical bot
  #  newstate = (0,0,0)
 #   reward = 1
   # return newstate, reward

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

#def advanceState(dt,s,theta1Dot,theta2Dot,rho=1,n=101,sys=threeWheel):
def advanceState(dt,s,theta1Dot,theta2Dot,rho=1,n=50,sys=threeWheel):
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

#######################################################################################
#------------------------------Execute Selected Action--------------------------------#
#######################################################################################


def execute_action(state,action):
    
    ###Convert Jesse's notation to Will's notation
    s=[state[2],state[0],state[1]]
    theta1Dot=action[0]
    theta2Dot=action[1]
    dt=action[2]
    
    #Feed variables to solver, get displacement and new state in Will's notation
    displacement, sNew = advanceState(dt=dt,s=s,theta1Dot=theta1Dot,theta2Dot=theta2Dot)
    
    #Convert state to Jesse's notation
    newState=(sNew[1],sNew[2],sNew[0])
    
    
    if newState[0] > np.pi or newState[0]  < - np.pi:
        newState =(.1,-.1,.1)
        displacement[0] = 0
        
    elif newState[1] > np.pi or newState[1]  < - np.pi:
       newState=(.1,-.1,.1)
       displacement[0] = 0
    else:
        pass

    
    return newState,displacement[0]



#######################################################################################
#------------------------------Physical Enviroment--------------------------------#
#######################################################################################












