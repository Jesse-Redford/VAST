# SnakeBot_SETUP_GUIDE.md
Walk through for settting up SnakeBot from scratch


# Setting up Rasberry Pi
- download and burn copy of pre-installed snakebot OS and libraries 
- setup a remote desktop connection

# Establish Remote Access 

Team-Veiwer
- username: uncc.loco@gmail.com
- password: Kellybot

Real-VNC
- username: uncc.loco@gmail.com
- Password: Kellybot

Rasberry Pi
- password: sd123456
 
# Verify the following libraries are installed 
- keras
- pandas
- scikit learn
- pyrealsense2
- ect
- ect

# Installing Servo Hat
- For this we will need to install circuit python, see the example here. --> (website link)
                Add working test code here

# Wiring battery source
- Add diagram and schematic here

# Assembling Hardware
- Add schematics
- bill of materials
- solidworks files here


# System Validation and Test Files
- RealSense_Calibration.py # test realsense camera 
- Servo_Calibration        # test servo operation
- System_Validation        # validate system operation

# Setting up Intel Real Sense
You will need to purchase and install software on a Rasberry Pi B+3 using the following source code --> (Download Complete OS-Install) 
Next you will need to import the Intel_Real_Sense module, see program for details. 


                Add working test code & file here
                
                 
    # cd DeepRobots python3 DQN/DQN_physical_runner.py


    # Test script and functions for Intel Real Sense Camera
    import sys
    #sys.path.append('local/lib/python3.6/site-packages ')
    # Import Libraries 
    import pyrealsense2 as rs
    import math
    import time


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


     def GetLoc():
     frames = pipe.wait_for_frames()
     pose = frames.get_pose_frame()
     data = pose.get_pose_data()
     Q = [data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w]
     #rotVecs = getRotVecs(Q)
    #phi = rotVecs[2] # check domain of values this returns showing above 3.7
    phi = yaw(Q)
    #print('raw_phi',phi)
    x = round(-data.translation.z * 39.3701,2) # x inches
    y= round(-data.translation.x * 39.3701,2) # y inches
    #print('x', x, 'y', y, 'phi',phi)
    return[x,y,phi]

    rs.device.hardware_reset
    pipe=rs.pipeline()
    cfg=rs.config()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)



    # Test Loop 
    while True:
      x,y,phi = GetLoc() # Sample x,y,phi
      phi = phi 

    #phi_deg = round(abs(phi * 180/math.pi + 180),2) # postive range from 0-360deg
    x_in = round(x*39.3701,2)
    y_in = round(y*39.3701,2)
    print(x_in,y_in,phi) # x,y in inches, phi in deg (CW = (-) CCW = (+)     
    time.sleep(0.25)
    pipe.stop() ###At END OF TRIAL
   
    def sim(pipe):
     x,y,phi = GetLoc(pipe) # Sample x,y,phi
     rs.device.hardware_reset
     return(x,y,phi)

    def RunTest(steps):
    rs.device.hardware_reset
    pipe=rs.pipeline()
    cfg=rs.config()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)
    i=0
    while i < steps:
      x,y,phi = sim(pipe) # Sample x,y,phi
      print(x,y,phi)
      time.sleep(0.25)
      i = i+1
    pipe.stop() ###At END OF TRIAL
    return()
    RunTest(10)


# Tutorials and Reference Sites for Setup Troubleshooting
- adufruit
- realsense
- unbunu
- remote desktop
