# -*- coding: utf-8 -*-
"""
@author: Dominik
"""
from pypot.creatures import PoppyHumanoid
import pypot
import numpy as np

# own modules
import goals
#import q_networks
import poppy_qnetwork as q_network

#for simluation of goal
import vrep_draw


ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1
ANGULAR_ARM_VELOCITY = 1
GOAL_THRESHOLD = 0.5
MAX_STEPS = 500
NUM_OF_ACTIONS = 4
NUM_OF_STATES = 4
SIDE = 'l' # 'r'

JOINT_LIMITS = ((-20., 195.), (-148., 1.)) # LEFT ARM
#JOINT_LIMITS = ((-195., 20.), (-1., 148.)) # RIGHT ARM 


def set_action(poppy, action):
    waiting=False
    if action==0:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position - ANGULAR_ARM_VELOCITY, 0, wait=waiting)
    elif action==1:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position + ANGULAR_ARM_VELOCITY, 0, wait=waiting) 
    elif action==2:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position - ANGULAR_ARM_VELOCITY, 0, wait=waiting)
    elif action==3:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position + ANGULAR_ARM_VELOCITY, 0, wait=waiting)
    return

def get_poppy_angles(poppy):
    theta = [0.,0.]
    theta[0] = float(poppy.l_shoulder_x.present_position)
    theta[1] = float(poppy.l_elbow_y.present_position)
    return np.array(theta)

def get_poppy_normed_angles(poppy):
    # normalize thetas to [-1.0, +1.0]
    theta = get_poppy_angles(poppy)

    range1_half = 0.5*(JOINT_LIMITS[0][1] - JOINT_LIMITS[0][0])
    theta1 = (theta[0] - range1_half)/range1_half

    range2_half = 0.5*(JOINT_LIMITS[1][1] - JOINT_LIMITS[1][0])
    theta2 = (theta[1] - range2_half)/range2_half        
    return np.array([theta1, theta2])

def get_poppy_position(poppy):
    # relative position of hand with respect to shoulder
    #x,y = np.array(poppy.get_object_position(SIDE+'_forearm_visual')[0:2]) - np.array(poppy.get_object_position(SIDE+'_shoulder_x')[0:2])
    #return np.array([-y,x])*100.0

    # Forward Kinematics
    theta = get_poppy_angles(poppy)*np.pi/180.
    pos = np.array([0.0, 0.0])
    pos[0] = ARM_LENGTH_1*np.cos(theta[0]) + ARM_LENGTH_2*np.cos(theta[0]+theta[1])
    pos[1] = ARM_LENGTH_1*np.sin(theta[0]) + ARM_LENGTH_2*np.sin(theta[0]+theta[1])
    return pos

def get_4_state(poppy, goal):
    # normalized [dx, dy]
    dist = (goal.get_position() - get_poppy_position(poppy)) / (ARM_LENGTH_1+ARM_LENGTH_2)

    # normalized [theta_1, theta_2]
    theta = get_poppy_normed_angles(poppy)

    # state = [dx,dy,theta1,theta2]
    return np.array([dist, theta])

def episode_finished(poppy, goal):
    agent_pos = get_poppy_position(poppy)
    goal_pos = goal.get_position()
    distance = np.linalg.norm(agent_pos - goal_pos)
    if distance < GOAL_THRESHOLD:
        return True # episode finished if agent already at goal
    else:
        return False

if __name__ == "__main__":
    # load qnetwork
    #networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)
    network = q_network.QNetwork()

    # init poppy
    poppy = PoppyHumanoid(simulator='vrep')
    poppy.reset_simulation()
    
    #init goal plot
    plt = vrep_draw.Plot(poppy)

    # go to init position
    poppy.l_shoulder_y.goto_position(-90,1,wait=True)
    poppy.l_arm_z.goto_position(-90,1,wait=True)

    # some starting position
    poppy.l_elbow_y.goto_position(-90,1,wait=True)
    poppy.l_shoulder_x.goto_position(60, 1, wait=True)

    while True:
        # set a goal
        goal = goals.Goal_Arm(ARM_LENGTH_1, ARM_LENGTH_2)
        goal.pos = np.array([25., 0.]) 
        
        #goal change in vrep      
        v_pos = (goal.get_position()/100.).tolist()+[1.0]
        v_x = goal.get_position()[1]/100
        v_y = -goal.get_position()[0]/100
        v_z = poppy.get_object_position('l_shoulder_visual')[2] - 0.1
        v_goal_position = [v_x, v_y, v_z]
        plt.update_object_pos(v_goal_position)
        #print poppy.get_object_position('goal'), v_goal_position, goal.get_position()

        # control arm
        for step in range(MAX_STEPS):

            plt.update_object_pos(v_goal_position)
            #x,y = poppy.get_object_position('goal')[:2]
            #print (-y*100.,x*100.), goal.get_position()

            print "s: %3d; goal = [%s]; state [%s]" % (step, goal.get_position(), get_4_state(poppy, goal))

            # stop early when goal is reached
            if not episode_finished:
                break 

            # state
            state = get_4_state(poppy, goal)

            # prediction
            #q = networks.online_net.predict(state.reshape(1, NUM_OF_STATES), batch_size=1)
            q = network.predict(state.reshape(1, NUM_OF_STATES).tolist()[0])
            action = np.argmax(q) # choose best action from Q(s,a)
            
            # action
            set_action(poppy, action)

    #poppy.stop_simulation()
    poppy.close()
    #pypot.vrep.close_all_connections()