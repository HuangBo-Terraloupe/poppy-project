# -*- coding: utf-8 -*-
"""
@author: Dominik
"""
from pypot.creatures import PoppyHumanoid
import pypot
import numpy as np

# own modules
import goals
import q_networks


ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1
ANGULAR_ARM_VELOCITY = 1
GOAL_THRESHOLD = 0.5
MAX_STEPS = 500
NUM_OF_ACTIONS = 4
NUM_OF_STATES = 4


def set_action(poppy, action):
    # LEFT ARM
    if action==1:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position - ANGULAR_ARM_VELOCITY, 0, wait=True)
    elif action==2:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position + ANGULAR_ARM_VELOCITY, 0, wait=True) 
    elif action==3:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position - ANGULAR_ARM_VELOCITY, 0, wait=True)
    elif action==4:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position + ANGULAR_ARM_VELOCITY, 0, wait=True)
    
    # RIGHT ARM
    elif action==5:
        poppy.r_shoulder_x.goto_position(poppy.r_shoulder_x.present_position + ANGULAR_ARM_VELOCITY, 0, wait=True)
    elif action==6:
        poppy.r_shoulder_x.goto_position(poppy.r_shoulder_x.present_position - ANGULAR_ARM_VELOCITY, 0, wait=True) 
    elif action==7:
        poppy.r_elbow_y.goto_position(poppy.r_elbow_y.present_position + ANGULAR_ARM_VELOCITY, 0, wait=True)
    elif action==8:
        poppy.r_elbow_y.goto_position(poppy.r_elbow_y.present_position - ANGULAR_ARM_VELOCITY, 0, wait=True)
    return

def get_poppy_angles(poppy):
    # theta_1, theta_2
    # [theta_1, theta_2]
    # TODO: test like that, test with "int(angle)*np.pi/180.0"
    lt1 = poppy.l_shoulder_x.present_position*np.pi/180.0
    lt2 = poppy.l_elbow_y.present_position*np.pi/180.0
    rt1 = poppy.r_shoulder_x.present_position*np.pi/180.0
    rt2 = poppy.r_elbow_y.present_position*np.pi/180.0
    return np.array([lt1, lt2, rt1, rt2])

def get_poppy_position(poppy):
    # [x,y] arm
    return [poppy.get_object_position('l_forearm_visual')[0:2], poppy.get_object_position('r_forearm_visual')[0:2]]

def get_4_state(poppy, goal):
    # normalized [dx, dy]
    dist1 = (goal.get_position() - get_poppy_position(poppy)[0:2]) / (ARM_LENGTH_1+ARM_LENGTH_2)
    dist2 = (goal.get_position() - get_poppy_position(poppy)[2:4]) / (ARM_LENGTH_1+ARM_LENGTH_2)

    # normalized [theta_1, theta_2]
    theta = get_poppy_angles(poppy) / np.pi

    # state = [dx,dy,theta1,theta2]
    return np.array([dist, theta])

def episode_finished(poppy, goal):
    agent_pos = get_poppy_position(poppy)[0:2]
    goal_pos = self.goal.get_position()
    right_distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])

    agent_pos = get_poppy_position(poppy)[2:4]
    goal_pos = self.goal.get_position()
    left_distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])

    if (right_distance < GOAL_THRESHOLD) or (left_distance < GOAL_THRESHOLD):
        return True # episode finished if agent already at goal
    else:
        return False

if __name__ == "__main__":
    # load qnetwork
    networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)

    # init poppy
    poppy = PoppyHumanoid(simulator='vrep')
    poppy.reset_simulation()

    # go to init position
    poppy.l_shoulder_y.goto_position(-90,1,wait=True)
    poppy.l_arm_z.goto_position(-90,1,wait=True)
    poppy.r_shoulder_y.goto_position(-90,1,wait=True)
    poppy.r_arm_z.goto_position(90,1,wait=True)

    while True:
        # set a goal
        goal = goals.Goal_Arm(ARM_LENGTH_1, ARM_LENGTH_2)

        # control arm
        for step in range(MAX_STEPS):

            print "s: %3d; goal = [%s]" % (step, goal.get_position())

            # stop early when goal is reached
            if not episode_finished:
                break 

            # state
            state = get_4_state(poppy, goal)

            # prediction
            q = networks.online_net.predict(state.reshape(1, NUM_OF_STATES), batch_size=1)
            action = np.argmax(q) # choose best action from Q(s,a)
            
            # action
            set_action(poppy, action)

    #poppy.stop_simulation()
    poppy.close()
    #pypot.vrep.close_all_connections()