# -*- coding: utf-8 -*-
"""
Created on Wed May 03 13:49:51 2017

@author: Fares
"""

from pypot.creatures import PoppyHumanoid
import pypot
import time, math
import numpy as np

poppy = PoppyHumanoid(simulator='vrep')
poppy.reset_simulation()

def get_Jacobian(theta):
    ARM_LENGTH_1 = 0.151
    ARM_LENGTH_2 = 0.101
    return np.array([
                [-np.sin(theta[0])*ARM_LENGTH_1 - ARM_LENGTH_2*np.sin(theta[0]+theta[1]), 
                -ARM_LENGTH_2*np.sin(theta[0]+theta[1])], 

                [np.cos(theta[0])*ARM_LENGTH_1 + ARM_LENGTH_2*np.cos(theta[0]+theta[1]), 
                ARM_LENGTH_2*np.cos(theta[0]+theta[1])]
            ])

def get_control(distance, theta):
    J = get_Jacobian(theta)
    u = np.dot(np.linalg.pinv(J), distance)
    return u

# left init
poppy.l_shoulder_y.goto_position(-90,1,wait=True)
poppy.l_arm_z.goto_position(-90,1,wait=True)

# right init
poppy.r_shoulder_y.goto_position(-90,1,wait=True)
poppy.r_arm_z.goto_position(90,1,wait=True)

# some starting position
poppy.l_elbow_y.goto_position(-10,1,wait=True)
#poppy.l_shoulder_x.goto_position(10, 1, wait=True)


hand = poppy.get_object_position('l_forearm_visual')
goal = np.array([0.2, -0.1])

print hand[0:2], goal

def set_action(action):
    if action==1:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position - 1, 0, wait=True)
    elif action==2:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position + 1, 0, wait=True) 
    elif action==3:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position - 1, 0, wait=True)
    elif action==4:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position + 1, 0, wait=True)
    return

while True:

    hand = poppy.get_object_position('l_forearm_visual')
    hand = np.array(hand[0:2])

    distance = goal - hand
    print np.linalg.norm(distance)

    if np.linalg.norm(distance) < 0.01:
        break
    theta = [poppy.l_shoulder_x.present_position*np.pi/180.0, poppy.l_elbow_y.present_position*np.pi/180.0]

    # TODO: normalize state
    #state = np.vstack((hand,np.array(theta),goal))
    #Q = network.predict(state) <-- look it up
    #action = np.argmax(Q) <-- look it up
    #set_action(action) <-- look it up

    u = get_control(distance, theta)

    if np.argmax(np.abs(u)) == 0:
        if u[0] < 0:
            poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position - 1, 0, wait=True)
        else:
            poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position + 1, 0, wait=True) 
    else:
        if u[1] < 0:
            poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position - 1, 0, wait=True)
        else:
            poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position + 1, 0, wait=True)

    #time.sleep(0.01)

#poppy.stop_simulation()
poppy.close()
#pypot.vrep.close_all_connections()
