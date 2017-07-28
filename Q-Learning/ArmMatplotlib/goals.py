#!/usr/bin/python
import numpy as np


MARKERSIZE = 30
LINEWIDTH = 3

JOINT_LIMITS = ((-20./180*np.pi, 195./180*np.pi), (-148./180*np.pi, 1./180*np.pi)) # LEFT ARM
#JOINT_LIMITS = ((-195./180*np.pi, 20./180*np.pi), (-1./180*np.pi, 148./180*np.pi)) # RIGHT ARM 

class Goal_Arm:
    def __init__(self, arm_length1, arm_length2):
        self.ARM_LENGTH_1 = arm_length1
        self.ARM_LENGTH_2 = arm_length2
        
        angle1 = np.random.uniform(JOINT_LIMITS[0][0], JOINT_LIMITS[0][1])
        angle2 = np.random.uniform(JOINT_LIMITS[1][0], JOINT_LIMITS[1][1])
        x = arm_length1*np.cos(angle1) + arm_length2*np.cos(angle1+angle2)
        y = arm_length1*np.sin(angle1) + arm_length2*np.sin(angle1+angle2)
        self.pos = np.array([x, y])     

    def plot(self, ax):
        ax.plot(    self.pos[0], 
                    self.pos[1], 
                    'bo', markersize=MARKERSIZE, markeredgewidth=LINEWIDTH)

    def get_position(self):
        # goal position
        return np.copy(self.pos)

    def get_normalized_position(self):
        # goal position is normalized 2D [x,y] (positions are between -1 and 1)
        normalized_pos = self.pos / (self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        return normalized_pos

    def get_state(self):
        # goal state is only goal position
        normalized_pos = self.pos / (self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        return normalized_pos