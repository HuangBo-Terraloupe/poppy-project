#!/usr/bin/python
import numpy as np


# ARM PARAMETERS
ANGULAR_ARM_VELOCITY = 1.0/180.0*np.pi
ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1

JOINT_LIMITS = ((-20./180*np.pi, 195./180*np.pi), (-148./180*np.pi, 1./180*np.pi)) # LEFT ARM
#JOINT_LIMITS = ((-195./180*np.pi, 20./180*np.pi), (-1./180*np.pi, 148./180*np.pi)) # RIGHT ARM 
# TODO: extend actions to all combinations, i.e. instead of 4 actions, all 3*3=9 actions (if too much time)

class Arm:
    def __init__(self, angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2):
        self.base_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.ctrl = np.array([0.0, 0.0])

        angle1 = np.random.uniform(JOINT_LIMITS[0][0], JOINT_LIMITS[0][1])
        angle2 = np.random.uniform(JOINT_LIMITS[1][0], JOINT_LIMITS[1][1])
        self.theta = np.array([angle1,angle2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0])

        self.ANGULAR_VELOCITY_1 = angular_velocity_1
        self.ANGULAR_VELOCITY_2 = angular_velocity_2
        self.ARM_LENGTH_1 = arm_length_1
        self.ARM_LENGTH_2 = arm_length_2
        
        self.pos = self.get_end_effector_position()
        
    def get_end_effector_position(self):
        pos = np.array([0.0, 0.0])
        pos[0] = self.base_pos[0] + self.ARM_LENGTH_1*np.cos(self.theta[0]) + self.ARM_LENGTH_2*np.cos(self.theta[0]+self.theta[1])
        pos[1] = self.base_pos[1] + self.ARM_LENGTH_1*np.sin(self.theta[0]) + self.ARM_LENGTH_2*np.sin(self.theta[0]+self.theta[1])
        return pos

    def get_Jacobian(self):
        return np.array([
                    [-np.sin(self.theta[0])*self.ARM_LENGTH_1 - self.ARM_LENGTH_2*np.sin(self.theta[0]+self.theta[1]), 
                    -self.ARM_LENGTH_2*np.sin(self.theta[0]+self.theta[1])], 

                    [np.cos(self.theta[0])*self.ARM_LENGTH_1 + self.ARM_LENGTH_2*np.cos(self.theta[0]+self.theta[1]), 
                    self.ARM_LENGTH_2*np.cos(self.theta[0]+self.theta[1])]
                ])

    def get_control(self, distance):
        J = self.get_Jacobian()
        u = np.dot(np.linalg.inv(J), distance)
        return u

    def get_position(self):
        # get position without normalization
        return np.copy(self.pos)

    def get_normalized_position(self):
        # normalize position to [-1.0, +1.0]
        normalized_pos = (self.pos - self.base_pos)/(self.ARM_LENGTH_1+self.ARM_LENGTH_2)
        return np.hstack(normalized_pos)

    def get_normalized_angles(self):
        # normalize thetas to [-1.0, +1.0]
        range1_half = 0.5*(JOINT_LIMITS[0][1] - JOINT_LIMITS[0][0])
        theta1 = (self.theta[0] - range1_half)/range1_half

        range2_half = 0.5*(JOINT_LIMITS[1][1] - JOINT_LIMITS[1][0])
        theta2 = (self.theta[1] - range2_half)/range2_half        
        return np.array([theta1, theta2])

    def get_4_state(self, goal_pos):
        # state = [dx, dy, theta_1, theta_2]
        normalized_dist = (goal_pos - self.pos)/(self.ARM_LENGTH_1+self.ARM_LENGTH_2) 
        return np.hstack((normalized_dist, self.get_normalized_angles()))

    def plot(self, ax):
        # middle joint position
        pos = [ self.base_pos[0] + self.ARM_LENGTH_1*np.cos(self.theta[0]),
                self.base_pos[1] + self.ARM_LENGTH_1*np.sin(self.theta[0])]

        linewidth = 5
        markersize = 10
        vel_factor = 100

        # arm links
        h_arm1 = ax.plot(   [self.base_pos[0], pos[0]], 
                            [self.base_pos[1], pos[1]], 
                            'k', linewidth=linewidth)
        h_arm2 = ax.plot(   [pos[0], self.pos[0]], 
                            [pos[1], self.pos[1]], 
                            'k', linewidth=linewidth)

        # base, middle and end-effector joints
        h_base = ax.plot(   self.base_pos[0], self.base_pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)
        h_middle = ax.plot(  pos[0], pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)
        h_end = ax.plot(    self.pos[0], self.pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)

        # velocity indicators
        h_vel1 = ax.plot(   [self.base_pos[0], self.base_pos[0]-vel_factor*self.vel[0]*np.sin(self.vel[0])], 
                            [self.base_pos[1], self.base_pos[1]+vel_factor*self.vel[0]*np.cos(self.vel[0])], 
                            'b', linewidth=linewidth/2)

        h_vel2 = ax.plot(   [pos[0], pos[0]-vel_factor*self.vel[1]*np.sin(self.vel[1])], 
                            [pos[1], pos[1]+vel_factor*self.vel[1]*np.cos(self.vel[1])], 
                            'b', linewidth=linewidth/2)

    def set_action(self, action):
        self.ctrl = np.array([0.0, 0.0]) # reset control
        if action==0:
            self.ctrl[0] = -self.ANGULAR_VELOCITY_1
        elif action==1:
            self.ctrl[0] = self.ANGULAR_VELOCITY_1
        elif action==2:
            self.ctrl[1] = -self.ANGULAR_VELOCITY_2
        elif action==3:
            self.ctrl[1] = self.ANGULAR_VELOCITY_2
                
    def update(self):
        # update (angular) velocities
        self.vel = self.ctrl
        
        # update positions
        self.theta += self.vel

        #re-map joints to environment
        self.theta[0] = np.maximum(np.minimum(self.theta[0], JOINT_LIMITS[0][1]), JOINT_LIMITS[0][0])
        self.theta[1] = np.maximum(np.minimum(self.theta[1], JOINT_LIMITS[1][1]), JOINT_LIMITS[1][0])

        # update end-effector position
        self.pos = self.get_end_effector_position()                


#agent = Arm(arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
#fig,ax = plt.subplots(1,1)
#agent.plot(ax)
#print agent.get_state()

#agent.set_action(1)
#agent.update()

#print agent.get_state()
#agent.plot(ax)

#plt.show()