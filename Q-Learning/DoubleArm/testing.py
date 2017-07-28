#!/usr/bin/python
import numpy as np
np.set_printoptions(precision=4)
import os
import sys
import threading
import time
from collections import deque

# import own modules
import agents
import goals
import q_networks
import pyglet_draw


ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1
ARM_OFFSET = -10.95
ANGULAR_ARM_VELOCITY = 1.0*np.pi/180.0

GOAL_THRESHOLD = 0.5
HEIGHT = 70
MAX_STEPS = 500
NUM_OF_ACTIONS = 8
NUM_OF_ACTORS = 1
NUM_OF_PLOTS_X = 1
NUM_OF_PLOTS_Y = 1
NUM_OF_STATES = 8
PATH_LENGTH = 10
WIDTH = 70


class Actor(threading.Thread):
    def __init__(self, threadID, goal_threshold=GOAL_THRESHOLD, max_steps=MAX_STEPS):
        threading.Thread.__init__(self)
        self.agent = None                       # place-holder for agent
        self.left_arm = None
        self.right_arm = None
        self.goal = None 					    # placer-holder for goal
        self.GOAL_THRESHOLD = goal_threshold 	# desired distance to goal; episode is finished early if threshold is achieved
        self.MAX_STEPS = max_steps 				# maximal steps per episode
        self.THREAD_ID = threadID 				# thread id (integer)
        self.path = deque([], maxlen=PATH_LENGTH)

    def get_state(self):
        # state is composed by agent + goal states
        return np.hstack((self.left_arm.get_4_state(self.goal.get_position()), self.right_arm.get_4_state(self.goal.get_position())))

    # penalize distance to goal
        agent_pos = self.left_arm.get_normalized_position()
        goal_pos = self.goal.get_normalized_position()
        distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        left_reward = -distance 

        agent_pos = self.right_arm.get_normalized_position()
        goal_pos = self.goal.get_normalized_position()
        distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        right_reward = -distance 

        # return the reward/punishment for the closest arm
        return np.maximum(right_reward, left_reward)

    def plot(self):
        # stepwise refreshing of plot
        ax[0,self.THREAD_ID].clear()
        
        # plotting of AGENT, GOAL and set AXIS LIMITS
        self.goal.plot(ax[0,self.THREAD_ID])
        self.left_arm.plot(ax[0,self.THREAD_ID])
        self.right_arm.plot(ax[0,self.THREAD_ID])
        ax[0,self.THREAD_ID].set_xlim([-WIDTH/2, WIDTH/2])
        ax[0,self.THREAD_ID].set_ylim([-HEIGHT/2, HEIGHT/2])

        for point in self.path:
            ax[0,self.THREAD_ID].plot(point[0],point[1])

    def run(self):

        self.left_arm = agents.Arm(side='left', base_pos=[0.,-ARM_OFFSET], angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
        self.right_arm = agents.Arm(side='right', base_pos=[0.,ARM_OFFSET], angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)

        while True: 
            # init new episode
            plotting_lock.acquire()
            self.goal = goals.Goal_Arm(ARM_LENGTH_1, ARM_LENGTH_2)
            plotting_lock.release()

            for step in range(self.MAX_STEPS):
                print step 

                # produce experience
                state = self.get_state()
                self.path.append(self.agent.get_end_effector_position())

                # get lock to synchronize threads
                networks_lock.acquire()
                q = networks.online_net.predict(state.reshape(1,NUM_OF_STATES), batch_size=1)
                networks_lock.release()
                action = np.argmax(q) # choose best action from Q(s,a)

                # take action, observe next state s'
                if action < NUM_OF_ACTIONS/2:
                    self.left_arm.set_action(action)
                    self.left_arm.update()
                else:
                    action -= NUM_OF_ACTIONS/2
                    self.right_arm.set_action(action)
                    self.right_arm.update()
                next_state = self.get_state()
                
                # plot the scene
                plotting_lock.acquire()
                self.plot()
                plotting_lock.release()

                # check if agent at goal
                if self.episode_finished():
                    break # start new episode


if __name__ == "__main__":
    # create GLOBAL thread-locks
    console_lock = threading.Lock()
    networks_lock = threading.Lock()
    plotting_lock = threading.Lock()

    # create GLOBAL Q-NETWORKS
    networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)

    #initialize pyglet plotting
    plt = pyglet_draw.Plot()
    
    # initialize GLOBAL plotting
    fig, ax = plt.subplots(NUM_OF_PLOTS_X,NUM_OF_PLOTS_Y)

    # create threads
    threads = []
    threads.extend([Actor(i) for i in range(NUM_OF_ACTORS)])

    # set daemon, allowing Ctrl-C
    for i in range(len(threads)):
        threads[i].daemon = True

    # start new Threads
    [threads[i].start() for i in range(len(threads))]
 
    # from here on everything is handeled by pyglet_draw
    plt.run_loop()