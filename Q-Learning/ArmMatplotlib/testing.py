#!/usr/bin/python
import matplotlib
matplotlib.backend = 'Qt4Agg'
import matplotlib.pyplot as plt

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


ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1
ANGULAR_ARM_VELOCITY = 1.0*np.pi/180.0

GOAL_THRESHOLD = 0.5
HEIGHT = 50
MAX_STEPS = 500
NUM_OF_ACTIONS = 4
NUM_OF_ACTORS = 1
NUM_OF_STATES = 4
PATH_LENGTH = 500
WIDTH = 50


class Actor(threading.Thread):
    def __init__(self, threadID, goal_threshold=GOAL_THRESHOLD, max_steps=MAX_STEPS):
        threading.Thread.__init__(self)
        self.agent = None                       # place-holder for agent
        self.goal = None 					    # placer-holder for goal
        self.GOAL_THRESHOLD = goal_threshold 	# desired distance to goal; episode is finished early if threshold is achieved
        self.MAX_STEPS = max_steps 				# maximal steps per episode
        self.THREAD_ID = threadID 				# thread id (integer)
        self.path = deque([], maxlen=PATH_LENGTH)

    def get_state(self):
        # state is composed by agent + goal states
        return self.agent.get_4_state(self.goal.get_position())

    def episode_finished(self):
        agent_pos = self.agent.get_position()
        goal_pos = self.goal.get_position()
        distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        if distance < GOAL_THRESHOLD:
            return True # episode finished if agent already at goal
        else:
            return False

    def get_reward(self): 
        # penalize distance to goal
        agent_pos = self.agent.get_normalized_position()
        goal_pos = self.goal.get_normalized_position()
        distance = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        reward = -distance 
        return reward

    def plot(self):
        # stepwise refreshing of plot
        ax.clear()
        
        # plotting of AGENT, GOAL and set AXIS LIMITS
        self.goal.plot(ax)
        self.agent.plot(ax)
        tmp_path = np.array(self.path)
        ax.plot(tmp_path[:,0], tmp_path[:,1], 'c-', linewidth=3)

        ax.set_xlim([-WIDTH/2, WIDTH/2])
        ax.set_ylim([-HEIGHT/2, HEIGHT/2])
    

    def run(self):

        self.agent = agents.Arm(angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
        # (-0.34, 3.4)
        # (-2.5, 0.01) 
        #self.agent.theta = np.array([2.5, -1.8]) # SCENE 1
        #self.agent.theta = np.array([0.7, -0.3]) # SCENE 2
        self.agent.theta = np.array([60., -90.])*np.pi/180. # SCENE 3

        self.cumulative_reward = 0.

        for _ in range(1):#while True: 
            # init new episode
            plotting_lock.acquire()
            self.goal = goals.Goal_Arm(ARM_LENGTH_1, ARM_LENGTH_2)
            #self.goal.pos = np.array([20,10]) # SCENE 1
            #self.goal.pos = np.array([-10,10]) # SCENE 2
            self.goal.pos = np.array([20.,-10.]) # SCENE 3
            plotting_lock.release()

            for step in range(self.MAX_STEPS):
                # produce experience
                self.agent.pos = self.agent.get_end_effector_position()
                state = self.get_state()
                print state 
                print self.goal.pos
                print self.agent.get_end_effector_position()
                quit()
                #print step, state, self.goal.get_position(), self.agent.get_position()
                self.path.append(self.agent.get_end_effector_position())

                # get lock to synchronize threads
                if sys.argv[1]=='DQN':
                    networks_lock.acquire()
                    q = networks.online_net.predict(state.reshape(1,NUM_OF_STATES), batch_size=1)
                    networks_lock.release()
                    action = np.argmax(q) # choose best action from Q(s,a)
                elif sys.argv[1]=='IK':
                    # explore with guidance of inverse kinematics
                    try:
                        u = self.agent.get_control(self.goal.pos - self.agent.pos)
                        if np.argmax(np.abs(u)) == 0:
                            if u[0] < 0:
                                action = 0
                            else:
                                action = 1   
                        else:
                            if u[1] < 0:
                                action = 2
                            else:
                                action = 3
                    except:
                        action = np.random.randint(0, NUM_OF_ACTIONS) # choose random action if singularity
                else:
                    print 'need to define MODE = {DQN, IK}'
                    quit()

                # take action, observe next state s'
                self.agent.set_action(action)
                self.agent.update()
                next_state = self.get_state()

                self.cumulative_reward += self.get_reward()
                
                # plot the scene
                plotting_lock.acquire()
                self.plot()
                plotting_lock.release()

                # check if agent at goal
                if self.episode_finished():
                    break # start new episode

            plt.xlabel('x-axis [cm]', size=15)
            plt.ylabel('y-axis [cm]', size=15)
            print 'path length =',len(self.path)
            print 'cumulative reward =',self.cumulative_reward


if __name__ == "__main__":
    # create GLOBAL thread-locks
    console_lock = threading.Lock()
    networks_lock = threading.Lock()
    plotting_lock = threading.Lock()

    # create GLOBAL Q-NETWORKS
    networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)

    # initialize GLOBAL plotting
    fig, ax = plt.subplots(1,1)
    plt.ion()

    # create threads
    threads = []
    threads.extend([Actor(i) for i in range(NUM_OF_ACTORS)])

    # set daemon, allowing Ctrl-C
    for i in range(len(threads)):
        threads[i].daemon = True

    # start new Threads
    [threads[i].start() for i in range(len(threads))]
 
    # show plot
    plt.show()
    while True:
        plotting_lock.acquire()
        fig.canvas.flush_events()
        plotting_lock.release()
        time.sleep(0.1)