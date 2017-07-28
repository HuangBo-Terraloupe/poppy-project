# -*- coding: utf-8 -*-
"""
Created on Mon May 15 01:44:54 2017

@author: Fares
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import time as ti
import os
import random
#import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from keras.callbacks import Callback

import pyglet
import Arm
import math
from math import cos,sin,radians

EPISODES = 5000


#fig = plt.figure(figsize=(5, 2.5))

#global window
#global label
#global arm

def makeCircle(X,Y,r=6,numPoints=20):
    verts = []
    for i in range(numPoints):
        angle = radians(float(i)/numPoints * 360.0)
        x = r*cos(angle) + X
        y = r*sin(angle) + Y
        verts += [x,y]
    
    #print numPoints, verts
    return pyglet.graphics.vertex_list(numPoints, ('v2f', verts))
    

class GraphicArm(pyglet.window.Window):
    
    def __init__(self):
        super(GraphicArm,self).__init__()
        #window = pyglet.window.Window()
        
        self.arm_length = [100,100,100]
        self.goal_pos=[237,50]
        self.max_angles = [math.pi/2.0, math.pi, math.pi/4]
        self.min_angles = [0, 0, -math.pi/4]
        self.init_js_pos = [math.pi/4, math.pi/4, 0]
        
        self.arm = Arm.Arm2Link(L = np.array(self.arm_length), q0 = np.array(self.init_js_pos))
        self.arm.max_angles = self.max_angles
        self.arm.min_angles = self.min_angles
        
        self.init_pos = self.arm.q
        
        print(self.arm.get_xy(self.init_pos))
        
        self.goal_gl = pyglet.graphics.vertex_list(1,
                                            ('v2i', (self.goal_pos[0],self.goal_pos[1])))
        self.goal_circ = makeCircle(self.goal_pos[0],self.goal_pos[1])
        self.init_circ = None
        
        self.label = pyglet.text.Label('Pos (x,y)', font_name='Times New Roman', 
            font_size=36, x=self.width//2, y=abs(self.height/(1.5)),
            anchor_x='center', anchor_y='center')
              # create an instance of th
              
        self.path = []
        # make our window for drawin'
        #window = pyglet.window.Window()
        
    
    def rearm(self):
        self.clear()
        del self.arm
        self.arm = Arm.Arm2Link(L = np.array(self.arm_length), q0 = np.array(self.init_js_pos))
        self.arm.max_angles = self.max_angles
        self.arm.min_angles = self.min_angles
        self.path = []
        
    def get_joint_positions(self):
        """This method finds the (x,y) coordinates of each joint"""

        x = np.array([ 0, 
            self.arm.L[0]*np.cos(self.arm.q[0]),
            self.arm.L[0]*np.cos(self.arm.q[0]) + self.arm.L[1]*np.cos(self.arm.q[0]+self.arm.q[1]),
            self.arm.L[0]*np.cos(self.arm.q[0]) + self.arm.L[1]*np.cos(self.arm.q[0]+self.arm.q[1]) + 
                self.arm.L[2]*np.cos(np.sum(self.arm.q)) ]) + self.width/2

        y = np.array([ 0, 
            self.arm.L[0]*np.sin(self.arm.q[0]),
            self.arm.L[0]*np.sin(self.arm.q[0]) + self.arm.L[1]*np.sin(self.arm.q[0]+self.arm.q[1]),
            self.arm.L[0]*np.sin(self.arm.q[0]) + self.arm.L[1]*np.sin(self.arm.q[0]+self.arm.q[1]) + 
                self.arm.L[2]*np.sin(np.sum(self.arm.q)) ])

        return np.array([x, y]).astype('int')
    
    def start(self):
        self.jps =self.get_joint_positions()
        self.init_circ = makeCircle(self.jps[0][2],self.jps[1][2])
    
    
    #pyglet.app.run()
    
    def get_last_hand_pos(self):
        return [self.jps[0][2],self.jps[1][2]]
    
    def on_close(self):
        print "Exiting"
        pyglet.app.exit()
        
    def draw_all(self):
        self.label.draw()
        self.goal_gl.draw(pyglet.gl.GL_POINTS)
        self.goal_circ.draw(pyglet.gl.GL_POINTS)
        self.init_circ.draw(pyglet.gl.GL_POINTS)
        if (len(self.path) > 1):
            #print len(self.path), self.path
            vl = pyglet.graphics.vertex_list(len(self.path)//2, ('v2i', self.path))
            vl.draw(pyglet.gl.GL_POINTS)
        else:
            print "Path is empty!"
                
        xy = [self.jps[0][2],self.jps[1][2]]
        c = makeCircle(xy[0],xy[1])
        c.draw(pyglet.gl.GL_POINTS)
    
    #X [[320 222 288 288]
    #Y  [ 0  22  97  97]] >  [-31.815142936974894, 97.550059859613171]
    def on_draw(self):
        self.clear()
        self.draw_all()
        for i in range(3): 
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', 
                (self.jps[0][i], self.jps[1][i], 
                 self.jps[0][i+1], self.jps[1][i+1])))
            
        #print "lol"

    def set_new_angles(self,t1,t2):
        self.arm.q = [self.arm.q[0]+t1,self.arm.q[1]+t2,0]
        self.jps = self.get_joint_positions() # get new joint (x,y) positions

    def grend(self):
        # call the inverse kinematics function of the arm
        # to find the joint angles optimal for pointing at 
        # this position of the mouse 
        #self.label.text = '(x,y) = (%.3f, %.3f)'%(self.arm.q[0],self.arm.q[1])
        #arm.q = arm.inv_kin([x - window.width/2, y]) # get new arm angle
        self.jps = self.get_joint_positions() # get new joint (x,y) positions
        xy = self.get_last_hand_pos()
        self.path.append(xy[0])
        self.path.append(xy[1])
        
        #self.label.text = '(x,y) = (%.3f, %.3f)'%(self.jps[0][2],self.jps[1][2])
        self.label.text = '(x,y) = (%.3f, %.3f)'%(self.jps[0][2],self.jps[1][2])
        
        #print self.get_joint_positions(), "> ", self.arm.get_xy()
        #pyglet.clock.time.sleep(0.1)
        self.set_caption(pyglet.clock.get_fps().__str__())
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()

        pyglet.clock.tick()
        
        
class RobotArmEnv:
    def __init__(self):
        self.state_size = 6 #current pos, goal pos, angle mot1, angle mot2
        self.action_size = 4 #move 2 motors +1/-1
        self.delta = math.radians(1) #how much change angles

        
        self.garm = GraphicArm()
        self.garm.start()
        xy=self.garm.arm.get_xy()
        self.init_state = [xy[0],xy[1],self.garm.arm.q[0],self.garm.arm.q[1],self.garm.goal_pos[0],self.garm.goal_pos[1]]
        
    
    def step(self,action):
        #print action
        if action == 0:
            self.garm.arm.q[0] += self.delta
                        
        elif action == 1:
            self.garm.arm.q[0] -= self.delta
        elif action == 2:
            self.garm.arm.q[1] += self.delta
        elif action == 3:
            self.garm.arm.q[1] -= self.delta
        else:
            print "Error!"
            
        self.garm.arm.q[0] %= (2. * np.pi)
        self.garm.arm.q[1] %= (2. * np.pi)                      
        #
        
        #wait until reached
        #FIXME
        xy=self.garm.get_last_hand_pos()
        t=self.garm.arm.q
        next_state = [xy[0],xy[1],t[0],t[1],self.garm.goal_pos[0],self.garm.goal_pos[1]]
        #print next_state          
        if (xy == self.garm.goal_pos):
            print "Goal Reached!"
            reward = 100
            done = True
        else:
            reward = -1
            done = False
            
        if (xy[1] < 0):
            #print "Over limits!"
            reward = -10
            
        init_distance = 1.*np.linalg.norm(np.array(self.garm.init_pos[0:1]) - np.array(self.garm.goal_pos))
        distance_to_goal = 1.*np.linalg.norm(np.array(xy) - np.array(self.garm.goal_pos))  
        if ((distance_to_goal/init_distance) < 0.1):
            reward = 0
            print "Dist: ",distance_to_goal," ", init_distance
            
            
        #reward = -1
        #done = False
        info = "dummy"
        return next_state, reward, done, info
        
    def render(self):
        self.garm.grend()
        pass
    
    def reset(self):
        print "Resetting env"
        #self.garm.arm.q = self.garm.init_pos
        self.garm.rearm()
        self.render()
        print self.garm.get_last_hand_pos()
        #pyglet.clock.time.sleep(2)
        return self.init_state 
        
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = .999
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(next_state)[0])
            X[i], Y[i] = state, target
            #plt.plot(X[i][0:1], Y[i][0:1], label='data')
            #plt.show()
        hist = self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        
        print hist.epoch,hist.history
        
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = RobotArmEnv()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    if (os.path.isfile("./dqn-agent.txt") == True):
        agent.load("./dqn-agent.txt")
    #deadline = 4000
    deadline = 2000
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        for time in range(deadline):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -10
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            #TODO: every 100 episode change goal
            #TODO: reward should be a function of distance (or discrete)
            if done or time == (deadline-1):
                print("episode: {}/{}, score: {}, e: {:.2}, M: {}"
                      .format(e, EPISODES, score, agent.epsilon,len(agent.memory)))
                if not done:
                    print "Time elapsed"
                else:
                    print "Reached in ",time, " its"
                break
        agent.replay(32)
        
        #if e % 10 == 0:
        agent.save("./dqn-agent.txt")
        agent.memory
        #np.savetxt("./dqn-memory.txt",np.asarray(agent.memory))#np.hstack((agent.epsilon, agent.memory))))
