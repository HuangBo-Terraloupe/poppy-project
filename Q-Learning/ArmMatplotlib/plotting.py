#!/usr/bin/python
import numpy as np
np.set_printoptions(precision=4)
import pylab

# import own modules
import agents
import goals
import q_networks

ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1
LENGTH = ARM_LENGTH_1+ARM_LENGTH_2
ANGULAR_ARM_VELOCITY = 1.0*np.pi/180.0

NUM_OF_ACTIONS = 4
NUM_OF_ACTORS = 1
NUM_OF_STATES = 4


class Actor:
    def __init__(self):
        self.agent = None                       # place-holder for agent
        self.goal = None 					    # placer-holder for goal

    def get_state(self):
    	# state is composed by agent + goal states
    	return self.agent.get_4_state(self.goal.get_position())

    def plot(self,ax):
        # middle joint position
        pos = [ self.agent.base_pos[0] + ARM_LENGTH_1*np.cos(self.agent.theta[0]),
                (self.agent.base_pos[1] + ARM_LENGTH_1*np.sin(self.agent.theta[0]))]

        linewidth = 5
        markersize = 10

        # arm links
        h_arm1 = ax.plot(   [self.agent.base_pos[0], pos[0]], 
                            [self.agent.base_pos[1], pos[1]], 
                            'k', linewidth=linewidth)
        h_arm2 = ax.plot(   [pos[0], self.agent.pos[0]], 
                            [pos[1], self.agent.pos[1]], 
                            'k', linewidth=linewidth)

        # base, middle and end-effector joints
        h_base = ax.plot(   self.agent.base_pos[0], self.agent.base_pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)
        h_middle = ax.plot(  pos[0], pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)
        h_end = ax.plot(    self.agent.pos[0], self.agent.pos[1],
                            'ro', markersize=markersize, markeredgewidth=linewidth)

    def run(self):

        D = []

        self.agent = agents.Arm(angular_velocity_1=ANGULAR_ARM_VELOCITY, angular_velocity_2=ANGULAR_ARM_VELOCITY, arm_length_1=ARM_LENGTH_1, arm_length_2=ARM_LENGTH_2)
        self.agent.theta = np.array([0.7,-0.3], dtype=np.float32)
        self.agent.pos = self.agent.get_end_effector_position()
        self.goal = goals.Goal_Arm(ARM_LENGTH_1, ARM_LENGTH_2)

        N = 50
        GOALX = np.linspace(-LENGTH,LENGTH,N)
        GOALY = np.linspace(-LENGTH,LENGTH,N)

        for goal_x in GOALX:
            for goal_y in GOALY:

                self.goal.pos = np.array([goal_x, goal_y])  

                state = self.get_state()
                q = networks.online_net.predict(state.reshape(1,NUM_OF_STATES), batch_size=1)
                qmax = np.max(q)

                D.append([goal_x, goal_y, qmax])
                #print goal_x, goal_y, qmax

        D = np.array(D)
        zz = D[:,2].reshape(len(GOALX),len(GOALY))
        zz = zz.T
        pylab.pcolor(GOALX, GOALY, zz)
        pylab.title('max(Q(s,a)) for theta=(%.1f,%.1f) [rad]' % (self.agent.theta[0], self.agent.theta[1]), fontsize=20)
        pylab.xlabel('goal x-position [cm]', fontsize=15)
        pylab.ylabel('goal y-position [cm]', fontsize=15)
        pylab.xlim([-LENGTH,LENGTH])
        pylab.ylim([-LENGTH,LENGTH])
        self.plot(pylab)
        pylab.show()


if __name__ == "__main__":
    # create GLOBAL Q-NETWORKS
    networks = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES)

    actor = Actor()
    actor.run()