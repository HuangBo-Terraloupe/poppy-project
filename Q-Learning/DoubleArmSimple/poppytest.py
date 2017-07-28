from poppy.creatures import PoppyHumanoid
import numpy as np
import cv2

# own modules
import goals
#import q_networks
import poppy_qnetwork as q_network
import vision
import vrep_draw


# real poppy: r_elbow motor clockwise!


ARM_LENGTH_1 = 15.1
ARM_LENGTH_2 = 10.1
ARM_OFFSET = 10.95
ANGULAR_ARM_VELOCITY = 1
GOAL_THRESHOLD = 0.5
NUM_OF_ACTIONS = 4
NUM_OF_STATES = 4
WAITING = False


def set_left_action(poppy, action):
    if action==0:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position - ANGULAR_ARM_VELOCITY, 0, wait=WAITING)
    elif action==1:
        poppy.l_shoulder_x.goto_position(poppy.l_shoulder_x.present_position + ANGULAR_ARM_VELOCITY, 0, wait=WAITING) 
    elif action==2:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position - ANGULAR_ARM_VELOCITY, 0, wait=WAITING)
    elif action==3:
        poppy.l_elbow_y.goto_position(poppy.l_elbow_y.present_position + ANGULAR_ARM_VELOCITY, 0, wait=WAITING)
    return


def set_right_action(poppy, action):
    if action==0:
        poppy.r_shoulder_x.goto_position(poppy.r_shoulder_x.present_position - ANGULAR_ARM_VELOCITY, 0, wait=WAITING)
    elif action==1:
        poppy.r_shoulder_x.goto_position(poppy.r_shoulder_x.present_position + ANGULAR_ARM_VELOCITY, 0, wait=WAITING) 
    elif action==2:
        poppy.r_elbow_y.goto_position(poppy.r_elbow_y.present_position + ANGULAR_ARM_VELOCITY, 0, wait=WAITING)
    elif action==3:
        poppy.r_elbow_y.goto_position(poppy.r_elbow_y.present_position - ANGULAR_ARM_VELOCITY, 0, wait=WAITING)
    return

def get_right_poppy_angles(poppy):
    theta = [0.,0.]
    theta[0] = float(poppy.r_shoulder_x.present_position)
    theta[1] = -float(poppy.r_elbow_y.present_position)
    return np.array(theta)

def get_left_poppy_angles(poppy):
    theta = [0.,0.]
    theta[0] = float(poppy.l_shoulder_x.present_position)
    theta[1] = float(poppy.l_elbow_y.present_position)
    return np.array(theta)

def get_normed_right_poppy_angles(poppy):
    JOINT_LIMITS = ((-195., 20.), (-1., 148.)) # RIGHT ARM 
    # normalize thetas to [-1.0, +1.0]
    theta = get_right_poppy_angles(poppy)

    range1_half = 0.5*(JOINT_LIMITS[0][1] - JOINT_LIMITS[0][0])
    theta1 = (theta[0] - range1_half)/range1_half

    range2_half = 0.5*(JOINT_LIMITS[1][1] - JOINT_LIMITS[1][0])
    theta2 = (theta[1] - range2_half)/range2_half        
    return np.array([theta1, theta2])

def get_normed_left_poppy_angles(poppy):
    JOINT_LIMITS = ((-20., 195.), (-148., 1.)) # LEFT ARM
    # normalize thetas to [-1.0, +1.0]
    theta = get_left_poppy_angles(poppy)

    range1_half = 0.5*(JOINT_LIMITS[0][1] - JOINT_LIMITS[0][0])
    theta1 = (theta[0] - range1_half)/range1_half

    range2_half = 0.5*(JOINT_LIMITS[1][1] - JOINT_LIMITS[1][0])
    theta2 = (theta[1] - range2_half)/range2_half        
    return np.array([theta1, theta2])

def get_left_poppy_position(poppy):
    # Forward Kinematics
    theta = get_left_poppy_angles(poppy)*np.pi/180.
    l_pos = np.array([0.0, 0.0])
    l_pos[0] = ARM_LENGTH_1*np.cos(theta[0]) + ARM_LENGTH_2*np.cos(theta[0]+theta[1])
    l_pos[1] = ARM_OFFSET + ARM_LENGTH_1*np.sin(theta[0]) + ARM_LENGTH_2*np.sin(theta[0]+theta[1])
    return l_pos

def get_right_poppy_position(poppy):
    # Forward Kinematics
    theta = get_right_poppy_angles(poppy)*np.pi/180.
    r_pos = np.array([0.0, 0.0])
    r_pos[0] = ARM_LENGTH_1*np.cos(theta[0]) + ARM_LENGTH_2*np.cos(theta[0]+theta[1])
    r_pos[1] = -ARM_OFFSET + ARM_LENGTH_1*np.sin(theta[0]) + ARM_LENGTH_2*np.sin(theta[0]+theta[1])
    return r_pos

def get_left_4_state(poppy, goal_pos):
    # normalized [dx, dy]
    dist = (goal_pos - get_left_poppy_position(poppy)) / (ARM_LENGTH_1+ARM_LENGTH_2)

    # normalized [theta_1, theta_2]
    theta = get_normed_left_poppy_angles(poppy)

    # state = [dx,dy,theta1,theta2]
    return np.array([dist, theta])

def get_right_4_state(poppy, goal_pos):
    # normalized [dx, dy]
    dist = (goal_pos - get_right_poppy_position(poppy)) / (ARM_LENGTH_1+ARM_LENGTH_2)

    # normalized [theta_1, theta_2]
    theta = get_normed_right_poppy_angles(poppy)

    # state = [dx,dy,theta1,theta2]
    return np.array([dist, theta])

def episode_finished(poppy, goal_pos):
    agent_pos = get_left_poppy_position(poppy)
    l_distance = np.linalg.norm(agent_pos - goal_pos)

    agent_pos = get_right_poppy_position(poppy)
    r_distance = np.linalg.norm(agent_pos - goal_pos)

    if min(l_distance, r_distance) < GOAL_THRESHOLD:
        return True # episode finished if agent already at goal
    else:
        return False


if __name__ == "__main__":
    # load qnetwork
    #network_l = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES, network_name='online_network_left')
    #network_r = q_networks.QNetworks(NUM_OF_ACTIONS, NUM_OF_STATES, network_name='online_network_right')
    network_l = q_network.QNetwork(network_name='online_network_left')
    network_r = q_network.QNetwork(network_name='online_network_right')

    # init poppy
    poppy = PoppyHumanoid(simulator='vrep')
    poppy.reset_simulation()
    
    #init goal plot
    plt = vrep_draw.Plot(poppy)

    # init camera
    #camera = cv2.VideoCapture(0)

    # activate motors
    L_MOTORS = [poppy.l_shoulder_x, poppy.l_shoulder_y, poppy.l_elbow_y, poppy.l_arm_z]
    R_MOTORS = [poppy.r_shoulder_x, poppy.r_shoulder_y, poppy.r_elbow_y, poppy.r_arm_z]
    for m in L_MOTORS:
        m.compliant = False

    # move to initial position
    t = 1
    poppy.l_shoulder_y.goto_position(-90,t,wait=True)
    poppy.l_arm_z.goto_position(-90,t,wait=True)
    poppy.r_shoulder_y.goto_position(-90,t,wait=True)
    poppy.r_arm_z.goto_position(90,t,wait=True)

    # some starting position
    poppy.l_elbow_y.goto_position(-90,t,wait=True)
    poppy.l_shoulder_x.goto_position(60,t,wait=True)

    poppy.r_elbow_y.goto_position(-90,t,wait=True)
    poppy.r_shoulder_x.goto_position(-60,t,wait=True)

    # set initial goal position
    #goal_pos = np.array([0.,0.])
    #print 'WARNING: set initial goal to a reachable position!'

    # simulated GOALS
    goal = goals.Goal_Arm(ARM_LENGTH_1, ARM_LENGTH_2)
    print "WARNING: goal position space != robot's working space"

    while True:
        # get goal (VISION)
        #_, image = camera.read()
        #tmp_goal = vision.getGoalPosition(image)
        #if tmp_goal: # only update goal if a goal could be detected
        #    goal_pos = tmp_goal
        #cv2.imshow('image', image)
        #cv2.waitKey(1)

        # get goal (simulated GOALS)
        goal_pos = np.array([25.,ARM_OFFSET]) 
        #goal_pos = goal.get_position()

        # get state
        l_state = get_left_4_state(poppy, goal_pos)
        r_state = get_right_4_state(poppy, goal_pos)
        #print "goal=[%s] | left=[%s] | right=[%s]" % (goal_pos, get_left_poppy_position(poppy), get_right_poppy_position(poppy))

        # stop actions when goal reached
        if not episode_finished(poppy, goal_pos):
            #q_l = network_l.online_net.predict(l_state.reshape(1, NUM_OF_STATES), batch_size=1)
            #q_r = network_r.online_net.predict(r_state.reshape(1, NUM_OF_STATES), batch_size=1)
            q_l = network_l.predict(l_state.reshape(1, NUM_OF_STATES).tolist()[0])
            q_r = network_r.predict(r_state.reshape(1, NUM_OF_STATES).tolist()[0])

            # choose best action from Q(s,a)
            dist_l = np.linalg.norm(goal_pos-np.array([0.,ARM_OFFSET]))
            dist_r = np.linalg.norm(goal_pos-np.array([0.,-ARM_OFFSET]))
            if dist_l < dist_r:
            #if np.max(q_r) < np.max(q_l):
                action = np.argmax(q_l) 
                set_left_action(poppy, action)
            else:
                action = np.argmax(q_r)
                set_right_action(poppy, action)
            print "WARNING: robot arms could collide!" 
        else:
            print 'reached goal position'           
    
    #cv2.destroyAllWindows()
    #camera.release()
    poppy.close()