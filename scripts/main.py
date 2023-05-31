#! /usr/bin/env python3


import rospy
import copy
import numpy as np


from sac import SAC
from environment import Env
from stage_diferencial import Robot

from std_msgs.msg import String

def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action

is_training = True

max_episodes  = 10001
max_steps   = 10000#1000
rewards     = []
batch_size  = 256#512

action_dim = 2
state_dim  = 545
hidden_dim = 1635
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -0.25 # rad
ACTION_V_MAX = 0.5 # m/s
ACTION_W_MAX = 0.25 # rad
world = 'create_hokuyo'
buffer_size = 1000000#50000

print(" ")
print("---------------------------------")
print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad')
print("---------------------------------")
print(" ")

agent = SAC(state_dim, action_dim, buffer_size)

ep_0 = 0



if __name__ == '__main__':
    rospy.init_node('sac_stage_controller')
    status_pub = rospy.Publisher("status", String, queue_size=10)
    reward_pub = rospy.Publisher("reward", String, queue_size=10)

    

    if (ep_0 != 0):
        agent.load_models(agent.policy, agent.critic_target, ep_0)

    
    #result = Float32()
    env = Env(state_dim, action_dim, max_steps)
    before_training = 4
    past_action = np.array([0.,0.,0.0])

    rospy.loginfo("Starting at episode: %s ", str(ep_0))
    status = "Pose: ({:.2f}, {:.2f}) - {:.2f} - Target: ({:.2f}, {:.2f})".format(env.robot.pose[0], env.robot.pose[1], env.robot.distance, env.robot.target[0], env.robot.target[1])
    status_pub.publish(status)
    for ep in range(ep_0, max_episodes):
        done = False
        state = env.reset(ep)        
        if is_training and not ep%10 == 0 and len(agent.memory) > before_training*batch_size:
            rospy.loginfo("---------------------------------")
            rospy.loginfo("Episode: %s training", str(ep))
            #rospy.loginfo("---------------------------------")
            
        else:
            if len(agent.memory) > before_training*batch_size:
                rospy.loginfo("---------------------------------")
                rospy.loginfo("Episode: %s evaluating", str(ep))
                rospy.loginfo("---------------------------------")
            else:
                rospy.loginfo("---------------------------------")
                rospy.loginfo("Episode: %s adding to memory", str(ep))
                rospy.loginfo("---------------------------------")

        rewards_current_episode = 0.

        for step in range(max_steps):
            state = np.float32(state)
            # print('state___', state)
            if is_training and not ep%10 == 0:
                action = agent.get_action(state)
            else:
                action = agent.get_action(state, eval=True)

            if not is_training:
                action = agent.get_action(state, eval=True)
            
            
            unnorm_action = np.array([action_unnormalized(action[0], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)])

            next_state, reward, done = env.step(unnorm_action)
            past_action = copy.deepcopy(action)
            reward_pub.publish("Reward: {}".format(reward))
            rewards_current_episode += reward
            next_state = np.float32(next_state)
            if not ep%10 == 0 or not len(agent.memory) > before_training*batch_size:
                if reward >= 200:
                    rospy.loginfo("--------- Maximum Reward ----------")
                    for _ in range(3):
                        agent.memory.push(state, action, reward, next_state, done)
                else:
                    agent.memory.push(state, action, reward, next_state, done)
            
            if len(agent.memory) > before_training*batch_size and is_training and not ep% 10 == 0:
                agent.update(batch_size)
            state = copy.deepcopy(next_state)

            status = "Pose: ({:.2f}, {:.2f}) - {:.2f} - Target: ({:.2f}, {:.2f})".format(env.robot.pose[0], env.robot.pose[1], env.robot.distance, env.robot.target[0], env.robot.target[1])
            status_pub.publish(status)

            if done:
                rospy.loginfo("Reward per ep: %s", str(rewards_current_episode))
                rospy.loginfo("Break step: %s", str(step))
                rospy.loginfo("Pose: ({:.2f}, {:.2f}) - {:.2f} - Target: ({:.2f}, {:.2f})".format(env.robot.pose[0], env.robot.pose[1], env.robot.distance, env.robot.target[0], env.robot.target[1]))
                rospy.loginfo("Entropy: %s", str(agent.target_entropy))
                rospy.loginfo("---------------------------------")
                break
            
            
            #rospy.Rate(100).sleep()

        if not done:
            rospy.loginfo("Reward per ep: %s", str(rewards_current_episode))
            rospy.loginfo("Break step: %s", str(step))
            rospy.rospy.loginfo("Entropy: %s", str(agent.target_entropy))
            rospy.loginfo("---------------------------------")
        
        if ep%20 == 0:
            agent.save_models(agent.policy, agent.critic, world, ep)

    rospy.spin()
