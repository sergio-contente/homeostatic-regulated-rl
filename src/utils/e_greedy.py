import numpy as np
import random
import gymnasium as gym

def greedy_policy(Qtable, state):
	# Exploitation: take the action with the highest state, action value
	action = np.argmax(Qtable[state][:])
	return action

def epsilon_greedy_policy(Qtable, state, epsilon, env):
	# Randomly generate a number between 0 and 1
	random_num = random.uniform(0,1)
	# if random_num > greater than epsilon -> exploitation
	if random_num > epsilon:
		# Take th action with the highest value given a state
		action = greedy_policy(Qtable, state)
	else: # Exploration
		action = env.action_space.sample()

	return action

	
