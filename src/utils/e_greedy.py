import numpy as np
import random
import gymnasium as gym

def greedy_policy(qvalues):
	# Exploitation: take the action with the highest state, action value
	action = np.argmax(qvalues)
	return action

def epsilon_greedy_policy(qvalues, epsilon):
	# Randomly generate a number between 0 and 1
	random_num = random.uniform(0,1)
	# if random_num > greater than epsilon -> exploitation
	if random_num > epsilon:
		# Take th action with the highest value given a state
		action = greedy_policy(qvalues)
	else: # Exploration
		action = np.random.choice(len(qvalues))

	return action

	
