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
	# if random_num <  epsilon -> Exploration
	if random_num < epsilon:
		action = np.random.choice(len(qvalues))
	else: # Exploitation
		# Take th action with the highest value given a state
		action = greedy_policy(qvalues)
	return action

	
