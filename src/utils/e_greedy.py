import numpy as np
from greedy_policy import greedy_policy
import random
import gymnasium as gym

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

	
