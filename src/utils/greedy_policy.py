import numpy as np

def greedy_policy(Qtable, state):
	# Exploitation: take the action with the highest state, action value
	action = np.argmax(Qtable[state][:])
	return action
