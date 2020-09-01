import numpy as np
from random import random, randrange
import gym


training_epochs = 1000
epsilon = 0.1  # parameter for epsilon-greedy policy
alpha = 0.1    # learning rate
gamma = 1      # discounting factor

env = gym.make('Blackjack-v0')

# q-table for all the states:
#  - sum of cards in hand (4-21)
#  - dealers card (1-10)
#  - usable ace (0-1)
#  - action (0-1)
q = np.zeros((18,10,2,2))

def map_state(state):
    return (state[0]-4, state[1]-1, int(state[2]))

def policy_greedy(state):
    qs = q[state]
    
    if qs[0] == qs[1]:
        return randrange(2)
    else:
        return qs.argmax()

def policy_epsilon(state):
    if random() < epsilon:
        return randrange(2)
    else:
        return policy_greedy(state)
    
def training(num_epochs):
    for i in range(num_epochs):
        state = map_state(env.reset())
        done = False
        
        while not done:
            action = policy_epsilon(state)
            new_state, reward, done, _ = env.step(action)
            new_state = map_state(new_state)
            
            if done:
                q[state + (action,)] += alpha * (reward - q[state + (action,)])
            else:
                q[state + (action,)] += alpha * (reward + gamma*max(q[new_state]) - q[state + (action,)])
            
            state = new_state

def evaluate(num_epochs):
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(num_epochs):
        state = map_state(env.reset())
        done = False
        
        while not done:
            action = policy_greedy(state)
            new_state, reward, done, _ = env.step(action)
            state = map_state(new_state)
            
        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    
    return wins, draws, losses
        
  

