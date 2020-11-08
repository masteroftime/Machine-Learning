import numpy as np
from random import random, randrange
import gym

epsilon = 0.1  # parameter for epsilon-greedy policy
alpha = 0.1  # learning rate
gamma = 1  # discounting factor


class QAgent:
    def __init__(self, env, dims):
        self.env = env
        self.q = np.zeros(dims)
        self.dims = dims

    def policy_greedy(self, state):
        qs = self.q[state]

        # return maximum value, break ties randomly
        return np.random.choice(np.flatnonzero(np.isclose(qs, qs.max())))

    def policy_epsilon(self, state):
        if random() < epsilon:
            return randrange(self.dims[-1])
        else:
            return self.policy_greedy(state)

    def training(self, num_epochs):
        for i in range(num_epochs):
            state = self.env.reset()
            done = False

            while not done:
                action = self.policy_epsilon(state)
                new_state, reward, done, _ = self.env.step(action)

                if done:
                    self.q[state + (action, )] += alpha * (
                        reward - self.q[state + (action, )])
                else:
                    self.q[state + (action, )] += alpha * (
                        reward + gamma * max(self.q[new_state]) -
                        self.q[state + (action, )])

                state = new_state

    def evaluate(self, num_epochs):
        wins = 0
        draws = 0
        losses = 0

        for i in range(num_epochs):
            state = self.env.reset()
            done = False

            while not done:
                action = self.policy_greedy(state)
                new_state, reward, done, _ = self.env.step(action)

            if reward > 0:
                wins += 1
            elif reward == 0:
                draws += 1
            else:
                losses += 1

        return wins, draws, losses


#  - sum of cards in hand (4-21)
#  - dealers card (1-10)
#  - usable ace (0-1)
#  - action (0-1)
dims = (18, 10, 2, 2)

class BlackjackWrapper:
  def __init__(self):
    self.env = gym.make('Blackjack-v0')

  def map_state(self, state):
    return (state[0] - 4, state[1] - 1, int(state[2]))
  
  def reset(self):
    return self.map_state(self.env.reset())
  
  def step(self, action):
    new_state, reward, done, _ = self.env.step(action)
    return self.map_state(new_state), reward, done, _

env = BlackjackWrapper()
agent = QAgent(env, dims)
