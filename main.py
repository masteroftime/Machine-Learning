import numpy as np
from random import random, randrange
import gym

epsilon = 0.1  # parameter for epsilon-greedy policy
alpha = 0.1  # learning rate
gamma = 1  # discounting factor


class QAgent:
    def __init__(self, env, dims, map_func=None):
        self.env = env
        self.q = np.zeros(dims)
        self.map = map_func

    def policy_greedy(self, state):
        qs = self.q[state]

        if qs[0] == qs[1]:
            return randrange(2)
        else:
            return qs.argmax()

    def policy_epsilon(self, state):
        if random() < epsilon:
            return randrange(2)
        else:
            return self.policy_greedy(state)

    def training(self, num_epochs):
        for i in range(num_epochs):
            state = self.env.reset()
            if self.map is not None:
                state = self.map(state)
            done = False

            while not done:
                action = self.policy_epsilon(state)
                new_state, reward, done, _ = self.env.step(action)
                if self.map is not None:
                    new_state = self.map(new_state)

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
            if self.map is not None:
                state = self.map(state)
            done = False

            while not done:
                action = self.policy_greedy(state)
                new_state, reward, done, _ = self.env.step(action)
                if self.map is not None:
                    state = self.map(new_state)

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


def map_state(state):
    return (state[0] - 4, state[1] - 1, int(state[2]))


env = gym.make('Blackjack-v0')
agent = QAgent(env, dims, map_state)
