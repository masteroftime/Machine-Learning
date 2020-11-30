# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import default_rng
import gym

rng = default_rng()

class TableAgent:
    def __init__(self, env):
        # assert(isinstance(env.action_space, gym.spaces.Discrete))
        # assert(isinstance(env.observation_space, gym.spaces.Discrete))
        
        self.env = env
        
        self.epsilon = 0.2
        self.gamma = 0.9
        
        n_actions = env.action_space.n
        n_observations = env.observation_space.n
            
        self.q = np.zeros((n_observations, n_actions))
        
        self.rewards = []
        self.steps = []
    
    def policy_greedy(self, state):
        actions = self.q[state]
        
        max_actions = np.where(actions == np.max(actions))[0]
        
        if len(max_actions) > 1:
            return rng.choice(max_actions)
        else:
            return max_actions[0]
    
    def policy_epsilon(self, state):
        if rng.random() < self.epsilon:
            return rng.integers(len(self.q[state]))
        else:
            return self.policy_greedy(state)
    
    def avg_rewards(self):
        sum = 0
        avg = np.empty(len(self.rewards))
        for i in range(len(self.rewards)):
            sum += self.rewards[i]
            avg[i] = sum / (i+1)
        return avg
    
    def evaluate(self, n=1000):
        sum_rewards = 0
        sum_steps = 0
        
        for i in range(n):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy_greedy(state)
                state, reward, done, _ = self.env.step(action)
                sum_rewards += reward
                sum_steps += 1
        
        return sum_rewards/n, sum_steps/n
            
class MonteCarloAgent(TableAgent):
    def __init__(self, env):
        super().__init__(env)
        self.n = np.zeros(self.q.shape, dtype=int)

        
    def train(self, episodes=1000, verbose=True):        
        for i in range(episodes):
            state = self.env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            G = 0
            
            self.rewards.append(0)
            self.steps.append(0)
            
            while not done:
                action = self.policy_epsilon(state)
                states.append(state)
                actions.append(action)
                
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                
                self.rewards[-1] += reward
                self.steps[-1] += 1
            
            if verbose:
                print(f'Episode {i+1} - reward: {sum(rewards)}   steps: {len(states)}')
            
            for j in range(1,len(states)+1):
                s = states[-j]
                a = actions[-j]
                G = self.gamma*G + rewards[-j]
                self.n[s,a] += 1
                self.q[s,a] += (G - self.q[s,a]) / self.n[s,a]
                
                

env = gym.make('FrozenLake-v0', is_slippery=True)
agent = MonteCarloAgent(env)


for i in range(1,20):
    print(f'{i} ...')
    agent.epsilon = 0.2/i
    agent.train(10000, verbose=False)
    print(f'Epsilon -> avg reward: {agent.avg_rewards()[-1]}')
    
    avg_reward, avg_steps = agent.evaluate()
    print(f'Greedy -> avg reward: {avg_reward}   avg steps: {avg_steps}')