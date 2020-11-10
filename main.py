import numpy as np
from random import random, randrange
import gym

epsilon = 0.2  # parameter for epsilon-greedy policy
alpha = 0.02  # learning rate
gamma = 0.5  # discounting factor


class MCAgent:
    def __init__(self, env, dims):
        self.env = env
        self.q = np.zeros(dims)
        self.n = np.zeros(dims, dtype=np.int32)
        self.dims = dims

    def policy_greedy(self, state):
        qs = self.q[state]
        
        #if qs[0] == qs[1]:
        #    return randrange(2)
        #else:
        #    return qs.argmax()

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
            state_actions = []
            rewards = []
            G = 0

            while not done:
                action = self.policy_epsilon(state)
                state_actions.append(state + (action,))
                
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
            
            for t in range(len(state_actions)-1, -1, -1):
                G = gamma*G + rewards[t]
                self.n[state_actions[t]] += 1
                self.q[state_actions[t]] += (G - self.q[state_actions[t]]) / self.n[state_actions[t]]
                
                
                

class QAgent:
    def __init__(self, env, dims):
        self.env = env
        self.q = np.zeros(dims)
        self.dims = dims

    def policy_greedy(self, state):
        qs = self.q[state]
        
        #if qs[0] == qs[1]:
        #    return randrange(2)
        #else:
        #    return qs.argmax()

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
                    #oldq = self.q[state + (action, )]
                    self.q[state + (action, )] += alpha * (reward - self.q[state + (action, )])
                    #print(f'update: {oldq} -> {self.q[state + (action, )]}')
                else:
                    #oldq = self.q[state + (action, )]
                    self.q[state + (action, )] += alpha * (reward + gamma * max(self.q[new_state]) - self.q[state + (action, )])
                    #print(f'Update: {oldq} -> {self.q[state + (action, )]}')

                state = new_state

class SarsaAgent:
    def __init__(self, env, dims):
        self.env = env
        self.q = np.zeros(dims)
        self.dims = dims

    def policy_greedy(self, state):
        qs = self.q[state]
        
        if qs[0] == qs[1]:
            return randrange(2)
        else:
            return qs.argmax()

        # return maximum value, break ties randomly
        #return np.random.choice(np.flatnonzero(np.isclose(qs, qs.max())))
        

    def policy_epsilon(self, state):
        if random() < epsilon:
            return randrange(self.dims[-1])
        else:
            return self.policy_greedy(state)

    def training(self, num_epochs):
        for i in range(num_epochs):
            state = self.env.reset()
            action = self.policy_epsilon(state)
            done = False

            while not done:
                new_state, reward, done, _ = self.env.step(action)

                if done:
                    #oldq = self.q[state + (action, )]
                    self.q[state + (action, )] += alpha * (reward - self.q[state + (action, )])
                    #print(f'update: {oldq} -> {self.q[state + (action, )]}')
                else:
                    #oldq = self.q[state + (action, )]
                    new_action = self.policy_epsilon(new_state)
                    self.q[state + (action, )] += alpha * (reward + gamma * self.q[new_state + (new_action,)] - self.q[state + (action, )])
                    action = new_action
                    #print(f'Update: {oldq} -> {self.q[state + (action, )]}')

                state = new_state
    
class DiceEnvironment:
    dims = (6, 2)
    
    limit = 6
    
    def __init__(self):
        self.val = -1

    def reset(self):
        self.val = randrange(6)
        return (self.val,)
    
    def step(self, action):
        if action == 1 and self.val < self.limit:
            self.val += randrange(6)
            
        if self.val < self.limit:
            reward = self.val
        else:
            reward = 0
        
        return (self.val,), reward, True, []
            
        
class FrozenLakeEnvironment:
    dims = (16,4)
    
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')

    def map_state(self, state):
        return (state,)
  
    def reset(self):
        return (self.env.reset(),)
  
    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)
        return (new_state,), reward, done, _
    
    def evaluate_agent(self, agent, num_epochs=1000):
        wins = 0
        losses = 0
        win_steps = 0
        loss_steps = 0

        for i in range(num_epochs):
            state = self.reset()
            done = False
            steps = 0

            while not done:
                action = agent.policy_greedy(state)
                new_state, reward, done, _ = self.step(action)
                steps += 1

            if reward > 0:
                wins += 1
                win_steps += steps
            else:
                losses += 1
                loss_steps += steps

        print(f'{wins} wins (avg. {win_steps} steps) - {losses} losses (avg. {loss_steps} steps)')
    

class BlackjackEnvironment:
    #  - sum of cards in hand (4-21)
    #  - dealers card (1-10)
    #  - usable ace (0-1)
    #  - action (0-1)
    dims = (18, 10, 2, 2)

    def __init__(self):
        self.env = gym.make('Blackjack-v0')

    def map_state(self, state):
        return (state[0] - 4, state[1] - 1, int(state[2]))
  
    def reset(self):
        return self.map_state(self.env.reset())
  
    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)
        return self.map_state(new_state), reward, done, _

    def evaluate_agent(self, agent, num_epochs=1000):
        wins = 0
        draws = 0
        losses = 0

        for i in range(num_epochs):
            state = self.reset()
            done = False

            while not done:
                action = agent.policy_greedy(state)
                new_state, reward, done, _ = self.step(action)

            if reward > 0:
                wins += 1
            elif reward == 0:
                draws += 1
            else:
                losses += 1

        return wins, draws, losses

env = DiceEnvironment()
agent = MCAgent(env, env.dims)