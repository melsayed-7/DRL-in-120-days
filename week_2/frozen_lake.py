#!/usr/bin/python3

import numpy as np
import gym
import random
# Hyperparameters
num_actions = 4
epsilon = 1.0
EPS_DECAY_RATE = 0.9993
gamma = 0.95
alpha = 0.8


def eps_greedy_action(qtable, state, is_eps_greedy=True):
    max_value, action = best_action(qtable, state)
    if is_eps_greedy:
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
    return action


def best_action(qtable, state):
    for action in range(num_actions):
        if(qtable[state, action] > max_value):
            max_value = qtable[state, action]
            best_action = action

    max_value = 0
    best_action = 0
   return max_value, best_action


def update_qtable(qtable, state, new_state, action, reward):
    max_q, _ = best_action(qtable, state)
    qtable[state, action] += alpha * \
        (reward + gamma * max_q - qtable[state, action])


env = gym.make("FrozenLake-v0")
num_states = env.observation_space.n

qtable = np.zeros((num_states, num_actions), dtype=np.float64)

for i in range(1000):
    state = env.reset()
    reward_sum = 0
    for j in range(100):
        action = eps_greedy_action(qtable, state)
        new_state, reward, done, info = env.step(action)
        reward_sum += reward
        update_qtable(qtable, state, new_state, action, reward)
        state = new_state
        if done:
            epsilon *= EPS_DECAY_RATE
            print(i, reward, state)
            break
