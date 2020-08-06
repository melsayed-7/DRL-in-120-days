#!/usr/bin/python3

import gym
import numpy as np


env = gym.make('CartPole-v1')

def play(env, policy):
    observation = env.reset()
    done = False
    score = 0
    observations = []
    
    for _ in range(5000):
        observations += [observation.tolist()]
        
        if done:
            break

        outcome = np.dot(policy, observation)
        action = 1 if outcome > 0 else 0
        
        observation, reward, done, info = env.step(action)
        score += reward

    return score, observations


for _ in range(10):
    policy = np.random.rand(1,4)
    score, observations = play(env, policy)
    print('policy score', score)





