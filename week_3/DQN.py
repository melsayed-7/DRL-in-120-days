#!/usr/bin/python3

import numpy as np
import random
import torch
import cv2
import gym
import collections
import time

env = gym.make("BeamRider-v0")

# Hyperparameters
num_actions = env.action_space.n
N = int(2e4)
d = 84
num_frames = 4
num_episodes = 100
epsilon = 1.0
epsilon_decay = 0.993

buffer = collections.deque(maxlen=N)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 110))
    gray = gray[(110//2 - 84//2):(110//2 + 84//2), :]
    return gray


def pick_action(observation):
    if(random.random() < epsilon):
        return random.randint(0, num_actions-1)
    # else:
    #     return DQN(observation)


starttime = time.time()
for i in range(num_episodes):
    # time.sleep(0.1 - ((time.time() - starttime) % 0.1))
    observation = env.reset()
    observation = preprocess(observation)
    while(True):
        env.render()
        action = pick_action(observation)
        new_observation, reward, done, info = env.step(action)
        new_observation = preprocess(new_observation)
        buffer.append((observation, action, reward, new_observation))
        if done:
            break
