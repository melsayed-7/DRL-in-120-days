#!/usr/bin/python3

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import gym
import collections
import time
from network import DQN

env = gym.make("BeamRider-v0")

# Hyperparameters
num_actions = env.action_space.n
N = int(2e4)
d = 84
num_frames = 4
num_episodes = 100
epsilon = 1.0
epsilon_decay = 0.993
gamma = 0.9

# preprocess the frame


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 110))
    gray = gray[(110//2 - 84//2):(110//2 + 84//2), :]
    return gray

# epsilon greedy


def pick_action(observation, net):
    if(random.random() < epsilon):
        return random.randint(0, num_actions-1)

    action = torch.argmax(
        net(torch.tensor(observation).float().unsqueeze(0)))

    return action


net = DQN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
starttime = time.time()
buffer = collections.deque(maxlen=N)
lr = 1e-3


for i in range(num_episodes):
    observation = env.reset()
    observation = preprocess(observation)
    observation = [observation, observation, observation, observation]
    j = 0

    while(True):
        j += 1
        # time.sleep(0.1 - ((time.time() - starttime) % 0.1))
        if j % 4:
            # env.render()
            action = pick_action(observation, net)
            new_observation, reward, done, info = env.step(action)
            old_observation = observation.copy()
            new_observation = preprocess(new_observation)
            observation.append(new_observation)
            observation.pop(0)

            if len(buffer) > 32:
                batch = random.sample(list(buffer), 32)
                y = [(batch[k][2] + gamma * torch.max(net(torch.tensor(batch[k]
                                                                       [3]).float().unsqueeze(0)))) if k < j else batch[k][2] for k in range(len(batch))]
                y = torch.tensor(y)
                # print(net(torch.tensor(batch[0][0]).float().unsqueeze(0)))
                phi = [torch.max(
                    net(torch.tensor(batch[k][0]).float().unsqueeze(0))) for k in range(len(batch))]
                loss = criterion(y, torch.tensor(phi))

                # the loss must be computed to do the gradient
                loss.requires_grad_()
                net.zero_grad()
                loss.backward()

        # if j % 500:
        buffer.append((old_observation, action, reward, observation, j))

        if done:
            print("Episode {}".format(i))
            # decay the epsilon after each episode
            epsilon *= epsilon_decay
            torch.save(net.state_dict(), "model.h5")
            break
