"""
    Training an agent with (stochastic) Policy Gradients on Pong game using OpenAI Gym.
    This file is me trying to bunch of stuff from Andrej's small code (https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
"""

import numpy as np
import pickle
import gym


# hyperparameters

H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = False
render = False

D = 80*80 # input dimenstionality

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.random((H, D)) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.random(H) / np.sqrt(H)

grad_buffer = {k : np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k : np.zeros_like(v) for k,v in model.items()}


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def preprocess(I):
    I = I[35:195] # crop
    I = I[::2, ::2, 0]
    I[I == 144] = 0   # erase background
    I[I == 109] = 0
    I[I != 0] = 1 # other than that set it to 1
    
    return I.astype(np.float).ravel()

def discount_rewards(r):
    running_add = 0
    discounted_r = np.zeros_like(r)

    for t in reversed(range(r.size)):
        if r[t] != 0: running_add = 0
        running_add += running_add*gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 #ReLU
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)

    return p, h

def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}
h

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None

xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)
    hs.append(h)

    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        xs,hs,dlogps,drs = [],[],[],[]

        discounted_epr = discount_rewards(epr)

        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)

        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) ,('' if reward == -1 else ' !!!!!!!!')






