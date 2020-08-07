#!/usr/bin/python3

import gym
import numpy as np


env = gym.make('CartPole-v1')

def play(env, policy, is_render=False):
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
        if(is_render==True):
            env.render()
        observation, reward, done, info = env.step(action)
        score += reward
    env.close()
    return score, observations


max_score = 0
for _ in range(100):
    policy = np.random.rand(1,4)
    score, observations = play(env, policy)
    if(score > max_score):
        max_score_policy = (policy, score)
    print('policy score', score)

policy = np.random.rand(1,4)
score, observations = play(env, max_score_policy[0], True)
print('policy score', score)



#from flask import Flask
#import json 
#
#app = Flask(__name__, static_folder='.')
#@app.route("/data")
#def data():
#    return json.dump(observations)
#
#@app.route('/')
#def root():
#    return app.send_static_file('./index.html')
#
#
#app.run(host='0.0.0.0', port='3000')
#
















