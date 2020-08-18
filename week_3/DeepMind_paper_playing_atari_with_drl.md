## my notes on this Deep Mind [paper](https://arxiv.org/pdf/1312.5602.pdf)

### Target

- Using only visual input and deep learning capabilities we need to teach an agent to play atari games.

### Probelms with Deep learning

- the reward is not instant like normal supervised learning tasks that has instant penalty.
- reward function is frequently sparse and the sampling is not independent at all, it is actually highly correlated.
- not only that, the reward function can be noisy and highly delayed.

### To overcome those problems

- They use variant of Q-learning algorithm with stochastic gradient descent
- They solve the sampling problem and non-stationary distributions by the replay buffer
-

## The network

- The networks plays the role of a function-approximator

This is a model-free because the agent knows nothing about the dynamics of the environment.

It is also off-policy: learning greedy strategy $a= max_a Q(s,a; \theta)$

And the differentiation of that is

$\nabla_{\theta_{i}} L_{i}\left(\theta_{i}\right)=\mathbb{E}_{s, a \sim \rho(\cdot) ; s^{\prime} \sim \mathcal{E}}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta_{i-1}\right)-Q\left(s, a ; \theta_{i}\right)\right) \nabla_{\theta_{i}} Q\left(s, a ; \theta_{i}\right)\right]$
