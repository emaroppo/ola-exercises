import sys

sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from tutorial_2.learner.ucb_learner import UCBMatching
from tutorial_1.environment.environment import Environment


p = np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
opt = linear_sum_assignment(-p)
n_experiments = 10
T= 3000
regret_ucb_matching = np.zeros((n_experiments, T))

for e in tqdm(range(n_experiments)):
    learner = UCBMatching(p.size, *p.shape)
    rewards_per_experiment = []
    opt_rewards= []
    env = Environment(p.size, p)
    for t in range(T):
        pulled_arms = learner.pull_arm()
        reward = env.round(pulled_arms)
        learner.update(pulled_arms, reward)
        rewards_per_experiment.append(reward.sum())
        opt_rewards.append(p[opt].sum())

    regret_ucb_matching[e, :] = np.cumsum(opt_rewards) - np.cumsum(rewards_per_experiment)

plt.figure(0)
plt.plot(np.mean(regret_ucb_matching, axis=0), 'r')
plt.ylabel('Regret')
plt.xlabel('t')
plt.show()