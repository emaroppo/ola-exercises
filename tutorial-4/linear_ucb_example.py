import numpy as np
import matplotlib.pyplot as plt
from linear_ucb import LinearMABEnvironment, LinearUCB

n_arms=100
T=1000
n_experiments=100
lin_ucb_rewards_per_experiment=list()

env=LinearMABEnvironment(n_arms=n_arms, dim=10)

for e in range(n_experiments):
    lin_ucb_learner = LinearUCB(arms_features=env.arms_features)
    for t in range(T):
        pulled_arm=lin_ucb_learner.pull_arm()
        reward=env.round(pulled_arm)
        lin_ucb_learner.update(pulled_arm, reward)

opt = env.opt()
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - np.array(lin_ucb_rewards_per_experiment), axis=0)), 'r')
plt.legend(["LinUCB"])
plt.show()