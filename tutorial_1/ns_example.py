import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from environment.ns_environment import NonStationaryEnvironment
from learner.ts_learner import TSLearner
from learner.swts_learner import SWTS_Learner


n_arms = 4
n_phases = 4
p = np.array([[0.15, 0.1, 0.1, 0.35],
                [0.45, 0.21, 0.2, 0.35],
                [0.1, 0.1, 0.5, 0.15],
                [0.1, 0.21, 0.1, 0.15]])

T = 500
phases_len = int(T / n_phases)
n_experiments = 100
ts_rewards_per_experiment = []
swts_rewards_per_experiment = []
window_size = int(np.sqrt(T))

for e in tqdm(range(0, n_experiments)):
    ts_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p, horizon=T)
    swts_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p, horizon=T)

    ts_learner = TSLearner(n_arms=n_arms)
    swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size)

    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        # SW-TS Learner
        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)
    
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    swts_rewards_per_experiment.append(swts_learner.collected_rewards)

ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)
opt_per_phases = p.max(axis=1)
opt_per_round = np.zeros(T)

for i in range(n_phases):
    t_index = np.arange(i * phases_len, (i + 1) * phases_len)
    opt_per_round[t_index] = opt_per_phases[i]
    ts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiment, axis=0)[t_index]
    swts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(swts_rewards_per_experiment, axis=0)[t_index]

plt.figure(0)
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'g')
plt.plot(opt_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])
plt.ylabel("Reward")
plt.xlabel("t")
plt.show()

plt.figure(1)
plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret), 'g')
plt.legend(["TS", "SW-TS"])
plt.ylabel("Regret")
plt.xlabel("t")
plt.show()
