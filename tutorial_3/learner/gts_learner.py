import sys
sys.path.append('.')
from tutorial_1.learner.learner import Learner
import numpy as np

class GTSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms)*1e+3
    
    def pull_arms(self):
        idx = np.argmax(np.random.normal(self.means, self.sigmas))
        return idx
    def update(self, pulled_arm, reward):
        self.t+=1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = len(self.rewards_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm])/(n_samples**0.5)



