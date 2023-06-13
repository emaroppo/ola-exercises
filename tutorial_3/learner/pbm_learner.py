import sys
sys.path.append('.')
from tutorial_1.learner.learner import Learner
import numpy as np

class PBMUCBLearner(Learner):

    def __init__(self, n_arms, n_positions, position_probabilities, delta):
        super().__init__(n_arms)
        self.position_probabilities = position_probabilities
        self.n_arms = n_arms
        self.n_positions = n_positions

        assert n_positions == len(position_probabilities)
        self.S_kl = np.zeros((n_arms, n_positions)) #number of clicks for ad k in position l
        self.S_k = np.zeros(n_arms) #number of clicks for ad k
        self.N_kl = np.zeros((n_arms, n_positions)) #number of times ad k is shown in position l
        self.N_k = np.zeros(n_arms) #number of times ad k is shown
        self.N_kl_adj=np.zeros((n_arms, n_positions)) #number of times ad k is shown in position l adjusted for the confidence interval
        self.N_k_adj=np.zeros(n_arms) #number of times ad k is shown adjusted for the confidence interval

        self.delta = delta
        self.empirical_means = np.zeros((n_arms, n_positions))
        self.confidence = np.array([np.inf]*n_arms)
    
    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.argsort(upper_conf)[::-1][:self.n_positions]
    
    def update(self, super_arm, reward):
        self.t+=1
        for pos, arm in enumerate(super_arm):
            self.S_kl[arm][pos] += reward[pos]
            self.S_k[arm] += reward[pos]
            self.N_kl[arm][pos] += 1
            self.N_kl_adj[arm][pos] +=  self.position_probabilities[pos]
        
        self.S_k = self.S_kl.sum(axis=1)
        self.N_k = self.N_kl.sum(axis=1)
        self.N_k_adj = self.N_kl_adj.sum(axis=1)

        self.empirical_means = self.S_k/self.N_k_adj
        self.confidence = np.sqrt(self.N_k/self.N_k_adj)*np.sqrt(self.delta/(2*self.N_k_adj))
        self.empirical_means[self.N_k==0]= np.inf
        self.confidence[self.N_k==0]= np.inf
        self.update_observations(super_arm, reward)
    
    def update_observations(self, pulled_arm, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward.sum())
