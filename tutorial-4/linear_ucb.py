import numpy as np

class LinearMABEnvironment:
    def __init__(self, n_arms, dim) -> None:
        self.theta = np.random.dirichlet(np.ones(dim), size=1)
        #initialize arms features matrix (rows are arms, columns are features)        
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms, dim))
        self.p = np.zeros(n_arms)
        for i in range(n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i,:])
    
    def round(self, pulled_arm):
        return 1 if np.random.rand() < self.p[pulled_arm] else 0
    
    
    def opt(self):
        return np.max(self.p)
    

class LinearUCB:
    def __init__(self, arms_features) -> None:
        self.arms=arms_features
        self.dim=arms_features.shape[1]
        self.collected_rewards=list()
        self.pulled_arms=[]
        self.c=2.0
        self.M=np.identity(self.dim)
        self.b=np.atleast_2d(np.zeros(self.dim)).T
        self.theta=np.dot(np.linalg.inv(self.M), self.b)
    def compute_ucbs(self):
        self.theta=np.dot(np.linalg.inv(self.M), self.b)
        ucbs=list()
        for arm in self.arms:
            arm=np.atleast_2d(arm).T
            ucb=np.dot(self.theta.T, arm) + self.c*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
            ucbs.append(ucb[0][0])
        return ucbs
    
    def pull_arm(self):
        ucbs=self.compute_ucbs()
        pulled_arm=np.argmax(ucbs)
        return pulled_arm
    
    def update_estimation(self, pulled_arm_idx, reward):
        arm=np.atleast_2d(self.arms[pulled_arm_idx]).T
        self.M+=np.dot(arm, arm.T)
        self.b+=reward*arm
    
    def update(self, pulled_arm_idx, reward):
        self.pulled_arms.append(pulled_arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(pulled_arm_idx, reward)