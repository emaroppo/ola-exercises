import numpy as np


#define function to estimate
def n(x):
    return (1.0-np.exp(-5.0*x))*100.0

class BiddingEnvironment:
    def __init__(self, bids, sigma):
        self.bids = bids
        self.means = n(bids)
        self.sigma = sigma
    
    def round(self, pulled_arm):
        return np.random.normal(n(pulled_arm), self.sigma)
    
