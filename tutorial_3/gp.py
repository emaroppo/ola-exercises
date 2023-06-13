import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


#function to estimate
def n(x):
    return (1.0-np.exp(-5.0*x))*100.0

def generate_observation(x, noise_std):
    return n(x)+np.random.normal(0, noise_std, size=n(x).shape[0])

n_obs = 50
bids = np.linspace(0.0, 1.0, 20)
x_obs = np.array([])
y_obs = np.array([])

noise_std = 5.0

for i in range(n_obs):
    new_x_obs = np.random.choice(bids, 1)
    new_y_obs = generate_observation(new_x_obs, noise_std)

    x_obs = np.concatenate((x_obs, new_x_obs))

    print(len(x_obs))
    y_obs = np.concatenate((y_obs, new_y_obs))


    #Normalise data
    X = np.atleast_2d(x_obs).T
    Y= y_obs.ravel()

    #set hyperparameters
    theta = 1.0
    l = 1.0

    #specify kernel function
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    #fit GP
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=noise_std**2) # does not work with standardise y set to true?
    gp.fit(X, Y)
    #estimate hyperparameters from data
    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    plt.figure(i)
    plt.plot(x_pred, n(x_pred), 'r:', label = r'$n(x)$')
    plt.plot(X.ravel(), Y, 'ro', label=u'Observations')
    plt.plot(x_pred, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]), np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.5, fc='b', ec='None', label=r'95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$n(x)$')
    plt.legend(loc='upper left')
    plt.show()