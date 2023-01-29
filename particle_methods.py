# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:41:55 2023

@author: pkw32
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log
from scipy.special import factorial


def pushfwd(particles, f_fwd):
    J = particles.shape[0]
    if particles.ndim == 1:
        return np.array([f_fwd(p) for p in particles])
    else:
        return np.stack([f_fwd(particles[i,...]) for i in range(J)])
    

def weights_from_like(particles, like_fnc):
    # assume first dimension is number of particles
    J = particles.shape[0]
    if particles.ndim == 1:    
        wt = np.array([like_fnc(particles[j]) for j in range(J)])
        wt /= np.sum(wt)
    else:
        wt = np.array([like_fnc(particles[j, ...]) for j in range(J)])
        wt /= np.sum(wt, axis=0)
    return wt


def resample(particles, weights, N=None):
  # we assume number of parameters is first dimension
  if N is None:
    N = particles.shape[0]

  weights_cumul = np.cumsum(weights)/np.sum(weights)
  alpha = np.random.uniform(size=N)
  indices = np.searchsorted(weights_cumul, alpha)
  if particles.ndim > 1:
    return particles[indices,...]
  else:
    return particles[indices] 

def rejuvenate(particles, cov=None):
    # assume first dimension is number of particles
    J = particles.shape[0]
    if particles.ndim == 1:   
        # cov needs to be a number!
        if cov is None:
            # heuristic guess at rejuvenation covariance: 1% of inner population variance. Will not work well with multiple clusters I guess
            cov = np.var(particles) * 0.0001 
        particles += np.random.normal(0, sqrt(cov), J)
    else:
        if cov is None:
            cov = np.cov(particles, rowvar = False)*0.05
        particles += np.random.multivariate_normal(np.zeros(particles.shape[1]), cov, J)
    
    
    return particles 

if __name__ == "__main__":
    particles = np.random.multivariate_normal(np.array([1,2]), np.array([[6,1],[1,1]]), size=100)
    plt.figure()
    plt.scatter(particles[:,0], particles[:,1])
    plt.axis("equal")
    particles2 = rejuvenate(particles)
    plt.scatter(particles2[:,0], particles2[:,1])
    
