# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:14:04 2023

@author: pkw32
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log
from scipy.special import factorial
from particle_methods import *

np.random.seed(5)

def f_step(n_old, a):
  # returns Euler discretization of exponential dynamics
  # n(t+1) = Poiss(((1+a)*n(t))
  return np.random.poisson((1+a)*n_old)

def like_step(n_new, n_old, a):
  lam = n_old*(1+a)
  return lam**n_new * exp(-lam)/factorial(n_new)

def f_obs(n, delta):
  return np.random.poisson(delta*n)

def like_obs(obs, n, delta):
  lam = n*delta
  return lam**obs * exp(-lam)/factorial(obs)

def log_gen_poisson(lam, N_max):
  ks = np.arange(1,N_max)
  if lam < 1e-7:
    logterm = [-100 for j in range(N_max-1)]
  else:
    logterm = ks*log(lam) - lam - np.cumsum(np.log(ks))
  logterm_full = np.concatenate(([-lam], logterm))
  return (logterm_full)

def log_gen_poisson2(k, N_max):
  ks = np.arange(1,N_max)
  if k == 0:
    element0 = 0
  else: 
    element0 = -100
  return np.concatenate(([element0], k * np.log(ks*delta) - ks*delta - np.sum(np.log(np.arange(1,k)))))
  #return (ks*delta)**k * np.exp(-ks*delta)/factorial(k)

def resample(xs, weights, N=None):
  # we assume number of parameters is first dimension
  if N is None:
    N = xs.shape[0]
  weights_cumul = np.cumsum(weights)/np.sum(weights)
  alpha = np.random.uniform(size=N)
  indices = np.searchsorted(weights_cumul, alpha)
  if xs.ndim > 1:
    return xs[indices,...]
  else:
    return xs[indices]

#%%
# =========================================
# example data
# =========================================

a = 0.1
n0 = 10
delta = 0.01
N_time = 50

ns = np.zeros(N_time)
ns[0] = n0

obs = np.zeros(N_time)
obs[0] = f_obs(n0, delta)

for m in range(N_time-1):
  ns[m+1] = f_step(ns[m], a)
  obs[m+1] = f_obs(ns[m+1], delta)

fig, [ax0,ax1] = plt.subplots(2,1)
ax0.plot(ns)
ax0.set_title("population")
ax1.plot(obs);
ax1.set_title("observation");
fig.tight_layout();

# =========================================
# start inference
# =========================================

N_part = 500
particles = np.zeros((N_part, 2, N_time)) # first col is a, second col is n
particles0 = np.zeros((N_part, 2))
particles0[:,0] =  np.random.uniform(-.5,.5,N_part)
particles0[:,1] =  np.random.randint(0,1001,N_part)


f_fwd = lambda var: np.array([var[0], f_step(var[1], var[0])])


show_iteration_at = np.linspace(0, N_time, num=10, dtype=int)

for m in range(N_time):
    if m in show_iteration_at:
        flag_plt = True
    else:
        flag_plt = False
    
    if m > 0: # no push forward needed for first observation
        particles_old = particles[:,:,m-1]
        particles_now = pushfwd(particles_old, f_fwd)
        if flag_plt:
            plt.figure(figsize=(10,10))
            plt.suptitle(f"iteration = {m}")
            plt.subplot(2,2,1)
            plt.scatter(particles_now[:, 0], particles_now[:, 1], s=5)
            plt.plot(a, ns[m], 'r.', label="true value", markersize=15)
            plt.legend()
            plt.title("prior")
            plt.ylim([0,1001])
            plt.xlim([-.5,.5])
    else: 
        particles_now = particles0
        if flag_plt:
            plt.figure(figsize=(10,10))
            plt.suptitle(f"iteration = {m}")
            plt.subplot(2,2,1)
            plt.scatter(particles0[:, 0], particles0[:, 1], s=5)
            plt.plot(a, ns[m], 'r.', label="true value", markersize=15)
            plt.legend()
            plt.title("pushed fwd")
            plt.ylim([0,1001])
            plt.xlim([-.5,.5])
        
    # now incorporate observation
    like_fnc = lambda n: like_obs(obs[m], n[1], delta)
    wt = weights_from_like(particles_now, like_fnc)
    if flag_plt:
        plt.subplot(2,2,2)
        plt.scatter(particles_now[:,0], particles_now[:,1], s=5*wt/np.max(wt));
        plt.plot(a, ns[m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.ylim([0,1001])
        plt.xlim([-.5,.5])
        plt.title("weighted by likelihood")
    
    
    particles_now = resample(particles_now, wt)
    if flag_plt:
        plt.subplot(2,2,3)
        plt.scatter(particles_now[:,0], particles_now[:,1], s=5, label="particles")
        plt.plot(a, ns[m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.ylim([0,1001])
        plt.xlim([-.5,.5])
        plt.title("resampled")
    
    particles_now = rejuvenate(particles_now)
    # make sure that 0 is not crossed
    particles_now[:,1] = np.maximum(0, particles_now[:,1])
    if flag_plt:
        plt.subplot(2,2,4)
        plt.scatter(particles_now[:,0], particles_now[:,1], s=5, label="particles")
        plt.plot(a, ns[m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.title("rejuvenated")
        plt.ylim([0,1001])
        plt.xlim([-.5,.5])
    
    particles[:,:,m] = particles_now

fig, ax0 = plt.subplots(figsize=(10,5))
ax0.boxplot(particles[:,1,:]);
ax0.plot(ns, label="population")
plt.legend()
plt.tight_layout()

fig, ax0 = plt.subplots(figsize=(10,5))
ax0.boxplot(particles[:,0,:]);
ax0.hlines(a, 1, 50, color="blue", label="true param")
plt.legend()
plt.tight_layout()

       
    




