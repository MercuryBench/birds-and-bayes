# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:00:03 2023

@author: pkw32
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, copysign
from scipy.special import factorial
from particle_methods import *

def f_step(ns_old, alphas, betas, h):
  n1 = ns_old[0] + copysign(1, alphas[0]*(1-ns_old[0]/5000.0)) * np.random.binomial(ns_old[0], h*abs(alphas[0]*(1-ns_old[0]/5000.0)))  + copysign(1, betas[0]) * np.random.binomial(ns_old[0]*ns_old[1], h*abs(betas[0]))
  n2 = ns_old[1] + copysign(1, alphas[1]*(1-ns_old[1]/5000.0)) * np.random.binomial(ns_old[1], h*abs(alphas[1]*(1-ns_old[1]/5000.0)))  + copysign(1, betas[1]) * np.random.binomial(ns_old[0]*ns_old[1], h*abs(betas[1]))
  return np.array([n1,n2])

def f_step_cont(ns_old, alphas, betas, h):
  n1 = ns_old[0] + h*(alphas[0]*ns_old[0]*(1-ns_old[0]/5000) + betas[0]*ns_old[0]*ns_old[1])
  n2 = ns_old[1] + h*(alphas[1]*ns_old[1]*(1-ns_old[1]/5000) + betas[1]*ns_old[0]*ns_old[1])
  return np.array([n1,n2])

def f_obs(ns, delta):
  return np.array([np.random.poisson(delta*ns[0]), np.random.poisson(delta*ns[1])])

def like_obs(obs, n, delta):
  lam0 = n[0]*delta
  lam1 = n[1]*delta
  return (lam0**obs[0] * exp(-lam0)/factorial(obs[0]))*(lam1**obs[1] * exp(-lam1)/factorial(obs[1]))

#%%
# =========================================
# example data
# =========================================

alphas = np.array([20.0, -30])
betas = np.array([-0.2, 0.1])


n0 = np.array([200,50])
delta = 0.1
N_time = 500
h = 0.001

ns = np.zeros((2, N_time))
ns[:,0] = n0
ns_cont = np.zeros((2, N_time))
ns_cont[:,0] = n0

for m in range(N_time-1):
  ns_cont[:, m+1] = f_step_cont(ns_cont[:,m], alphas, betas, h)
  ns[:, m+1] = f_step(ns[:,m], alphas, betas, h)
  #obs[m+1] = f_obs(ns[m+1], delta)


ind_obs = np.arange(2, N_time, 10, dtype=int)
Nobs = len(ind_obs)
obs = np.zeros((2,Nobs))


for m, m_obs in enumerate(ind_obs):
    obs[:,m] = f_obs(ns[:,m_obs], delta)


fig, [ax0,ax1] = plt.subplots(2,1)
ax0.plot(ns_cont.T)
ax0.set_title("continuous LV model")
ax1.plot(ns.T)
ax1.set_title("binomial LV model");
fig.tight_layout();


fig, [ax0,ax1] = plt.subplots(2,1)
ax0.plot(ns.T)
ax0.set_title("population")
ax1.plot(ind_obs, obs.T, '.-')
ax1.set_title("observations");
fig.tight_layout();


#%%
# =========================================
# start inference
# =========================================

N_part = 500
particles = np.zeros((N_part, 6, N_time)) # order of variables: x1, x2, a1, a2, beta1, beta2
particles0 = np.zeros((N_part, 6))
particles0[:,0] =  np.random.randint(0,1000,N_part)
particles0[:,1] =  np.random.randint(0,1000,N_part)
particles0[:,2] =  np.random.uniform(-50,50,N_part)
particles0[:,3] =  np.random.uniform(-50,50,N_part)
particles0[:,4] =  np.random.uniform(-.5,.5,N_part)
particles0[:,5] =  np.random.uniform(-.5,.5,N_part)



show_iteration_at = np.linspace(0, Nobs, num=10, dtype=int)
show_iteration_at = [0]

ind_obs = np.array([ind_obs[0]])

for m, m_obs in enumerate(ind_obs):
    if m in show_iteration_at:
        flag_plt = True
    else:
        flag_plt = False
    
    if m==0:
        particles_old = particles0
        def f_fwd(var):
            parts = var[0:2]
            alphas = var[2:4]
            betas = var[4:]
            for it in range(ind_obs[0]):
                parts = f_step(parts, alphas, betas, h)
            return np.concatenate((parts, alphas, betas))
    else:
        particles_old = particles[:,:,m-1]
        def f_fwd(var):
            parts = var[0:2]
            alphas = var[2:4]
            betas = var[4:]
            for it in range(ind_obs[m]-ind_obs[m-1]):
                parts = f_step(parts, alphas, betas, h)
            return np.concatenate((parts, alphas, betas))
    
    # push ensemble forward
    particles_now = pushfwd(particles_old, f_fwd)           
    
    
    if flag_plt:
        plt.figure(figsize=(10,10))
        plt.suptitle(f"iteration = {m}")
        plt.subplot(3,3,1)
        plt.scatter(particles_now[:, 0], particles_now[:, 1], s=5)
        plt.plot(ns[0,m], ns[1, m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.title("prior")
        plt.ylim([0,1000])
        plt.xlim([0,1000])
        ax0 = plt.subplot(3,3,2)
        ax0.boxplot(particles_now[:,2:4])
        ax0.plot(np.array([1,2]), alphas)
        ax1 = plt.subplot(3,3,3)
        ax1.boxplot(particles_now[:,4:])
        ax1.plot(np.array([1,2]), betas)
    
        
    # now incorporate observation
    like_fnc = lambda n: like_obs(obs[:,m], n[0:2], delta)
    wt = weights_from_like(particles_now, like_fnc)
    if flag_plt:
        plt.subplot(3,3,4)
        plt.scatter(particles_now[:,0], particles_now[:,1], s=5*wt/np.max(wt));
        plt.plot(ns[0,m], ns[1, m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.ylim([0,1000])
        plt.xlim([0,1000])
        plt.title("weighted by likelihood")
    
    
    particles_now = resample(particles_now, wt)
    if flag_plt:
        plt.subplot(3,3,5)
        plt.scatter(particles_now[:,0], particles_now[:,1], s=5, label="particles")
        plt.plot(ns[0,m], ns[1, m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.ylim([0,1000])
        plt.xlim([0,1000])
        plt.title("resampled")
    
    particles_now = rejuvenate(particles_now)
    # make sure that 0 is not crossed
    particles_now[:,1] = np.maximum(0, particles_now[:,1])
    if flag_plt:
        plt.subplot(3,3,6)
        plt.scatter(particles_now[:,0], particles_now[:,1], s=5, label="particles")
        plt.plot(ns[0,m], ns[1, m], 'r.', label="true value", markersize=15)
        plt.legend()
        plt.title("rejuvenated")
        plt.ylim([0,1000])
        plt.xlim([0,1000])
        
        ax2 = plt.subplot(3,3,7)
        ax2.boxplot(particles_now[:,2:4])
        ax2.plot(np.array([1,2]), alphas)
        ax3 = plt.subplot(3,3,8)
        ax3.boxplot(particles_now[:,4:])
        ax3.plot(np.array([1,2]), betas)
    
    # particles[:,:,m] = particles_now