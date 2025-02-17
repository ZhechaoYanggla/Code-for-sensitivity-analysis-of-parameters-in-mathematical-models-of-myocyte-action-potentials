#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:08:43 2023

@author: hgao
"""

import numpy as np
import math
import myokit
from scipy import integrate

from SALib.sample import saltelli
from SALib.analyze import sobol

import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot

times = np.linspace(0, 500, 5000, endpoint=False)



param_values = np.load('15para262144.npy', mmap_mode='r')
#param_values = saltelli.sample(problem, 2**10)
L = param_values.shape[0]
Yapd30 = np.zeros(L)
Yvmax = np.zeros(L)
Yintegral = np.zeros(L)
Yplateau = np.zeros(L)
Yvrest = np.zeros(L)
Yapd90 = np.zeros(L)

model, protocol, script = myokit.load('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/tests/data/shannonRado.mmt')

def compute_apd90(voltage, time):
    # Find the peak voltage
    peak_voltage = np.max(voltage)
    
    
    # Find the time at which the voltage returns to 90% of the resting potential
    resting_voltage = np.min(voltage)
    resting_threshold_voltage = resting_voltage + 0.1 * (peak_voltage - resting_voltage)
    indices = np.where(voltage >= resting_threshold_voltage)[0]
    
    if len(indices) == 0:
        return None  # APD90 couldn't be determined
    
    apd90_index = indices[0]
    apd90_time = time[apd90_index]
    
    apd90_end_index = indices[-1]
    apd90_end_time = time[apd90_end_index]
    
    # Calculate the APD90
    apd90 = apd90_end_time - apd90_time
    
    return apd90


def compute_apd30(voltage, time):
    # Find the peak voltage
    peak_voltage = np.max(voltage)
    
    
    # Find the time at which the voltage returns to 30% of the resting potential
    resting_voltage = np.min(voltage)
    resting_threshold_voltage = resting_voltage + 0.7 * (peak_voltage - resting_voltage)
    indices = np.where(voltage >= resting_threshold_voltage)[0]
    
    if len(indices) == 0:
        return None  # APD30 couldn't be determined
    
    apd30_index = indices[0]
    apd30_time = time[apd30_index]
    
    apd30_end_index = indices[-1]
    apd30_end_time = time[apd30_end_index]
    
    # Calculate the APD30
    apd30 = apd30_end_time - apd30_time
    
    return apd30


def compute_plateau(voltage, time):
    # Find the peak voltage
    peak_voltage = np.max(voltage)
    
    
    # Find the time at which the voltage returns to 90% of the resting potential
    resting_voltage = np.min(voltage)
    resting_threshold_voltage = resting_voltage + 0.1 * (peak_voltage - resting_voltage)
    indices = np.where(voltage >= resting_threshold_voltage)[0]
    
    if len(indices) == 0:
        return None  # APD90 couldn't be determined
    
    apd90_index = indices[0]
    apd90_time = time[apd90_index]
    
    apd90_end_index = indices[-1]
    apd90_end_time = time[apd90_end_index]
    
    # Calculate the plateau
    plateau = 76+ (apd90_end_time - apd90_time)/2
    
    indices1 = np.where(time >= plateau)[0]
    plateau_index = indices1[0]
    Vplateau = voltage[plateau_index]
    
    
    return Vplateau
    


index_run = 260001
for params in param_values[index_run-1:262144,:]:
    a,b,c,d,e,f,g,h,j,k,m,n,o,p,q = params
    
    model.set_value('INaK.p', a)
    model.set_value('IKr.p', b)
    model.set_value('Itos.p',c)
    model.set_value('IK1.p',d)
    model.set_value('INaCa.p',e)
    model.set_value('IClb.p',f)
    model.set_value('ICaL.p',g)
    model.set_value('IKs.p',h)
    
    model.set_value('ICab.p',j)
    model.set_value('ICap.p',k)
    model.set_value('ICl_Ca.p',m)
    model.set_value('IKp.p',n)
    model.set_value('INa.p',o)
    model.set_value('INab.p',p)
    model.set_value('Itof.p',q)
    
    sim = myokit.Simulation(model, protocol)
    sim.reset()
    sim.pre(99 * 500)
    d = sim.run(500.0,log_times=times)
    var = 'cell.V'
    Yapd30[index_run-1] = compute_apd30(d[var], d.time())
    Yapd90[index_run-1] = compute_apd90(d[var], d.time())
    Yvmax[index_run-1] = max(d[var])
    Yintegral[index_run-1] = integrate.simps(np.abs(d[var]),d.time())
    Yplateau[index_run-1] = compute_plateau(d[var], times)
    Yvrest[index_run-1] = (sum(d[var][4500:5000])/500)
    
    
    index_run = index_run + 1
    
    print(index_run)



np.save("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/apd30_Y260001_262144.npy",Yapd30)
np.save("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/apd90_Y260001_262144.npy",Yapd90)
np.save("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/vmax_Y260001_262144.npy",Yvmax)
np.save("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/integral_Y260001_262144.npy",Yintegral)
np.save("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/plateau_Y260001_262144.npy",Yplateau)
np.save("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/vrest_Y260001_262144.npy",Yvrest)

    
    


    
