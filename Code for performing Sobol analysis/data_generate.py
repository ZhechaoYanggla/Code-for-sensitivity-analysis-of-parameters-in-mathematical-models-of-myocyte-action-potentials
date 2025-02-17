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



problem = {
    'num_vars': 15,
    'names': ['INaK','IKr','Itos','IK1','INaCa','IClb','ICaL','IKs','ICab','ICap','ICl_Ca','IKp','INa','INab','Itof'],
    'bounds': [[0.0001, 1.07], 
               [0.0001, 3.2],  
               [0.0001, 2.5], 
               [0.03, 1.02],   
               [0.6, 3], 
               [0.0001, 10], 
               [0.7,1.6],  
               [0.0001, 10],
               [0.34,10],
               [0.0001,4.5],
               [0.0001,10],
               [0.0001,10],
               [1,10],
               [0.0001,10],
               [0.0001,4]] 
}



param_values = saltelli.sample(problem, 2**13)

#Y = np.zeros(L)

np.save("15para262144.npy", param_values)



    
    


    
