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



#param_values = saltelli.sample(problem, 2**10)

Y = np.zeros(262144)
Y1 = np.load('apd30_Y1_10000.npy', mmap_mode='r+')

Y2 = np.load('apd30_Y10001_20000.npy', mmap_mode='r+')


Y3 = np.load('apd30_Y20001_30000.npy', mmap_mode='r+')

Y4 = np.load('apd30_Y30001_40000.npy', mmap_mode='r+')


Y5 = np.load('apd30_Y40001_50000.npy', mmap_mode='r+')


Y6 = np.load('apd30_Y50001_60000.npy', mmap_mode='r+')

Y7 = np.load('apd30_Y60001_70000.npy', mmap_mode='r+')

Y8 = np.load('apd30_Y70001_80000.npy', mmap_mode='r+')


Y9 = np.load('apd30_Y80001_90000.npy', mmap_mode='r+')

Y10 = np.load('apd30_Y90001_100000.npy', mmap_mode='r+')

Y11 = np.load('apd30_Y100001_110000.npy', mmap_mode='r+')


Y12 = np.load('apd30_Y110001_120000.npy', mmap_mode='r+')



Y13 = np.load('apd30_Y120001_130000.npy', mmap_mode='r+')


Y14 = np.load('apd30_Y130001_140000.npy', mmap_mode='r+')

Y15 = np.load('apd30_Y140001_150000.npy', mmap_mode='r+')

Y16 = np.load('apd30_Y150001_160000.npy', mmap_mode='r+')

Y17 = np.load('apd30_Y160001_170000.npy', mmap_mode='r+')

Y18 = np.load('apd30_Y170001_180000.npy', mmap_mode='r+')

Y19 = np.load('apd30_Y180001_190000.npy', mmap_mode='r+')

Y20 = np.load('apd30_Y190001_200000.npy', mmap_mode='r+')

Y21 = np.load('apd30_Y200001_210000.npy', mmap_mode='r+')

Y22 = np.load('apd30_Y210001_220000.npy', mmap_mode='r+')

Y23 = np.load('apd30_Y220001_230000.npy', mmap_mode='r+')

Y24 = np.load('apd30_Y230001_240000.npy', mmap_mode='r+')

Y25 = np.load('apd30_Y240001_250000.npy', mmap_mode='r+')

Y26 = np.load('apd30_Y250001_260000.npy', mmap_mode='r+')

Y27 = np.load('apd30_Y260001_262144.npy', mmap_mode='r+')



Y[0:10000] = Y1[0:10000]
Y[10000:20000] = Y2[10000:20000]
Y[20000:30000] = Y3[20000:30000]
Y[30000:40000] = Y4[30000:40000]
Y[40000:50000] = Y5[40000:50000]
Y[50000:60000] = Y6[50000:60000]
Y[60000:70000] = Y7[60000:70000]
Y[70000:80000] = Y8[70000:80000]
Y[80000:90000] = Y9[80000:90000]
Y[90000:100000] = Y10[90000:100000]
Y[100000:110000] = Y11[100000:110000]
Y[110000:120000] = Y12[110000:120000]
Y[120000:130000] = Y13[120000:130000]
Y[130000:140000] = Y14[130000:140000]
Y[140000:150000] = Y15[140000:150000]   
Y[150000:160000] = Y16[150000:160000] 
Y[160000:170000] = Y17[160000:170000]
Y[170000:180000] = Y18[170000:180000]
Y[180000:190000] = Y19[180000:190000]
Y[190000:200000] = Y20[190000:200000]
Y[200000:210000] = Y21[200000:210000]
Y[210000:220000] = Y22[210000:220000]
Y[220000:230000] = Y23[220000:230000]
Y[230000:240000] = Y24[230000:240000]
Y[240000:250000] = Y25[240000:250000]
Y[250000:260000] = Y26[250000:260000]
Y[260000:262144] = Y27[260000:262144]



print(Y[19999])

print(Y2[19999])

# Perform analysis 
Si = sobol.analyze(problem, Y, print_to_console=True)
Si_df = Si.to_df()

plt.figure()
barplot(Si_df[1])
plt.title("First order for apd30")
plt.savefig("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/1stapd30.png")

plt.figure()
barplot(Si_df[2])
plt.title("Second order for apd30")
#plt.savefig("/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/2ndvrest.png")

plt.show()


