import math
import myokit
from scipy import integrate
from scipy import optimize

import numpy as np


import matplotlib.pyplot as plt



import numpy as np
from scipy.optimize import fsolve 

fo = np.loadtxt('fo.txt')
sample = np.loadtxt('sample.txt')

fo90 = np.loadtxt('fo90.txt')
sample90 = np.loadtxt('sample90.txt')

fointegral = np.loadtxt('fointegral.txt')
sampleintegral = np.loadtxt('sampleintegral.txt')

foplateau = np.loadtxt('foplateau.txt')
sampleplateau = np.loadtxt('sampleplateau.txt')

fovmax = np.loadtxt('fovmax.txt')
samplevmax = np.loadtxt('samplevmax.txt')

fovrest = np.loadtxt('fovrest.txt')
samplevrest = np.loadtxt('samplevrest.txt')

a=plt.scatter(sample90,fo90)
b=plt.scatter(sample,fo)
c=plt.scatter(sampleintegral,fointegral)
d=plt.scatter(sampleplateau,foplateau)
e=plt.scatter(samplevmax,fovmax)
f=plt.scatter(samplevrest,fovrest)

plt.ylim((0.0,0.5))
plt.xlabel("Sample number")
plt.ylabel("First order index for IClb")
plt.legend((a,b,c,d,e,f),
           ('APD90', 'APD30','Integral','Vplateau','Vmax','Vrest'),
           loc='upper right',
           ncol=3,
           fontsize=8)


plt.savefig('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/convergence.pdf',bbox_inches='tight')
plt.show()