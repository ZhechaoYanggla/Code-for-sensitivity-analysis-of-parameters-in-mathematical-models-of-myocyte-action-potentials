
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
from scipy.stats import norm
import scipy.stats as stats

Fo = np.loadtxt('STconfapd90.txt')
print(np.mean(Fo))
#((1 / (np.sqrt(2 * np.pi) * sigma)) *np.exp(-0.5 * (1 / sigma * (bins - mu))**2)






mu = np.mean(Fo)
  # mean of distribution
sigma = Fo.std()  # standard deviation of distribution

xmin, xmax = np.min(Fo), np.max(Fo)
zz = np.linspace(xmin, xmax, 500)
                    

fig, ax = plt.subplots()
plt.axvline(mu,color='red')
plt.errorbar(mu,15, xerr = sigma, capsize=6,capthick=1.2, elinewidth=1.3,fmt='none', color='orangered')
plt.errorbar(mu,4, xerr = 1.96*sigma, capsize=6,capthick=1.2, elinewidth=1.3,fmt='none', color='orangered')
# the histogram of the data
n, bins, patches = ax.hist(Fo, 20, density=True,ec='white', alpha=0.7)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.plot(zz, stats.gaussian_kde(Fo)(zz))
ax.tick_params(axis='x', labelsize=12,pad=2)
ax.tick_params(axis='y', labelsize=12,pad=2)
ax.set_xlabel('Total order index', fontsize =12)
ax.set_ylabel('Probability density', fontsize =12)

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.savefig('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/confidence_interval/BootstrapST.pdf',bbox_inches='tight')
plt.show()