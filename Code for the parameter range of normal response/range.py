from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import math
import myokit
from scipy import integrate
import pints

import matplotlib
matplotlib.use('Agg')


from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plt 
plt.ioff()




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


# Define the model inputs


interval = np.linspace(0.0001,10,500)
times = np.linspace(0, 50000, 500000, endpoint=False)

def evaluate(a):  
    model, protocol, script = myokit.load('shannon.mmt')
    model.set_value('INaK.p', a)
    sim = myokit.Simulation(model, protocol)
    sim.reset()
    sim.pre(99 * 500)
    #d = sim.run(30000)
    d = sim.run(500.0,log_times=times)
    var = 'cell.V'
    
    return d.time(), d[var]




#plot APD90
apd90 = np.zeros(500)
 
for i in range(500):  
    t,v = evaluate(interval[i])
    apd90[i] = compute_apd90(v, t)
    
plt.scatter(interval,apd90)
plt.axvline(x=0.0001,linestyle='--', color='red')
plt.axvline(x=1.07,linestyle='--', color='red')
plt.axvline(x=2.1,linestyle='--', color='red')
plt.ylim((0,550))
plt.xlabel('P$_{INaK}$',fontsize=12)
plt.ylabel('A$_{90}$ [ms]',fontsize=12)





