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
    model, protocol, script = myokit.load('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/tests/data/shannonRado.mmt')
    model.set_value('INaK.p', a)
    sim = myokit.Simulation(model, protocol)
    sim.reset()
    sim.pre(99 * 500)
    #d = sim.run(30000)
    d = sim.run(500.0,log_times=times)
    var = 'cell.V'
    
    return d.time(), d[var]


#t1,v1 = evaluate(interval[0])
#t2,v2 = evaluate(interval[1])
#t3,v3 = evaluate(interval[2])
#t4,v4 = evaluate(interval[3])
#t5,v5 = evaluate(interval[4])
#t6,v6 = evaluate(interval[5])
#t7,v7 = evaluate(interval[6])
#t8,v8 = evaluate(interval[7])
#t9,v9 = evaluate(interval[8])
#t10,v10 = evaluate(interval[9])
t,v= evaluate(1.5)
plt.plot(t,v)
plt.tick_params(axis='x', labelsize=12,pad=2)
plt.tick_params(axis='y', labelsize=12,pad=2)
plt.xlabel('t [ms]',fontsize=12)
plt.ylabel('V [mV]',fontsize=12)

plt.savefig('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/Range_refine/intermittent.pdf',bbox_inches='tight')

#plt.plot(t,v)
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
#np.savetxt('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/detailed/Clo_p_0.0001_10.txt',apd90)
#plt.plot(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)
#plt.xlabel("Time/(ms)")
#plt.ylabel("Voltage/(mV)")
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(t,h,'r--',label='Itos_Y_gate')
#ax1.plot(t,m,'b-.',label='Itos_R_gate')
#ax1.plot(t,j,'k:',label='j')
#ax1.set_xlabel('Time/(ms)')
#ax1.set_ylabel('Itos_Y_gate,Itos_R_gate')
#ax1.legend(loc='upper right')
#ax2 = ax1.twinx()
#ax2.plot(t,v,'g-',label='Voltage')
#ax2.set_ylabel('Voltage/(mV)')
#ax2.legend(loc='center right')
#plt.savefig('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/plot/15para262144/detailed/Clo.pdf',bbox_inches='tight')
#plt.savefig('/home/pgrad1/2712549y/.local/lib/python3.6/site-packages/myokit/Range_refine/INaK_two_lines.pdf',bbox_inches='tight')
#plt.show()




