import numpy as np
import math
import myokit
from scipy import integrate
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot
import argparse

def generate_parameters():
    """Generate parameter samples."""
    problem = {
        'num_vars': 15,
        'names': ['INaK','IKr','Itos','IK1','INaCa','IClb','ICaL','IKs','ICab','ICap','ICl_Ca','IKp','INa','INab','Itof'],
        'bounds': [[0.0001, 1.07], [0.0001, 3.2], [0.0001, 2.5], [0.03, 1.02], [0.6, 3], [0.0001, 10], [0.7,1.6],
                   [0.0001, 10], [0.34,10], [0.0001,4.5], [0.0001,10], [0.0001,10], [1,10], [0.0001,10], [0.0001,4]] 
    }
    param_values = saltelli.sample(problem, 2**13)
    np.save("15para262144.npy", param_values)
    print("Parameter generation complete.")


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


def run_simulations():
    """Run the simulations and extract AP features."""
    times = np.linspace(0, 500, 5000, endpoint=False)
    param_values = np.load('15para262144.npy', mmap_mode='r')
    L = param_values.shape[0]
    Yapd30, Yapd90, Yvmax, Yintegral, Yplateau, Yvrest = (np.zeros(L) for _ in range(6))
    
    model, protocol, _ = myokit.load('shannonRado.mmt')
    


    index_run = 260001
    for params in param_values[index_run-1:262144,:]:
        a,b,c,d,e,f,g,h,j,k,m,n,o,p,q = params
        model.set_value('INaK.p', a)
        model.set_value('IKr.p', b)
        model.set_value('Itos.p', c)
        model.set_value('IK1.p', d)
        model.set_value('INaCa.p', e)
        model.set_value('IClb.p', f)
        model.set_value('ICaL.p', g)
        model.set_value('IKs.p', h)
        
        sim = myokit.Simulation(model, protocol)
        sim.reset()
        sim.pre(99 * 500)
        d = sim.run(500.0, log_times=times)
        var = 'cell.V'
        Yapd90[index_run-1] = compute_apd90(d[var], d.time())
        Yapd30[index_run-1] = compute_apd30(d[var], d.time())
        
        Yvmax[index_run-1] = max(d[var])
        k=np.array(list(d[var]))-np.min(d[var])
        Yintegralnew[index_run-1] = integrate.simps(k,d.time())
        Yplateau[index_run-1] = compute_plateau(d[var], times)
        Yvrest[index_run-1] = (sum(d[var][250:750])/500)
        index_run += 1
    
    np.save("apd90_results.npy", Yapd90)
    print("Simulation complete.")

def analyze_sensitivity():
    """Perform sensitivity analysis."""
    problem = {
        'num_vars': 15,
        'names': ['INaK','IKr','Itos','IK1','INaCa','IClb','ICaL','IKs','ICab','ICap','ICl_Ca','IKp','INa','INab','Itof'],
        'bounds': [[0.0001, 1.07], [0.0001, 3.2], [0.0001, 2.5], [0.03, 1.02], [0.6, 3], [0.0001, 10], [0.7,1.6],
                   [0.0001, 10], [0.34,10], [0.0001,4.5], [0.0001,10], [0.0001,10], [1,10], [0.0001,10], [0.0001,4]] 
    }
    Y = np.load("apd90_results.npy", mmap_mode='r')
    Si = sobol.analyze(problem, Y, print_to_console=True)
    Si_df = Si.to_df()
    plt.figure()
    barplot(Si_df[1])
    plt.title("First order for APD90")
    plt.savefig("sensitivity_analysis.png")
    plt.show()
    print("Sensitivity analysis complete.")

def main():
    """Main function to handle command-line options."""
    parser = argparse.ArgumentParser(description="Run the full pipeline or specific steps.")
    parser.add_argument("--generate", action="store_true", help="Run parameter generation")
    parser.add_argument("--simulate", action="store_true", help="Run simulations")
    parser.add_argument("--analyze", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if args.all:
        generate_parameters()
        run_simulations()
        analyze_sensitivity()
    else:
        if args.generate:
            generate_parameters()
        if args.simulate:
            run_simulations()
        if args.analyze:
            analyze_sensitivity()
    
if __name__ == "__main__":
    main()
