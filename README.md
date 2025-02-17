# Code for sensitivity analysis of parameters in mathematical models of myocyte action potentials
This is a suite of Python codes for sensitivity analysis of parameters in mathematical models of myocyte action potentials.

The codes perform generation of the sampling, sensitivity analysis of parameters in a mathematical model of action potential, and check consistency and bias of Sobol's index estimators. 

The mathematical model used is [doi.org/10.1529/biophysj.104.047449]. 

The bias of Sobol's index estimators is based on bootstrapping principle.

The code dependencies include standard Python libraries and the Myokit [myokit.org] and SALib [github.com/SALib)].
packages. All dependencies are declared in the code by Python "import" statements. 
