import time, sys, os

from mpi4py import MPI
import numpy as np

from src.ABM_package import ABM_simulation
from src.custom_functions import get_parameters_files

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

#Model 4 (rm_pint_rp_a0_a1)
parameter_names = "rm_pint_rp_a0_a1"
model_type = "ABM"
data_type = "artificial"
simulation = 7
pred_type = "fit_all"

input_parameters = [None, parameter_names, model_type, data_type, simulation, pred_type]

(parameter_names, 
    model_type, 
    data_type, 
    dens, 
    simulation,
    initial_perc,
    trueParams,
    delay, 
    lower_bounds,
    upper_bounds,
    indepParams,
    dxs,
    prediction_type,
    _, #ignore prior_dir 
    histogramDir, 
    posteriorDir) = get_parameters_files(input_parameters)

# ==========================================
# Run the ABM 1,000 times and record the walltimes
# ==========================================

dir_ = "../../results/timing/"

if rank == 0:
    
    N = int(1e3)
    
    np.random.seed(0)
        
    numDataPerRank = int(np.floor(N/size))    
    
    paramsList = []
    for i in range(size):    
        
        #add more params to get N total
        remainder = N%size
        
        if i < remainder:
            numParams = numDataPerRank + 1
        else:
            numParams = numDataPerRank
        
        ### Sample the parameters uniformly
        if parameter_names == "rm_pint_rp_a0_a1":
            rms = np.random.uniform(low  = lower_bounds[0], 
                                    high = upper_bounds[0], 
                                    size = (numParams,1))
            ppulls = np.random.uniform(low  = lower_bounds[1], 
                                    high = upper_bounds[1], 
                                    size = (numParams,1))      
            rps = np.random.uniform(low  = lower_bounds[2], 
                                    high = upper_bounds[2], 
                                    size = (numParams,1))
            a0s = np.random.uniform(low  = lower_bounds[3], 
                                    high = upper_bounds[3], 
                                    size = (numParams,1))
            a1s = np.random.uniform(low  = lower_bounds[4], 
                                    high = upper_bounds[4], 
                                    size = (numParams,1))
            
        
            
        #combine all samples into one large matrix
        paramsList.append(np.hstack([rms, 
                                     ppulls,
                                     rps,
                                     a0s,
                                     a1s])
                         )
        
    time0 = time.time()
        
else:
    paramsList = None

#send out parameter assignments to all workers
params = comm.scatter(paramsList, root=0)

#perform ABM simulations (walltime will be saved automatically)
for param in params:    
    ABM_simulation(param,
                   data_type,
                   dens=dens,
                   perc=[initial_perc],
                   resultsDir=dir_,
                   delay=delay
                   )