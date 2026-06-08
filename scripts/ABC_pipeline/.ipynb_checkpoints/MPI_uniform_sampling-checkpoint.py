import time, sys, os
from mpi4py import MPI
import numpy as np

from src.DE_simulation import DE_simulation
from src.ABM_package import ABM_simulation
from src.custom_functions import get_parameters_files

# ==========================================
# MPI ENVIRONMENT INITIALIZATION
# ==========================================
comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

densities = np.arange(10,22,2)

# Unpack command-line configurations via custom parser
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
priorDir, 
kernelDir, 
posteriorDir) = get_parameters_files(sys.argv)

# ==========================================
# MASTER PROCESS (RANK 0): PARAMETER GENERATION
# ==========================================
if rank == 0:
    # 1. Logging Setup Context
    if data_type == "WH":
        print(f"\nPerforming prior sampling for parameters {parameter_names} for {model_type} model for {data_type} data with density {dens}\n")
    elif data_type == "artificial":
        print(f"\nPerforming prior sampling for parameters {parameter_names} for {model_type} model for {data_type} data with simulation {simulation}\n")

    
    #initialize data-directory
    if not os.path.exists(priorDir):
        os.mkdir(priorDir)
        
    # 2. Total prior sample threshold
    N = int(1e4)
    
    # 3. Reproducible Seeding Strategy based on configuration
    if dens is not None:
        np.random.seed(dens)    
    elif simulation is not None:
        np.random.seed(1000 + 2*simulation)
        
    numDataPerRank = int(np.floor(N/size))    
    paramsList = []
    
    # 4. Allocate parameter segments uniformly across MPI ranks
    for i in range(size):    
        
        #Baseline amount of simulations for each worker
        remainder = N%size
        
        #Determine if the worker gets assigned a leftover simulation to ensure the
        #total number of simulations is N
        if i < remainder:
            numParams = numDataPerRank + 1
        else:
            numParams = numDataPerRank
    
        # Uniformly sample for each parameter
        if parameter_names == "rm_rp":
            rms = np.random.uniform(low  = lower_bounds[0], 
                                    high = upper_bounds[0], 
                                    size = (numParams,1))
            ppulls = np.zeros(rms.shape)          
            rps = np.random.uniform(low  = lower_bounds[1], 
                                    high = upper_bounds[1], 
                                    size = (numParams,1))
            a0s = np.zeros(rms.shape)          
            a1s = np.zeros(rms.shape)          
        elif parameter_names == "rm_pint_rp":
            rms = np.random.uniform(low  = lower_bounds[0], 
                                    high = upper_bounds[0], 
                                    size = (numParams,1))
            ppulls = np.random.uniform(low  = lower_bounds[1], 
                                    high = upper_bounds[1], 
                                    size = (numParams,1))         
            rps = np.random.uniform(low  = lower_bounds[2], 
                                    high = upper_bounds[2], 
                                    size = (numParams,1))
            a0s = np.zeros(rms.shape)          
            a1s = np.zeros(rms.shape)     
        elif parameter_names == "rm_rp_a0_a1":
            rms = np.random.uniform(low  = lower_bounds[0], 
                                    high = upper_bounds[0], 
                                    size = (numParams,1))
            ppulls = np.zeros(rms.shape)          
            rps = np.random.uniform(low  = lower_bounds[1], 
                                    high = upper_bounds[1], 
                                    size = (numParams,1))
            a0s = np.random.uniform(low  = lower_bounds[2], 
                                    high = upper_bounds[2], 
                                    size = (numParams,1))
            a1s = np.random.uniform(low  = lower_bounds[3], 
                                    high = upper_bounds[3], 
                                    size = (numParams,1))
        elif parameter_names == "rm_rp_fix_a0_a1":
            rms = np.random.uniform(low  = lower_bounds[0], 
                                    high = upper_bounds[0], 
                                    size = (numParams,1))
            ppulls = np.zeros(rms.shape)          
            rps = np.random.uniform(low  = lower_bounds[1], 
                                    high = upper_bounds[1], 
                                    size = (numParams,1))
            a0s = trueParams[-2]*np.ones(rms.shape)
            a1s = trueParams[-1]*np.ones(rms.shape)
        elif parameter_names == "rm_pint_rp_a0_a1":
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
            
        
            
        
        #Combine vectors into matrix structure for this rank
        paramsList.append(np.hstack([rms, 
                                     ppulls,
                                     rps,
                                     a0s,
                                     a1s])
                         )
        
else:
    paramsList = None

# ==========================================
# WORKER EXECUTION: DISTRIBUTION & SIMULATION
# ==========================================

# Scatter the stacked payload array from Rank 0 down to all individual workers
params = comm.scatter(paramsList, root=0)

# Execute the local specific loop assigned to this worker node
if model_type == "PDE":
    for param in params:
        DE_simulation(param,
                      data_type,
                      dens=dens,
                      simulation=simulation,
                      resultsDir=priorDir,
                      delay=delay)

elif model_type == "ABM":
        
    for param in params:    
        ABM_simulation(param,
                       data_type,
                       dens=dens,
                       perc=[initial_perc],
                       resultsDir=priorDir,
                       delay=delay
                       )