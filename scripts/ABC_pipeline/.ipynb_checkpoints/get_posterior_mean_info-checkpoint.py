import numpy as np
from src.custom_functions import get_parameters_files, get_model_sims, load_histogram
from src.DE_simulation import DE_simulation
from src.ABM_package import ABM_simulation
import glob, sys, pdb

# ==========================================
# GRID INITIALIZATION
# ==========================================
#WH densities
densities = np.arange(10,22,2,dtype=int)  
#data spatial grid
x_d = np.arange(75, 1875.1, 50)
#ABM spatial grid
x_m = np.arange(62.5, 1900,  25) 

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
histogramDir, 
posteriorDir) = get_parameters_files(sys.argv)

#number of ABMs simulations to compute over
if model_type == "ABM":
    num_samples = 60
else:
    num_samples = None

# ==========================================
# POSTERIOR HISTOGRAM ANALYSIS
# ==========================================
# Load calculated probability weights and boundary bounds
counts, edges, probabilities = load_histogram(histogramDir)
centers = [.5*(edge[1:] + edge[:-1]) for edge in edges]

# Construct d-Dimensional coordinate system mirroring parameter space
Centers = np.meshgrid(*centers,indexing="ij")

# Point Estimate 1: Calculate Mean Vector
mean = np.array([np.sum(Center*probabilities) for Center in Centers])
# Point Estimate 2: Maximum A Posteriori (MAP)
MAP_index = np.unravel_index(np.argmax(probabilities, axis=None), probabilities.shape)
MAP = tuple([Centers[i][MAP_index] for i in np.arange(len(Centers))])
   
# Standardize arrays to unified 5-parameter signatures

#mean
if parameter_names == "rm_rp":
    mean = (mean[0], 0, mean[1], 0, 0)
elif parameter_names == "rm_pint_rp":
    mean = (mean[0], mean[1], mean[2], 0, 0)    
elif parameter_names == "rm_rp_a0_a1":
    mean = (mean[0], 0, mean[1], mean[2], mean[3])    
elif parameter_names == "rm_pint_rp_a0_a1":
    mean = mean

#MAP
if parameter_names == "rm_rp":
    MAP = (MAP[0], 0, MAP[1], 0, 0)
elif parameter_names == "rm_pint_rp":
    MAP = (MAP[0], MAP[1], MAP[2], 0, 0)    
elif parameter_names == "rm_rp_a0_a1":
    MAP = (MAP[0], 0, MAP[1], MAP[2], MAP[3])    
elif parameter_names == "rm_pint_rp_a0_a1":
    MAP = MAP
    
    
# ==========================================
# FORWARD SIMULATION ORCHESTRATION
# ==========================================

#mean PDE
if model_type == "PDE":
    mean_prediction = DE_simulation(mean,
                              data_type,
                              dens=dens,
                              simulation=simulation,
                              delay=delay)
#mean ABM
elif model_type == "ABM":
    mean_prediction = ABM_simulation(mean,
                               data_type,
                               dens=dens,
                               perc=[initial_perc],
                               delay=delay,
                               output_type = "individuals",
                               n=num_samples)

#MAP PDE
if model_type == "PDE":
    MAP_prediction = DE_simulation(MAP,
                              data_type,
                              dens=dens,
                              simulation=simulation,
                              delay=delay)

#MAP ABM    
elif model_type == "ABM":
    MAP_prediction = ABM_simulation(MAP,
                               data_type,
                               dens=dens,
                               perc=[initial_perc],
                               delay=delay,
                               output_type = "individuals",
                               n=num_samples)
    

# ==========================================
# DATA EXPORT
# ========================================== 
data = {"mean":mean,
        "mean_prediction":mean_prediction,
        "MAP":MAP,
        "MAP_prediction":MAP_prediction}

# Save structured validation results to target directory
np.save(posteriorDir + "mean_info.npy",data)
    