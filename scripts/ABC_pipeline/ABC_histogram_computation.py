import time, sys, os, pdb

import numpy as np
import matplotlib.pyplot as plt

import itertools, glob

from src.custom_functions import MSE, select_quantile_params, get_simulation_results, freedman_diaconis, compute_histogram_posterior, get_parameters_files, dd_histogram_generation, select_params_MSE_perc
from src.data_loader import load_data
from src.ABM_package import migration_step_pulling, simulate_nonlinear_migration_ABM

# ==========================================
# GRID INITIALIZATION
# ==========================================
#WH densities
densities = np.arange(10,22,2,dtype=int)  
#data spatial grid
x_d = np.arange(75, 1875.1, 50)
#ABM spatial grid
x_m = np.arange(62.5, 1900,  25) 

# Unpack command line tracking configurations
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

# Logging run context
if data_type == "WH":
    print(f"\nComputing the histogram for parameters {parameter_names} for {model_type} model for {data_type} data with density {dens}\n")
elif data_type == "artificial":
    print(f"\nComputing the histogram for parameters {parameter_names} for {model_type} model for {data_type} data with simulation {simulation}\n")

# ==========================================
# DATA LOADING & PERFORMANCE EVALUATION
# ==========================================
# 1. Load data
Data = load_data(data_type, dens=dens, simulation=simulation)

# 2. Collect all prior simulations from storage
simulations = glob.glob(f"{priorDir}*")

# 3. Parse forward outputs and match dimensions against baseline shapes
results, params = get_simulation_results(simulations, Data, x_m, model_type,prediction_type)

# ==========================================
# PARAMETER DIMENSION REDUCTION
# ==========================================
# Slice the parameter array down to ONLY the dimensions that were actively sampled
if parameter_names == "rm_rp":
    params = np.column_stack((params[:,0], params[:,2]))
elif parameter_names == "rm_pint_rp":
    params = params[:,:3]
elif parameter_names == "rm_rp_a0_a1":
    params = np.column_stack((params[:,0], params[:,2:]))
elif parameter_names == "rm_pint_rp_a0_a1":
    params = params #using all params

# ==========================================
# ABC REJECTION FILTER & DENSITY CALCULATION
# ==========================================
# Initial target constraint: Accept only the fits within 25% of the best-fit
perc = 0.25
selectedParams = select_params_MSE_perc(results, params, perc)

# Safety Loop: If the constraint is too tight, widen acceptance window 
# to guarantee a statistically valid sample depth (minimum of 5 records)
while(len(selectedParams) < 5):
    perc+=0.25
    selectedParams = select_params_MSE_perc(results, params, perc)

# Determine histogram grid resolution based on Freedman Diaconis rule
bin_width = freedman_diaconis(selectedParams)

# Compute multivariate probability density grid and save
compute_histogram_posterior(selectedParams,lower_bounds,upper_bounds,bin_width,savefile=histogramDir)

