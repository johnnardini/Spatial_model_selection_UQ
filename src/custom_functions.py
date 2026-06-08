import sys, time, pdb
sys.path.append('../../')
from src.data_manipulation import convert_model_to_data_shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob

from src.DE_simulation import DE_simulation
from src.ABM_package import ABM_simulation
densities = np.arange(10,22,2,dtype=int)  
x_d = np.arange(75, 1875.1, 50)
x_m = np.arange(62.5, 1900,  25) 

def get_parameters_files(inputs):
    """
    Parse setup configurations, generate file directories, and determine prior 
    parameter bounds for an Approximate Bayesian Computation (ABC) pipeline.

    This function unpacks structured configuration lists for PDE or ABM models. 
    It differentiates between "artificial" data and 
    experimental wound-healing ("WH") data to set up uniform prior sample grids, 
    parameter domains, and output directory names.

    Parameters
    ----------
    inputs : list or tuple
        A sequence of exactly 6 elements containing configuration details:
        - inputs[0] : ignored (often a script name or placeholder).
        - inputs[1] : str
            The combination of active parameters to evaluate. Options:
            'rm_rp', 'rm_pint_rp', 'rm_rp_a0_a1', 'rm_pint_rp_a0_a1'.
        - inputs[2] : str
            The modeling framework used. Options: 'PDE' or 'ABM'.
        - inputs[3] : str
            The type of data being analyzed. Options: 'artificial' or 'WH'.
        - inputs[4] : int or str
            Index map: If `data_type` is 'WH', this represents an index ID (0 to 5)
            for the target density array `[10, 12, 14, 16, 18, 20]`. 
            If 'artificial', it acts as an integer simulation ID (0 to 7).
        - inputs[5] : str
            The configuration for predictions. Options: 'fit_all' or 'pred_final'.
            'fit_all' refers to fitting to all timepoints, 'pred_finl' refers to 
            out-of-sample forecasting.

    Returns
    -------
    parameter_names : str
        The designated string identifier for the target parameter set.
    model_type : str
        The modeling paradigm chosen ('PDE' or 'ABM').
    data_type : str
        The category of the dataset ('artificial' or 'WH').
    dens : int or None
        The cell density value used if `data_type` is 'WH'; otherwise None.
    simulation : int or None
        The specific simulation ID if `data_type` is 'artificial'; otherwise None.
    initial_perc : float or None
        The initial percentage configuration (defaults to 0.2 if artificial, 
        otherwise None).
    trueParams : array-like or None
        The ground-truth parameter values for 'artificial' simulations; otherwise None.
    delay : bool
        True if temporal delay parameters ('a0_a1') are included in the setup, 
        False otherwise.
    lower_bounds : numpy.ndarray
        Slices of lower thresholds bounding the uniform prior distributions.
    upper_bounds : numpy.ndarray
        Slices of upper thresholds bounding the uniform prior distributions.
    indepParams : list of numpy.ndarray
        List of 100-point linearly-spaced grids generated for each active parameter.
    dxs : list of float
        The step sizing interval grid spacing for each independent parameter.
    prediction_type : str
        Normalized prediction string type identifier ('fit_all' or 'pred_final').
    priorDir : str
        Calculated path to directory storing uniform prior distribution samples.
    histogramDir : str
        Calculated path to directory storing posterior histogram structures.
    posteriorDir : str
        Calculated path to directory storing target posterior ABC samples.

    Raises
    ------
    AssertionError
        If `inputs` does not contain exactly 6 indices, if parameter names or 
        model options are invalid, or if inputs map out-of-bounds.

    """
    
    assert(len(inputs)==6), "inputs must be of length 6"
    
    densities = np.arange(10,22,2)

    parameter_names = inputs[1]
    model_type      = inputs[2]
    data_type       = inputs[3]
    prediction_type = inputs[5]

    assert parameter_names in ["rm_rp",
                               "rm_pint_rp",
                               "rm_rp_a0_a1",
                               "rm_pint_rp_a0_a1"],"Parameters must be (rm and rp), (rm, rint, and rp), (rm, rp, a0 and a1), or (rm, pint, rp, a0 and a1)"
    
    assert model_type in ["PDE", "ABM" ],"model type must be PDE or ABM"
    assert prediction_type in ["fit_all", "pred_final" ],"model type must be fit_all or pred_final"

    if prediction_type == "pred_final":
        prediction_str = "_pred_final"
        prediction_type = "pred_final"
    else:
        prediction_str = ""
        prediction_type = "fit_all"
    
    if data_type == "WH":
        dens         = densities[int(inputs[4])]
        simulation   = None
        initial_perc = None
        priorDir     = f"../../results/prior_samples/ABC_{parameter_names}_uniform_sampling_{model_type}_{data_type}_dens_{dens}/"
        histogramDir    = f"../../results/histograms/ABC_{parameter_names}_histogram_{model_type}_{data_type}_dens_{dens}{prediction_str}"
        posteriorDir = f"../../results/posterior_samples/ABC_{parameter_names}_posterior_sampling_{model_type}_{data_type}_dens_{dens}{prediction_str}/"
        

    elif data_type == "artificial":
        simulation   = int(inputs[4])
        assert simulation in np.arange(8), "simulation must be an integer between 0 and 7"
        
        dens         = None
        initial_perc = 0.2
        
        priorDir     = f"../../results/prior_samples/ABC_{parameter_names}_uniform_sampling_{model_type}_{data_type}_simulation_{simulation}/"
        histogramDir    = f"../../results/histograms/ABC_{parameter_names}_histogram_{model_type}_{data_type}_simulation_{simulation}{prediction_str}"
        posteriorDir = f"../../results/posterior_samples/ABC_{parameter_names}_posterior_sampling_{model_type}_{data_type}_simulation_{simulation}{prediction_str}/"
        
    else:
        assert False,"data type must be artificial or WH"

    if "a0_a1" in parameter_names:
        delay = True
    else:
        delay = False
    
    ###
    # Parameter information
    ###
    
    if data_type == "artificial":
        lower_bounds = np.array([    0,   0,   0, -20.0,    0])
        upper_bounds = np.array([400.0, 1.0, 3.0,  20.0, 40.0])
    elif data_type == "WH":
        lower_bounds = np.array([    0,   0,   0, -20.0,    0])
        upper_bounds = np.array([400.0, 1.0, 5.0,  20.0, 40.0])
    
    rms  = np.linspace(lower_bounds[0],upper_bounds[0],100)
    pint = np.linspace(lower_bounds[1],upper_bounds[1],100)
    rps  = np.linspace(lower_bounds[2],upper_bounds[2],100)
    a0   = np.linspace(lower_bounds[3],upper_bounds[3],100)
    a1   = np.linspace(lower_bounds[4],upper_bounds[4],100)
    
    if parameter_names == "rm_rp":
        inds = [0,2]
    elif parameter_names == "rm_pint_rp":
        inds = [0,1,2]
    elif parameter_names == "rm_rp_a0_a1":
        inds = [0,2,3,4]
    elif parameter_names == "rm_pint_rp_a0_a1":
        inds = [0,1,2,3,4]
    
    lower_bounds = lower_bounds[inds]
    upper_bounds = upper_bounds[inds]
    
    if parameter_names == "rm_rp":
        indepParams = [rms,rps]
    elif parameter_names == "rm_pint_rp":
        indepParams = [rms,pint,rps]
    elif parameter_names == "rm_rp_a0_a1":
        indepParams = [rms,rps,a0,a1]
    elif parameter_names == "rm_pint_rp_a0_a1":
        indepParams = [rms,pint,rps,a0,a1]
    
    dxs = [x[1]-x[0] for x in indepParams]
    
    ###
    # True parameter values
    ###
    if data_type == "artificial":
        
        trueParams = get_true_params_artificial(simulation,parameter_names)
            
    elif data_type == "WH":
        trueParams = None
    
    return (parameter_names, 
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
            posteriorDir)

def get_true_params_artificial(simulation,parameter_names):
    """
    Retrieve the ground-truth parameter values for a specified Artificial simulation.

    This helper function maps a simulation ID to its true underlying biological parameters 
    (cell motility rate, initial profile intensity, proliferation rate, and temporal delay 
    coefficients) and slices them according to the target parameters being evaluated.

    Parameters
    ----------
    simulation : int
        The unique identifier for the synthetic simulation setup (expected 0 to 7 based on 
        current logic).
    parameter_names : str
        The specific combination of active parameters to slice and return.
        Options: 'rm_rp', 'rm_pint_rp', 'rm_rp_a0_a1', 'rm_rp_fix_a0_a1', 
        'rm_pint_rp_a0_a1'.

    Returns
    -------
    trueParams : tuple
        A tuple containing the ground-truth values corresponding strictly to the requested 
        `parameter_names` slice. For instance, if `parameter_names` is 'rm_rp', returns 
        `(rm, rp)`.

    """
    if simulation == 0:
        rm,pint,rp,a0,a1 = 50,0.0,0.5,0.0,0.0
    elif simulation == 1:
        rm,pint,rp,a0,a1 = 50,0.0,2.0,0.0,0.0
    elif simulation == 2:
        rm,pint,rp,a0,a1 = 250,0.0,0.5,0.0,0.0
    elif simulation == 3:
        rm,pint,rp,a0,a1 = 250,0.0,2.0,0.0,0.0
    elif simulation == 4:
        rm,pint,rp,a0,a1 =  50,0.5,0.5,0.0,0.0
    elif simulation == 5:
        rm,pint,rp,a0,a1 = 250,0.0,2.5,0.0,5.0
    elif simulation == 6:
        rm,pint,rp,a0,a1 = 250,0.0,2.5,-1.25,5.0
    elif simulation == 7:
        rm,pint,rp,a0,a1 = 250,0.5,2.5,0.0,5.0
        
        
    if parameter_names == "rm_rp":
        trueParams = (rm,rp)
    elif parameter_names == "rm_pint_rp":
        trueParams = (rm,pint,rp)    
    elif parameter_names == "rm_rp_a0_a1":
        trueParams = (rm,rp,a0,a1) 
    elif parameter_names == "rm_rp_fix_a0_a1":
        trueParams = (rm,rp,a0,a1) 
    elif parameter_names == "rm_pint_rp_a0_a1":
        trueParams = (rm,pint,rp,a0,a1)   
        
        
    return trueParams
    


def load_histogram(histogramDir):
    """
    Load saved histogram components from a .npy file.

    This function loads the NumPy array and unpacks 
    the raw counts, bin edges, and normalized probability densities.

    Parameters
    ----------
    histogramDir : str
        The target directory path or filename where the histogram is stored. 
        Can optionally include or omit the '.npy' file extension.

    Returns
    -------
    counts : numpy.ndarray or dict
        The frequency of data points falling within each bin across the parameter space.
    edges : list of numpy.ndarray
        The bin boundaries defined for each parameter dimension.
    probabilities : numpy.ndarray or dict
        The normalized probability density scale calculated across the histogram structure.

    """
    
    if histogramDir[-4:] != ".npy":
        histogramDir += ".npy"
    
    mat = np.load(histogramDir,allow_pickle=True).item()
    counts, edges, probabilities = (mat["counts"],
                                    mat["edges"],
                                    mat["probabilities"])
    
    
    return counts, edges, probabilities


def area_under_curve(f,dxs):
    """
    Approximate the total volume or area under a multidimensional grid-based surface 
    using a uniform Riemann sum.

    Parameters
    ----------
    f : numpy.ndarray
        An array of any shape containing evaluated function values or probabilities 
        at each discrete node across the parameter grid.
    dxs : list of float or numpy.ndarray
        The uniform step sizes (differential steps) for each dimension of the grid. 
        The length of `dxs` should correspond to the number of dimensions in `f`.

    Returns
    -------
    total_area : float
        The scalar approximation of the integrated volume under the surface.

    """
    return np.sum( np.prod(dxs)*f )

def histogram_marginal(edges,probabilities, indices):
    """
    Compute the marginal probability distribution over a subset of parameters.

    This function integrates (sums) out a set of unwanted dimensions from a 
    multidimensional joint probability grid, isolating the marginal distribution 
    for the parameters specified by `indices`.

    Parameters
    ----------
    edges : list of numpy.ndarray
        A list of arrays where each array defines the bin boundaries or coordinate 
        axes for a specific parameter dimension in the joint distribution.
    probabilities : numpy.ndarray
        A multi-dimensional array representing the joint probability distribution 
        matrix. The number of dimensions must match `len(edges)`.
    indices : int, tuple of int, or list of int
        The target axis index (or indices) to retain. All dimensions *not* specified 
        in this collection will be integrated out.

    Returns
    -------
    marginal_edges : list of numpy.ndarray
        The subset of `edges` corresponding strictly to the retained dimensions 
        specified in `indices`.
    marginal_probabilities : numpy.ndarray
        The collapsed probability tensor containing only the dimensions of interest.
    """
        
    if type(indices) == int: indices = (indices,)
    
    indices = tuple(np.unique(indices))
    
    num_params = len(edges)
    all_indices = np.arange(num_params)
    
    assert np.max(all_indices) <= num_params-1, "indices must be less than or equal to number of params"
    
    sum_indices = tuple(i for i in all_indices if i not in indices)
    
    marginal_edges = [edges[i] for i in indices]
    marginal_probabilities = np.sum(probabilities,axis=sum_indices)
    
    
    return marginal_edges, marginal_probabilities

def histogram_CI_threshold(probabilities):
    """
    Find the probability density threshold defining the 90% Highest Density Region 
    credible interval.

    This function identifies the cutoff threshold by sorting unique probability values 
    in descending order and cumulatively summing them. It isolates the smallest set of 
    highest-probability cells whose collective volume meets or exceeds the target 
    mass (alpha = 0.9).

    Parameters
    ----------
    probabilities : numpy.ndarray
        A multi-dimensional array representing a normalized discrete joint probability 
        distribution grid (where the sum of all elements equals 1.0).

    Returns
    -------
    prev_thres : float
        The minimum probability density value required for a grid cell to be 
        included within the 90% credible interval.
    """
    unique_probs = np.unique(probabilities)
    unique_probs.sort()
    unique_reverse = unique_probs[::-1]

    alpha = 0.9

    for threshold in unique_reverse:
        CIIndices = probabilities > threshold
        CI = probabilities[CIIndices]

        if np.sum(CI) >= alpha:
            break
        prev_thres = threshold
            
    return prev_thres

def convex_hull_construction(CI, marginal_edges):
    """
    Construct a geometric convex hull around a discrete 2D credible interval mask.

    This function extracts the matrix indices of a 2D credible interval (CI) boundary 
    mask, translates those cell coordinates into physical coordinate values using the 
    bin edges, and builds a bounding convex polygon encompassing all valid regions.

    Parameters
    ----------
    CI : numpy.ndarray
        A 2D binary mask/heatmap array (containing 1s inside the credible interval 
        and 0s outside) representing the targeted distribution region.
    marginal_edges : list of numpy.ndarray
        A list of two 1D arrays defining the bin boundaries for the x-axis and y-axis 
        respectively.

    Returns
    -------
    hull : scipy.spatial.ConvexHull
        The computed convex hull object containing geometric attributes like 
        `vertices`, `area`, and `volume` (perimeter in 2D).
    points : numpy.ndarray
        A unique 2D array of shape `(N, 2)` representing all the physical 
        corner points used to calculate the convex hull.
    """
    CI_indices = np.array(np.where(CI==1)).T
    
    points = []
    for x,y in CI_indices:
        points.append((marginal_edges[0][x],marginal_edges[1][y]))
        points.append((marginal_edges[0][x+1],marginal_edges[1][y]))
        points.append((marginal_edges[0][x],marginal_edges[1][y+1]))
        points.append((marginal_edges[0][x+1],marginal_edges[1][y+1]))    
    points = np.unique(points,axis=0)

    hull = ConvexHull(points)
    return hull, points

def freedman_diaconis(selectedParams):
    """
    Calculate the optimal histogram bin width for a 2D parameter dataset using 
    the Freedman-Diaconis rule.

    This rule minimizes the difference between the empirical histogram and the 
    true underlying probability density function by leveraging the Interquartile 
    Range (IQR). It is robust against outliers because it utilizes medians and 
    quartiles rather than the sample standard deviation.

    Parameters
    ----------
    selectedParams : numpy.ndarray
        A 2-dimensional array of shape `(n_samples, n_parameters)` containing the 
        observed distribution or posterior sample points.

    Returns
    -------
    bin_widths : numpy.ndarray
        A 1D array containing the mathematically calculated optimal bin width for 
        each respective parameter axis.
    """
    assert type(selectedParams) == np.ndarray, "selectedParams must be a ndarray"
    assert selectedParams.ndim == 2, "selectedParams must be 2-dimensional"
    
    Q1 = np.percentile(selectedParams,25,axis = 0)
    Q2 = np.percentile(selectedParams,75,axis = 0)
    n = selectedParams.shape[0]
    IQR = Q2 - Q1

    return 2*IQR/(n**(1/3))
    
def compute_histogram_posterior(params, lower_bounds, upper_bounds, bin_width, savefile=None):
    """
    Generate and serialize a multi-dimensional posterior probability histogram.

    This function calculates the necessary number of bins for each parameter dimension 
    based on predefined boundaries and target bin widths. It builds a linearly-spaced 
    grid array, passes it to a multidimensional histogram generator, and optionally 
    save the resulting components to file.

    Parameters
    ----------
    params : numpy.ndarray
        A 2D array of shape `(n_samples, n_parameters)` containing the accepted 
        posterior samples from the ABC pipeline.
    lower_bounds : array-like
        The minimum boundary domain for each parameter domain, matching 
        length `n_parameters`.
    upper_bounds : array-like
        The maximum boundary domain for each parameter domain, matching 
        length `n_parameters`.
    bin_width : array-like
        The target width/resolution for bins along each parameter dimension 
        (e.g., calculated via the Freedman-Diaconis rule).
    savefile : str, optional
        The file path destination (typically ending in `.npy`) where the 
        structured dictionary array data will be saved. If None, file saving 
        is skipped.

    Returns
    -------
    counts : numpy.ndarray
        The raw point frequency counts mapped inside each multidimensional bin cell.
    edges : list of numpy.ndarray
        A list of arrays describing the exact bin edge boundaries calculated for 
        each parameter dimension.
    probabilities : numpy.ndarray
        The normalized probability density across the multi-dimensional parameter space.

    """
    bins = []
    for i in range(params.shape[1]):
        num_bins = np.max([int(np.round((upper_bounds[i]-lower_bounds[i])/bin_width[i])),2])
        bins.append(np.linspace(lower_bounds[i],upper_bounds[i],num_bins))

    [counts, 
     edges, 
     probabilities] = dd_histogram_generation(params,
                                             bins=bins)
    
    if savefile is not None:
        
        data = {"counts":counts,
                "edges":edges,
                "probabilities":probabilities}
        np.save(savefile,data)
        
    return counts, edges, probabilities
    
def dd_histogram_generation(data, bins=10):
    """
    Generate a d-dimensional histogram using uniform sampling within bins.
    
    Parameters:
    - data: array-like, shape (n_samples, d)
      Input data where each row is a data point and each column is a feature/dimension.
    - bins: int or sequence of ints
      The number of bins for each dimension (can be a single value or a list for each dimension).
    
    Returns:
    - counts: array, shape (bins, bins)
      counts of samples in each histogram bin
    - edges: list, A list of D arrays describing the bin edges for each dimension.
    - probabilities: probability of observing data in each bin of the histogram
    """
    
    # Step 1: Create a d-dimensional histogram
    counts, edges = np.histogramdd(data, bins=bins, density=False)

    # Step 2: Normalize the counts to get probabilities
    probabilities = counts / np.sum(counts)
    
    return counts, edges, probabilities    

def dd_histogram_sampling(probabilities, edges, n_samples = 1000):
    """
    Generate samples from a d-dimensional histogram using uniform sampling within bins.
    
    Parameters:
    - edges: list, A list of D arrays describing the bin edges for each dimension.
    - probabilities: probability of observing data in each bin of the histogram
    - n_samples: int
      Number of samples to generate.
    
    Returns:
    - sampled_data: array, shape (n_samples, d)
      New samples generated from the estimated distribution.
    """
    
    # Step 1: Sample bins based on the probabilities
    flat_probs = probabilities.ravel()  # Flatten the d-dimensional counts into 1D
    chosen_bins = np.random.choice(len(flat_probs), size=n_samples, p=flat_probs)

    # Step 2: Map the chosen flat indices to d-dimensional bin indices
    chosen_bin_indices = np.array(np.unravel_index(chosen_bins, probabilities.shape)).T

    # Step 3: Sample uniformly within each chosen bin
    sampled_data = np.empty((n_samples, len(edges)))  # Initialize array for sampled data

    for i, bin_index in enumerate(chosen_bin_indices):
        # For each dimension, sample uniformly between the corresponding bin edges
        for dim in range(len(edges)):
            lower_edge = edges[dim][bin_index[dim]]
            upper_edge = edges[dim][bin_index[dim] + 1]
            sampled_data[i, dim] = np.random.uniform(lower_edge, upper_edge)

    return sampled_data


def select_params_MSE_perc(results, params, percentage):
    """
    Filter parameter sets whose Mean Squared Error (MSE) metrics fall within a 
    specified percentage threshold of the global minimum error.

    This function implements a threshold-based acceptance filter to isolate 
    near-optimal parameter combinations based on model performance.

    Parameters
    ----------
    results : numpy.ndarray
        A 1D array of shape `(n_samples,)` containing the calculated error metrics 
        (e.g., MSE) for each candidate parameter simulation.
    params : numpy.ndarray
        A 2D array of shape `(n_samples, n_parameters)` containing the parameter 
        combinations corresponding to the evaluations in `results`.
    percentage : float
        The tolerance percentage expressed as a decimal (e.g., `0.25` for 25%). 
        Determines how far above the minimum error a parameter set can be and still 
        be accepted.

    Returns
    -------
    selectedParams : numpy.ndarray
        A 2D array containing only the filtered parameter rows that satisfied 
        the threshold constraint.
    """
    min_result = np.min(results)
    
    selectedParams = params[results <= (1+percentage)*min_result]
    
    return selectedParams
    

def get_simulation_results(simulations, Data, x_m, model_type,prediction_type):
    """
    Load simulation outputs, normalize data shapes, and calculate Mean Squared Error (MSE) 
    against data.

    This function iterates through simulation files, extracts their underlying 
    parameters and results, converts Agent-Based Model (ABM) grids into comparable continuous data 
    shapes if necessary, and scales both model and experimental data by a carrying capacity 
    constant (K) before evaluating model performance using MSE.

    Parameters
    ----------
    simulations : list of str
        A collection of file paths pointing to serialized NumPy binary (.npy) files, 
        each containing the results and parameter configurations of a unique simulation run.
    Data : numpy.ndarray
        A 2D array representing the target dataset 
        used as the ground truth baseline for error evaluation.
    x_m : array-like or spatial_grid_config
        Spatial grid parameters passed to `convert_model_to_data_shape` to reshape ABM outputs
        to match the Data grid.
    model_type : str
        The framework of the simulation model being analyzed. Options: 'ABM' or 'PDE'.
    prediction_type : str
        The filtering criteria for MSE evaluation. If 'pred_final', the final column of 
        the matrices is sliced out of the error check (for out-of-sample forecasting); 
        if 'fit_all', the entire matrix space is included.

    Returns
    -------
    results : numpy.ndarray
        A 1D array of shape `(n_simulations,)` containing the computed scalar MSE value 
        for each simulation run.
    params : numpy.ndarray
        A 2D array of shape `(n_simulations, n_parameters)` capturing the raw parameter 
        combinations extracted from each simulation file.
    """
    
    assert model_type in ["ABM","PDE"], "model_type must be ABM or PDE"
    
    results = []
    params = []
    K = 120.0

    for simulation in simulations:
        mat = np.load(simulation
                  ,allow_pickle=True).item()

        param = mat['param']
        model = mat['result']

        if model_type == "ABM":
            model_d = convert_model_to_data_shape(model,x_m)
        elif model_type == "PDE":
            model_d = model
            
        if "pred_final" in prediction_type:
            results.append(MSE(model_d[:,:-1]/K, Data[:,:-1]/K))
        elif prediction_type == "fit_all":
            results.append(MSE(model_d/K, Data/K))

        
        
        params.append(param)

    results = np.array(results)
    params = np.array(params)
    
    return results, params

def MSE(a,b):
    """
    Calculate the mean squared error between two arrays.

    Args:
        a (numpy.ndarray): The first array.
        b (numpy.ndarray): The second array.

    Returns:
        float: The mean squared error between `a` and `b`.
    """
    assert a.shape == b.shape
    
    MSE = ((a - b)**2).mean()
    
    return MSE

def RMSE(true,pred):
    """
    Calculate the relative mean squared error between two arrays.

    Args:
        true (numpy.ndarray): The true array.
        pred (numpy.ndarray): The predicted array for true.

    Returns:
        float: The mean squared error between `a` and `b`.
    """
    assert true.shape == pred.shape
    
    RMSE = ( ((true - pred)/true)**2 ).mean()
    
    return RMSE