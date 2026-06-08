import random
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy import sparse
from scipy.integrate import odeint
from src.data_loader import load_data
from src.ABM_package import save_data, Tau

def DE_simulation(param, data_type, dens=None, simulation=None, resultsDir=None, delay=False):
    """
    Execute a forward continuous Reaction-Diffusion PDE numerical simulation.

    This function sets up a fixed spatial grid and temporal domain, extracts the simulation
    initial condition (IC), and invokes a standard numerical ordinary differential equation
    integrator (`scipy.integrate.odeint`) to resolve the system's states across time.

    Parameters
    ----------
    param : array-like
        The specific vector of physical and kinetic parameters evaluated in the
        simulation.
    data_type : str
        The category of data targeted. Options: 'WH' (Wound Healing) or 'artificial'.
    dens : int, optional
        The specific cell density setup if `data_type` is 'WH'. Otherwise, defaults to None.
    simulation : int, optional
        The structural simulation index if `data_type` is 'artificial'. Otherwise, defaults to None.
    resultsDir : str, optional
        The storage path destination directory. If a path is provided, the function 
        saves the parameters and numerical solutions via `save_data`. If None, 
        file writing is bypassed.
    delay : bool, default=False
        Flag passed to the system equations determining whether to delay the temporal dynamics.

    Returns
    -------
    sol : numpy.ndarray
        A 2D array of shape `(n_spatial_nodes, n_time_steps)` containing the calculated 
        population densities, transposed so rows represent space and columns represent 
        time points.
    """
    assert data_type in ["WH","artificial"], "simulation_type must be WH or artificial"
    
    x_d = np.arange(75, 1875.1, 50)
    t = np.linspace(0,2,5)

    Data = load_data(data_type, dens=dens, simulation=simulation)
    IC = Data[:,0]
        
    sol = odeint(Reaction_Diffusion_eqn, IC, t, args=(x_d, 
                                                      param, 
                                                      simple_pulling_diffusion, 
                                                      logistic_proliferation, 
                                                      delay)
                )
    sol = sol.T 
    
    if resultsDir is not None:
        #save updated results
        save_data(param, sol, resultsDir)
    
    return sol
    

def simple_pulling_diffusion(u,q):
    """
    Compute the Pulling ABM's mean-field DE model for a quantity `u` with parameters `q`.

    Parameters:
        u (np.ndarray): The quantity to be diffused.
        q (iterable): parameters
            - q[0]: rmp, the rate of pulling agent migration
            - q[1]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D(u;q)
    """
    return 625*q[0]/4.0*(1.0+3*q[1]*(u/120)**2)

def Reaction_Diffusion_eqn(u, t, x, q, diffusion_function, growth_function, delay=False):
    """
    Compute the right-hand side dynamics of a discretized Reaction-Diffusion partial 
    differential equation (PDE).

    This function defines the spatial state derivatives by combining a 
    density-dependent spatial diffusion operator with a localized population growth 
    kinetics function. It can optionally scale the global timeline dynamics using a 
    temporal delay/activation multiplier.

    Parameters
    ----------
    u : numpy.ndarray
        A 1D array of shape `(n_spatial_nodes,)` representing the current population 
        densities across the spatial grid coordinates.
    t : float
        The current simulation time step passed by the numerical ODE integrator.
    x : numpy.ndarray
        A 1D array representing the linearly-spaced spatial grid vector coordinates.
    q : list, tuple, or array-like
        The model parameter coefficient array containing kinetic constants
    diffusion_function : callable
        A function or operator with the signature `f(u, q)` that evaluates to a 
        density-dependent diffusion coefficient array matching the shape of `u`.
    growth_function : callable
        A function or operator with the signature `f(u, q)` that evaluates to a 
        density-dependent growth scaling coefficient array matching the shape of `u`.
    delay : bool, default=False
        If True, applies a time-dependent attenuation or activation scaling factor 
        via the `Tau(t, a0, a1)` operator.

    Returns
    -------
    final_dynamics : numpy.ndarray
        A 1D array of shape `(n_spatial_nodes,)` representing the instantaneous spatial 
        derivatives ($du/dt$).
    """
    dx = x[1] - x[0]
    
    D_matrix = D_u(diffusion_function(u,q),dx)
        
    growth_rate = growth_function(u,q)
    
    dynamics = D_matrix.dot(u) + growth_rate*u
    
    if delay is False:
        final_dynamics = dynamics
    elif delay is True:
        a0 = q[3]
        a1 = q[4]
        final_dynamics = Tau(t,a0,a1)*dynamics
        
    return final_dynamics

def D_u(D,dx):
    
    """
    Create the matrix operator for a discretized diffusion equation with density-varying diffusion coefficients.

    Parameters:
        D (np.ndarray): Vector of diffusion coefficient values.
        dx (float): Spatial resolution.

    Returns:
        scipy.sparse.coo_matrix: The matrix operator for the discretized diffusion equation.

    Note:
    - The matrix operator is constructed for solving (D(u)u_x)_x in a discretized form.
    - The discretization is based on Equation (4.13) from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    """
    n = len(D)
    
    D_ind = np.arange(n)

    #first consruct interior portion of D
    #exclude first and last point and include those in boundary
    D_ind = D_ind[1:-1] 
    Du_mat_row = np.hstack((D_ind,D_ind,D_ind))
    Du_mat_col = np.hstack((D_ind+1,D_ind,D_ind-1))
    Du_mat_entry = (1.0/(2*dx**2))*np.hstack((D[D_ind+1]+D[D_ind],
                   -(D[D_ind-1]+2*D[D_ind]+D[D_ind+1]),D[D_ind-1]+D[D_ind]))
    
    #boundary points
    Du_mat_row_bd = np.array((0,0,n-1,n-1))
    Du_mat_col_bd = np.array((0,1,n-1,n-2))
    Du_mat_entry_bd = (1.0/(2*dx**2))*np.array((-2*(D[0]+D[1]),
                    2*(D[0]+D[1]),-2*(D[-2]+D[-1]),2*(D[-2]+D[-1])))
    #add in boundary points
    Du_mat_row = np.hstack((Du_mat_row,Du_mat_row_bd))
    Du_mat_col = np.hstack((Du_mat_col,Du_mat_col_bd))
    Du_mat_entry = np.hstack((Du_mat_entry,Du_mat_entry_bd))

    return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)))



def logistic_proliferation(u, q):
    """
    Calculate the logistic proliferation rate for a cell population.

    Parameters
    ----------
    u : float or numpy.ndarray
        The current local cell density or population array.
    q : list, tuple, or array-like
        The vector containing model parameter coefficients.

    Returns
    -------
    growth_rate : float or numpy.ndarray
        The calculated proliferation rate value or array, matching the shape of `u`.
    """
    K = 120.0
    
    return q[2]*(1.0-u/K)