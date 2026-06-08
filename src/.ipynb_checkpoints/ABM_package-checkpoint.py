import xlrd#, math
import random
import numpy as np
from numbers import Number
import pdb
from scipy.interpolate import LinearNDInterpolator
import time
import copy

from numbers import Number
from src.data_manipulation import convert_num_d_to_num_m, convert_num_m_to_ABM
from src.data_loader import load_data

def ABM_simulation(p, simulation_type, dens = None, perc=None, resultsDir=None, n = 3, delay=False, output_type = "mean"):
    """
    Executes a non-linear pulling migration Agent-Based Model (ABM) simulation.

    This function acts as a wrapper around the underlying collective migration 
    simulator. It tracks execution time, handles optional file logging, and 
    allows the user to toggle between returning aggregated mean behavior or the 
    raw individual simulation trajectories.

    Parameters
    ----------
    p : array_like
        Parameters or initial conditions defining the migration model configuration.
    simulation_type : str
        The specific setup variation or scenario configuration for the simulation 
        (e.g., `artificial` or `WH`).
    dens : float, optional
        Density identifier for WH datasets. Defaults to None.
    perc : float, optional
        initial percentage of the ABM lattice points to occupy for the artificial datasets.
    resultsDir : str, optional
        Directory path where the output results and execution time profile will 
        be saved. If None, the data is not written to disk. Defaults to None.
    n : int, optional
        The number of stochastic simulation runs or repetitions to execute for 
        averaging. Defaults to 3.
    delay : bool, optional
        If True, introduces or enables a time-delay 
        within the migration and proliferation rules. Defaults to False.
    output_type : {"mean", "individuals"}, optional
        Determines the format of the returned data. "mean" computes and returns 
        the ensemble average across all runs. "individuals" returns the detailed 
        trajectory arrays for each individual simulation run. Defaults to "mean".

    Returns
    -------
    result : numpy.ndarray or list of numpy.ndarray
        Depending on `output_type`:
        - If "mean": A single array representing the averaged system state over time.
        - If "individuals": A list or array collection containing the full history 
          of each of the `n` individual stochastic runs.
    """
    assert output_type in ["mean","individuals"], "output_type must be \`mean\` or \`individuals\`"
    
    time0 = time.time()
    
    model_mean, models = simulate_nonlinear_migration_ABM(p,
                                             migration_step_pulling, 
                                             "pulling", 
                                             simulation_type,
                                             dens=dens,
                                             perc=perc,
                                             n=n, 
                                             T_end = 2.0,
                                             delay = delay)
    
    time_elapsed = time.time() - time0
    
    if output_type == "mean":
        result = model_mean
    elif output_type == "individuals":
        result = models
    
    #save updated results
    if resultsDir is not None:
        save_data(p, result, time_elapsed, resultsDir)
    
    return result

def save_data(p, result, time_elapsed, resultsDir):
    """
    Serializes and saves simulation results and metadata to a NumPy binary file.

    Unpacks the model parameters to construct a filename. 
    It then packages the raw results, parameter array, and execution 
    profile timing into a dictionary and writes it to disk as a `.npy` file.

    Parameters
    ----------
    p : tuple or list of float
        A collection of 5 model parameters ordered exactly as:
        `(rm, rint, rp, a0, a1)`.
        - `rm` : Migration rate.
        - `rint` : Pulling probability.
        - `rp` : Proliferation rate.
        - `a0` : Delay parameter 1.
        - `a1` : Delay parameter 2.
    result : numpy.ndarray or dict
        The data structure containing the processed results or individual 
        trajectories from the ABM simulation.
    time_elapsed : float
        The execution time (in seconds) taken to complete the simulation.
    resultsDir : str
        The target directory path where the resulting file will be stored.

    Returns
    -------
    None
    """
    rm, rint, rp, a0, a1 = p
    paramsStr = f"rm_{rm}_pint_{rint}_rp_{rp}_alpha0_{a0}_alpha1_{a1}"
    
    data_save = {}
    data_save['result'] = result
    data_save['param'] = p
    data_save['time_elapsed'] = time_elapsed
    np.save(f"{resultsDir}/{paramsStr}.npy",data_save)

def simulate_nonlinear_migration_ABM(params, migration_rules, interaction_string, data_type, dens=None, perc=None, n=5,T_end = 1000.0, save=None, delay = False):
    """
    Executes an ensemble of Agent-Based Model (ABM) simulations and computes the mean behavior.

    Runs `n` independent stochastic realizations of a cell migration/proliferation 
    simulation via `cell_interaction_ABM`. It aggregates the individual trajectory 
    results, computes the ensemble mean profile across space and time, and 
    optionally saves a comprehensive dataset dictionary to a `.npy` binary file.

    Parameters
    ----------
    params : tuple of float
        A 5-element tuple of model constants/coefficients ordered exactly as:
        `(Pm, Pint, Pp, a0, a1)`.
        - `Pm` : Base migration probability parameter.
        - `Pint` : Pulling or adhesion interaction strength parameter.
        - `Pp` : Proliferation rate coefficient.
        - `a0` : Intercept/bias parameter for the delay function.
        - `a1` : Slope parameter for the delay function.
    migration_rules : callable
        A function implementing the movement and spatial grid updates. Here, it uses 
        `migration_step_pulling`.
    interaction_string : {"pulling", "adhesion"}
        A label declaring the semantic type of interaction. Affects internal 
        dictionary key naming and the output file name string structure. Here, uses "pulling"
    data_type : {"WH", "artificial"}
        The configuration layout strategy selector used to construct the initial conditions.
    dens : float, optional
        The initial population density identifier passed down to the Wound Healing 
        initializer. Defaults to None.
    perc : float, optional
        The initial grid occupancy ratio passed down to the artificial layout initializer. 
        Defaults to None.
    n : int, optional
        The total number of independent stochastic simulations to run for ensemble averaging. 
        Defaults to 5.
    T_end : float, optional
        The maximum termination time horizon for each simulation run. Defaults to 1000.0.
    save : any, optional
        If not None (e.g., passing `True` or a path switch), triggers the serialization 
        block that exports parameters, meshgrid data, coordinates, and means to disk. 
        Defaults to None.
    delay : bool, optional
        Toggles whether the dynamics are delayed. Defaults to False.

    Returns
    -------
    C_mean : numpy.ndarray
        A 2D array of shape `(xn, 5)` representing the average agent spatial profile 
        over time, averaged over all `n` realizations.
    Cs : numpy.ndarray
        A 3D array of shape `(n, xn, 5)` holding the individual interpolated 
        density profile arrays from every simulation run.
    """
    
    Pm   = np.round(params[0],5)
    Pint = np.round(params[1],5)
    Pp   = np.round(params[2],5)
    
    assert data_type in ["WH","artificial"], "data_type must equal \"WH\" or \"artificial\"."
    
    #Initialize list containing all individual simulations
    Cs = []
    #Initialize dictionary storing all saved information
    data = {}
    #Used to track computation time
    computationTimeAll0 = time.time()
    
    for i in np.arange(n):
        
        C_out, t_out, x_out, plot_list = cell_interaction_ABM(params,
                                                              migration_rules = migration_rules,
                                                              T_end=T_end,
                                                              perc=perc,
                                                              simulation_type = data_type,
                                                              dens=dens,
                                                              delay=delay)
        #store computed information
        Cs.append(C_out)
        data[f'c{i}'] = C_out
    
    #record time of all simulations
    computationTimeAllFinal = time.time() - computationTimeAll0
    #convert to np arrays, get mean data
    Cs = np.array(Cs)    
    C_mean = np.mean(Cs,axis=0)

    T,X = np.meshgrid(t_out,x_out)

    if interaction_string == "pulling":
        Pint_string = "Ppull"
    if interaction_string == "adhesion":
        Pint_string = "Padh"
    
    if save is not None:
        data['Pm'] = Pm
        data[Pint_string] = Pint
        data['Pp'] = Pp

        data['x'] = x_out[:,None]
        data['X'] = X
        data['t'] = t_out[:,None].T
        data['T'] = T
        data['C'] = C_mean
        data['time_all'] = computationTimeAllFinal
        np.save(f"../data/simple_{interaction_string}_mean_{n}_Pm_{Pm}_{Pint_string}_{Pint}_Pp_{Pp}.npy",data)  
        
    return C_mean, Cs    
    
def Tau(t,a0,a1):
    """
    Computes the delay activation state over time using a logistic curve.

    This function determines whether a time-delay mechanism is active based on 
    the independent state variable `t`. It maps the input to a factor between 
    0 and 1, where:
    - Values close to 0 mean the delay factor is fully in effect (suppressed transition).
    - Larger values approaching 1 mean the delay phase has passed (transition enabled).

    Parameters
    ----------
    t : float or numpy.ndarray
        The independent variable (typically current time) used 
        to evaluate the delay progression.
    a0 : float
        The intercept/bias parameter. Controls the horizontal shift.
    a1 : float
        The slope parameter. Controls the rate of transition; a larger value 
        creates a sharper, more sudden transition out of the delay phase.

    Returns
    -------
    float or numpy.ndarray
        The activation multiplier bounded between 0.0 and 1.0, representing 
        the current intensity of the delay mechanism.

    """
    exponent = -(a1*t + a0)

    return 1/(1+np.exp(exponent))

def no_delay(t,a0,a1):
    """
    Simulates a bypassed delay state by returning a constant activation of 1.0.

    Serves as a baseline or control function that matches the signature of 
    `Tau`.

    Parameters
    ----------
    t : float or numpy.ndarray
        The independent variable (e.g., time or state) used to determine 
        the shape of the output data structure.
    a0 : float
        Unused parameter. Maintained to ensure signature compatibility 
        with `Tau`.
    a1 : float
        Unused parameter. Maintained to ensure signature compatibility 
        with `Tau`.

    Returns
    -------
    delay : float or numpy.ndarray
        A scalar value of 1.0 if `t` is a scalar, or a NumPy array filled with 
        1.0 matching the exact shape of `t`.
    """
    if np.isscalar(t):
        delay = 1.0
    else:
        delay = np.ones(t.shape)
    return delay


def cell_interaction_ABM(params,migration_rules,simulation_type,dens=None,T_end=5.0,perc=None, delay=False):
    """
    Simulates cell migration and proliferation dynamics using a 2D lattice Agent-Based Model (ABM).

    This function executes a Gillespie-style stochastic simulation tracking discrete agent 
    behaviors (movement and proliferation) on a 2D grid. The global propensity function can be 
    modulated by a time-dependent delay factor (`Tau` or `no_delay`). Snapshots of the grid 
    are collected at equal temporal intervals, and final spatial profiles are linearly interpolated 
    onto an equispaced spatial-temporal grid.

    Parameters
    ----------
    params : tuple of float
        A 5-element tuple containing the model coefficients ordered as:
        `(rm, rint, Pp, a0, a1)`.
        - `rm` : Base migration rate per agent.
        - `rint` : Interaction rate modifier (e.g., pulling probability).
        - `Pp` : Proliferation rate coefficient per agent.
        - `a0` : Delay parameter 1.
        - `a1` : Delay parameter 2.
    migration_rules : callable
        A function specifying the rules of structural movement. Typically handles 
        neighbor checks and updates grid arrays. Here, uses 
        `migration_step_pulling`.
    simulation_type : {"WH", "artificial"}
        The configuration style for initial condition generation. "WH" usually 
        denotes a Wound Healing assay setup, "artificial" is for the artificial datasets
    dens : float, optional
        Initial spatial density descriptor passed to `IC_initialization`. Defaults to None.
    T_end : float, optional
        The final termination time-horizon for the stochastic simulation loop. Defaults to 5.0.
    perc : float, optional
        Initial occupancy percentage of the lattice sites passed to `IC_initialization` for artifical data.
        Defaults to None.
    delay : bool, optional
        Determines if a time-lag mechanics factor scales propensity. If True, utilizes `Tau`; 
        if False, utilizes `no_delay`. Defaults to False.

    Returns
    -------
    A_out : numpy.ndarray
        A 2D array of shape `(xn, 5)` containing the interpolated cross-sectional spatial agent profiles 
        evaluated at 5 equispaced time points.
    t_out : numpy.ndarray
        A 1D array of length 5 representing the equispaced time coordinates from `0` to `T_end`.
    x_out : numpy.ndarray
        A 1D integer array of length `xn` representing individual lattice site coordinates along the x-axis.
    plot_list : list of numpy.ndarray
        A list of full 2D binary matrix snapshots capturing the spatial grid configurations at specific 
        simulation intervals.
    """
    
    rm, rint, Pp, a0, a1 = params
    
    assert np.all([isinstance(p, Number) for p in params ]), "All parameters must be numbers"
    
    assert simulation_type in ["WH","artificial"], "simulation_type must equal \"WH\" or \"artifical\"."
    
    A = IC_initialization(simulation_type=simulation_type, dens=dens, percs=perc)
    
    yn, xn = A.shape
    
    #count number of occupied lattice sites.
    A_num = np.sum(A==1)

    #initialize time
    t = 0

    #track time, agent proportions, and snapshots of ABM in these lists
    t_list = [t]
    A_list = [A_num]
    A_profiles = [np.sum(A,axis=0)]
    plot_list = [copy.deepcopy(A)]
    
    #number of snapshots saved so far in plot_list
    image_count = 1
    
    if delay is False:
        delay_function = no_delay
    elif delay is True:
        delay_function = Tau
        
    count = 0
    while t_list[-1] < T_end:

        ### late night thoughts 11-28-24: delay = 1 if false. else delay = Tau
        ### a = delay*(rm*A_num + Pp*A_num)
        ### later, Action <= delay*rm*A_num
        
        delay = delay_function(t,a0,a1)
        a = delay*(rm*A_num + Pp*A_num)
            
                
        tau = -np.log(np.random.uniform())/a
        t += tau

        Action = a*np.random.uniform()

        if Action <= delay*rm*A_num:
            #agent movement
            A = migration_trimolecular_reaction(A,
                                                params = (rint,),
                                                states = [1],
                                                migration_rules = migration_rules)
        elif Action <= delay*(rm*A_num + Pp*A_num):
            #proliferation
            A = bimolecular_reaction(A,
                                     state_1_alpha=[1],
                                     state_1_beta=0,
                                     state_2_alpha=1,
                                     state_2_beta=1)
            
        #count number of occupied lattice sites
        A_num = np.sum(A==1)
        
        #only save every 100 steps
        if count%100 == 0:
            #append information to lists
            t_list.append(t)
            A_list.append(A_num)
            A_profiles.append(np.sum(A,axis=0))
        
            if (t_list[-2] < image_count*T_end/100 and t_list[-1] >= image_count*T_end/100): 
                plot_list.append(copy.deepcopy(A))
                image_count+=1
        count+=1

    #interpolation to equispace grid
    x_out = np.arange(xn)
    t_out = np.linspace(0,T_end,5)
    t_list = np.array(t_list)
    A_profiles = np.array(A_profiles)
    
    X,T = np.meshgrid(x_out,t_list,indexing="ij")
    
    f = LinearNDInterpolator(list(zip(X.reshape(-1), T.reshape(-1))), A_profiles.T.reshape(-1))
    
    X_out,T_out = np.meshgrid(x_out,t_out,indexing="ij")
    shape = X_out.shape
    A_out = f(list(zip(X_out.reshape(-1),T_out.reshape(-1),))).reshape(shape)

    return A_out, t_out, x_out, plot_list

def bimolecular_reaction(A,state_1_alpha,state_1_beta,state_2_alpha=None,state_2_beta=None):
    """
    Simulates a stochastic neighbor-dependent bimolecular reaction on a 2D lattice.

    Selects a random focal agent belonging to any of the valid states listed in 
    `state_1_alpha`. It then randomly selects one of its four orthogonal neighbors 
    (Up, Down, Left, Right). If the neighbor's current state matches `state_1_beta`, 
    the reaction condition is met: the focal position is updated to `state_2_alpha` 
    and the neighbor's position is updated to `state_2_beta`.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D integer array representing the spatial lattice state grid.
    state_1_alpha : list or array_like of int
        The permissible state values for the initial focal agent triggering the 
        reaction.
    state_1_beta : int
        The mandatory target state value required of the selected neighbor cell 
        to successfully execute the reaction (e.g., `0` for an empty site).
    state_2_alpha : int, optional
        The updated state assigned to the original focal agent's coordinates if 
        the reaction succeeds. If None, it defaults to the neighbor's pre-reaction 
        state.
    state_2_beta : int, optional
        The updated state assigned to the selected neighbor's coordinates if the 
        reaction succeeds. If None, it defaults to the focal agent's pre-reaction 
        state.

    Returns
    -------
    A : numpy.ndarray
        The modified 2D lattice state grid after evaluating the reaction attempt.
    """
    yn,xn = A.shape

    # Select Random agent
    for i,state in enumerate(state_1_alpha):
        agent_loc_tmp = A == state
        if i == 0:
            agent_loc = agent_loc_tmp
        else:
            agent_loc = np.logical_or(agent_loc,agent_loc_tmp)
    agent_loc = np.where(agent_loc)
    agent_ind = np.random.permutation(len(agent_loc[0]))[0]
    loc = (agent_loc[0][agent_ind],agent_loc[1][agent_ind])
    
    if state_2_beta is None:
        state_2_beta = A[loc]


    ### Determine direction
    dir_select = np.ceil(np.random.uniform(high=4.0))

    #right
    if dir_select == 1 and loc[0] < yn-1:
        if A[(loc[0]+1,loc[1])] == state_1_beta:

            if state_2_alpha is None:
                state_2_alpha = A[(loc[0]+1,loc[1])]
    
            A[(loc[0],loc[1])] = state_2_alpha
            A[(loc[0]+1,loc[1])] = state_2_beta

    #left
    elif dir_select == 2 and loc[0]>0:
        if A[(loc[0]-1,loc[1])] == state_1_beta:

            if state_2_alpha is None:
                state_2_alpha = A[(loc[0]-1,loc[1])]

            A[(loc[0],loc[1])] = state_2_alpha
            A[(loc[0]-1,loc[1])] = state_2_beta

    #up        
    elif dir_select == 3 and loc[1]<xn-1:
        if A[(loc[0],loc[1]+1)] == state_1_beta:

            if state_2_alpha is None:
                state_2_alpha = A[(loc[0],loc[1]+1)]

            A[(loc[0],loc[1])] = state_2_alpha
            A[(loc[0],loc[1]+1)] = state_2_beta


    #down
    elif dir_select == 4 and loc[1]>0:
        if A[(loc[0],loc[1]-1)] == state_1_beta:

            if state_2_alpha is None:
                state_2_alpha = A[(loc[0],loc[1]-1)]

            A[(loc[0],loc[1])] = state_2_alpha
            A[(loc[0],loc[1]-1)] = state_2_beta

    return A


def migration_step(A,loc,migration_loc):
    """
    Perform a migration step

    Args:
        A (numpy.ndarray): An array representing the state of agents, where 0 represents an empty space and 1 represents an agent.
        loc (tuple): The location of the current agent.
        migration_loc (tuple): The location to which the agent is migrating.
        
    Returns:
        A (numpy.ndarray): The updated state of agents after the migration step.
    """
    state = A[loc]
    
    if A[migration_loc] == 0:
            #agent moves into migration_loc
            A[migration_loc] = state
            A[loc] = 0                                
    return A

def migration_step_pulling(A,params,loc,migration_loc,neighbor_loc):
    """
    Perform a single step of the Pulling ABM.

    Args:
        A (numpy.ndarray): An array representing the state of agents, where 0 represents an empty space and 1 represents an agent.
        params (list): A list containing a single parameter, Ppull, which is the probability of a pulling event.
        loc (tuple): The location of the current agent.
        migration_loc (tuple): The location to which the agent is migrating.
        neighbor_loc (tuple): The location of the neighboring agent.

    Returns:
        A (numpy.ndarray): The updated state of agents after the migration step.

    The function performs Rules A and B based on A[migration_loc] and A[neighbor_loc]
    """
    
    assert len(params)==1
    Ppull = params[0]
    
    #Can only move if chosen migration location is empty
    if A[migration_loc] == 0:
        
        #move to migration location
        A[migration_loc] = 1
        
        if A[neighbor_loc] == 1 and np.random.uniform() < Ppull:
            #Rule B -- successful pulling event
            A[loc] = 1
            A[neighbor_loc] = 0
        else:
            #Either Rule A (if A[neighbor_loc] == 0)
            #or Rule B -- unsuccessful pulling event
            A[loc] = 0            
                    
    return A

def migration_trimolecular_reaction(A, states, params, migration_rules):

    """
    Perform a trimolecular reaction involving agent migration and interaction.

    Args:
        A (numpy.ndarray): 2D array representing the lattice with agents.
        states (list): List of agent states involved in the reaction. [1] denotes adhesive agent, 2 denotes pulling agents
        params (tuple): A tuple containing reaction-specific parameters.
        migration_rules (callable): Function defining the interaction step.

    Returns:
        A (numpy.ndarray): Updated lattice array after the reaction.
        
    """
    
    assert type(params) == tuple
    
    yn, xn = A.shape

    # Select Random agent
    for i,state in enumerate(states):
        agent_loc_tmp = A == state
        if i == 0:
            agent_loc = agent_loc_tmp
        else:
            agent_loc = np.logical_or(agent_loc,agent_loc_tmp)
    agent_loc = np.where(agent_loc)
    agent_ind = np.random.permutation(len(agent_loc[0]))[0]
    loc = (agent_loc[0][agent_ind],agent_loc[1][agent_ind])
    
    ### Determine direction
    dir_select = np.ceil(np.random.uniform(high=4.0))
    
    #downward
    if dir_select == 1 and loc[0] < yn-1:
        dy = 1
        dx = 0
        
        if loc[0] > 0:
            #perform interaction-based migration in interior
            reaction = "interaction"
        else:
            #perform simple migration at boundary (no neighboring lattice site)
            reaction = "migration"
    
    #upward
    elif dir_select == 2 and loc[0] > 0:
        dy = -1
        dx = 0
        
        if loc[0] < yn-1: #interior
            reaction = "interaction"
        else:
            reaction = "migration"
        
    #rightward    
    elif dir_select == 3 and loc[1] < xn-1:
        dy = 0
        dx = 1
        
        if loc[1] > 0: #interior
            reaction = "interaction"
        else:
            reaction = "migration"
            
    #leftward        
    elif dir_select == 4 and loc[1] > 0: #left
        dy = 0
        dx = -1
        
        if loc[1] < xn-1: #interior
            reaction = "interaction"
        else:
            reaction = "migration"
            
    else:
        dy = 0
        dx = 0
        reaction = "aborted"
        
    migration_loc = (loc[0]+dy,loc[1]+dx)
    neighbor_loc  = (loc[0]-dy,loc[1]-dx)

    if reaction == "interaction":
        A = migration_rules(A,params,loc,migration_loc,neighbor_loc)       
    elif reaction == "migration":
        A = migration_step(A,loc,migration_loc)
    elif reaction == "aborted":
        pass

    return A

def IC_initialization(simulation_type, dens = None, percs = None):
    """
    Routes initial condition (IC) generation for the ABM based on simulation type.

    Selects the appropriate matrix initialization function based on the scenario. 
    "artificial" setups use preset grid dimensions, while "WH" (Wound Healing) configurations 
    utilize experimental or continuous density definitions.

    Parameters
    ----------
    simulation_type : {"artificial", "WH"}
        A string key defining which setup scenario structure to build.
    dens : float, optional
        The initial population density scalar required by the Wound Healing 
        initializer. Ignored if `simulation_type` is "artificial". Defaults to None.
    percs : float or array_like of float, optional
        Occupancy percentage boundaries or targets used to generate the 
        artificial matrix grid. Ignored if `simulation_type` is "WH". Defaults to None.

    Returns
    -------
    A : numpy.ndarray
        A 2D binary integer array representing the freshly initialized grid 
        state environment.
    """
    if simulation_type == "artificial":
        return ABM_IC_initialization(percs=percs,xn=74,yn=60)
    elif simulation_type == "WH":
        return WH_IC_initialization(simulation_type, dens)
    else:
        raise Exception("simulation_type must equal \"WH\" or \"artificial\".")
    
    return A

def WH_IC_initialization(simulation_type, dens):
    """
    Generates a 2D grid matrix initial condition for a Wound Healing assay simulation.

    Loads experimental baseline scratch assay data for a specific cell density, 
    extracts the first observed time profile, and doubles its spatial tracking 
    resolution. This expanded 1D distribution profile is then randomly projected 
    into a discrete 2D binary state matrix environment of fixed dimension 
    (60 rows by 74 columns).

    Parameters
    ----------
    simulation_type : str
        The identifier string passed to `load_data` to locate the source Wound 
        Healing data collection repository.
    dens : int
        The initial scratch assay experimental seeding density. Must belong 
        strictly to the even intervals between 10 and 20 inclusive: 
        `{10, 12, 14, 16, 18, 20}`.

    Returns
    -------
    A : numpy.ndarray
        A 2D integer matrix of shape `(60, 74)` representing the initialized 
        wound healing lattice state where 1s designate occupied grid spaces.
    """
    assert dens in np.arange(10,22,2)
    
    WH_data = load_data(simulation_type, dens=dens)
    initial_condition = WH_data[:,0]
    
    num_rows = 60
    num_cols = 74
    
    assert num_cols == 2*np.size(initial_condition), "num_cols must be double the size of the IC"
    
    data_num_d = np.array([int(x) for x in initial_condition])
    
    data_num_m = convert_num_d_to_num_m(data_num_d)
    
    A = convert_num_m_to_ABM(data_num_m, num_rows = num_rows, num_cols = num_cols)
    
    return A

def ABM_IC_initialization(percs=[0.75], xn=200, yn=40):
    """
    Initializes a 2D lattice domain with agents populated in the exterior quarters.

    Creates an artificial initial condition by defining an empty center corridor 
    (the middle 50% along the x-axis) and populating the outer left and right quarters 
    of the grid. For each active column, a specified percentage of rows are randomly 
    selected and filled. Supports multi-class agent populations if a list of 
    percentages is provided.

    Parameters
    ----------
    percs : float or list of float, optional
        The fraction(s) of row elements (`yn`) to populate in each active column. 
        If a list is provided, each value represents the density proportion for 
        a distinct agent class (assigned sequentially as integer states `1, 2, ...`). 
        Defaults to `[0.75]` for one population.
    xn : int, optional
        The total number of lattice columns along the horizontal x-axis. 
        Defaults to 200.
    yn : int, optional
        The total number of lattice rows along the vertical y-axis. 
        Defaults to 40.

    Returns
    -------
    A : numpy.ndarray
        A 2D integer matrix of shape `(yn, xn)` containing the generated 
        initial configuration state layout.
    """
    A = np.zeros((yn,xn))

    #convert number to iterable list.
    if isinstance(percs,Number):
        percs = [percs]
    
    #populate first and last quarter of the domain
    xn_exterior = np.hstack([np.arange(0,int(.25*xn)),
                         np.arange(int(.75*xn),xn,dtype=int)[1:]])

    for i in xn_exterior:
        #Place agents in perc% of y-locations for each x location in interior
        perm = np.random.permutation(yn)

        index_lb = 0
        for j in np.arange(len(percs)):
            index_ub = int(index_lb + percs[j]*yn)
            yn_perm = perm[index_lb : index_ub]
            A[yn_perm,i] = j+1
            index_lb = index_ub
    
    return A    
