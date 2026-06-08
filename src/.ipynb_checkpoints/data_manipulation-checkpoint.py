import numpy as np

def convert_model_to_data_shape(model):
    """
    Transforms model out from model dimensions to data dimensions.

    Parameters
    ----------
    model : numpy.ndarray
        A 2D array-like structure where columns represent timepoints and
        rows represent space points. Must have 5 columns.

    Returns
    -------
    numpy.ndarray
        A transposed 2D array of the converted data with shape (N, 5).
    """
    model_d = []
    for i in range(5):
        model_d.append(convert_num_m_to_num_d(model[:,i]))
    model_d = np.array(model_d).T

    return model_d


def convert_num_m_to_num_d(num_m):
    """
    Converts a 1D model-space vector to a data-space vector via block-diagonal multiplication.

    This is achieved by summing adjacent pairs of elements, reducing the length of the vector in half.
    This is done by constructing a block-diagonal transformation matrix where each block is 
    [[1.0, 1.0]], then multiplying this matrix by the input vector `num_m`. 
    

    Parameters
    ----------
    num_m : numpy.ndarray
        A 1D array representing the model-space vector. The size of the array 
        should ideally be even to ensure exact pairing of blocks.

    Returns
    -------
    numpy.ndarray
        The transformed data-space vector, downsampled to a size of 
        `len(num_m) // 2`.
    """
    
    assert num_m.ndim == 1, "num_m must be 1-d"
    
    num_cols = np.size(num_m)
    
    Xblock = np.array([[1.0, 1.0]])
    
    mblock, nblock = Xblock.shape
    
    X = np.copy(Xblock)
    
    for _ in range(num_cols//2 - 1):
        mtmp, ntmp = X.shape

        X = np.block([
                        [X, np.zeros((mtmp,nblock))],
                        [np.zeros((mblock,ntmp)), Xblock]
                    ])
        
    return np.matmul(X,num_m)

def convert_num_d_to_num_m(num_d):
    """
    Expands data-space counts into model-space counts with random remainder allocation.

    Takes a 1D array of counts, doubles its length, and splits each original 
    count evenly between two adjacent elements. If an original count is odd, 
    the truncated fractional remainder (the missing agent) is randomly assigned 
    to one of the two slots with a 50% probability.

    Parameters
    ----------
    num_d : numpy.ndarray
        A 1D array containing the original counts (e.g., data-space values) 
        to be expanded and split.

    Returns
    -------
    num_agents : numpy.ndarray
        A 1D integer array of size `2 * len(num_d)` containing the distributed 
        agent counts.
    """
    
    assert num_d.ndim == 1, "num_d must be 1-d"
    
    num_cols = 2*np.size(num_d)
    
    num_agents = np.zeros(num_cols, dtype=int)
    
    for i, num in enumerate(num_d):
        
        num_agents[2*i] = int(0.5*num)
        num_agents[2*i+1] = int(0.5*num)
        
        #if we're missing an agent, randomly add it to one column
        if (num_agents[2*i] + num_agents[2*i+1]) < num:
            if np.random.uniform() < 0.5:
                num_agents[2*i] += 1
            else:
                num_agents[2*i+1] += 1

    return num_agents

def convert_num_m_to_ABM(num_m,num_rows,num_cols=None):            
    """
    Initializes an agent-based model (ABM) binary state matrix from population counts.

    Creates a 2D grid `U` of shape `(num_rows, num_cols)` representing agent 
    occupancy. For each column $j$, the function randomly distributes a unique set 
    of agents across the available rows without replacement, based on the population 
    count specified in `num_m[j]`. Occupied positions are marked with a 1, 
    while empty positions remain 0.

    Parameters
    ----------
    num_m : numpy.ndarray
        A 1D array of integers, where each element specifies the number of 
        agents to randomly place in the corresponding column of the grid.
    num_rows : int
        The number of rows in the resulting matrix, representing the total 
        available capacity or distinct states/locations per column.
    num_cols : int, optional
        The number of columns in the resulting matrix. If None (default), 
        it defaults to the size of `num_m`.

    Returns
    -------
    U : numpy.ndarray
        A 2D integer matrix of shape `(num_rows, num_cols)` containing 
        randomly distributed 1s (agents) and 0s (empty spaces).
    """
    assert num_m.ndim == 1, "num_m must be 1-d"
    
    if num_cols == None:
        num_cols = np.size(num_m)
    
    U = np.zeros((num_rows,num_cols),dtype=int)

    for j in range(num_cols):

        agent_locs = np.random.choice(range(num_rows),size=num_m[j],replace=False)

        U[agent_locs,j] = 1
        
    return U