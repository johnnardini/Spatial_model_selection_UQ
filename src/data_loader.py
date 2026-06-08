import numpy as np
import xlrd

def load_data(data_type,*,dens=None,simulation=None):
    """
    Load experimental wound-healing data or artificial data.

    This function extracts profile data across time points. For experimental Wound 
    Healing ('WH') data, it parses a specified Excel spreadsheet, extracts mean values 
    from designated rows. 
    For synthetic data ('artificial'), it pulls pre-calculated profile grids directly 
    from a serialized NumPy array file.

    Parameters
    ----------
    data_type : str
        The target dataset class to load. Options are:
        - 'WH' : Real-world experimental wound healing assay data.
        - 'artificial' : Synthetic data generated via an Agent-Based Model (ABM).
    dens : int, optional
        The specific experimental cell density setup. Required if `data_type='WH'`.
    simulation : int, optional
        The discrete simulation trajectory number. Required if `data_type='artificial'`.

    Returns
    -------
    Data : numpy.ndarray
        A 2D array of shape `(n_spatial_nodes, n_time_steps)` containing the density 
        values, transposed so rows represent space and columns represent time increments.
    """
    if data_type == "WH":
    
        assert dens is not None, "Must specify dens for WH data"
        wb = xlrd.open_workbook(f"../../data/WH_data/1-s2.0-S0022519315005676-mmc{dens}.xls")

        WH_data = []
        for i in range(5):
            ws = wb.sheet_by_index(i)

            data = np.array(ws.row_values(9)[2:])
            
            WH_data.append(data)
            
        Data = np.array(WH_data).T
        
    elif data_type == "artificial":
        
        assert simulation is not None, "Must specify simulaiton for ABM data"
        mat = np.load(f"../../data/ABM_data/ABM_sim_{simulation}.npy",
                     allow_pickle=True).item()
        
        Data = mat['C']
        
    else:
        assert False, "data_type must be WH or artificial"
    
    return Data