# Prior sampling directories

This directory contains the prior sampling directories from the ABC pipeline under various scenarios. Note that these directories are empty, but will be populated after running `MPI_uniform_sampling.py` in `../../scripts/ABC_pipeline`.

## File Naming Convention Breakdown
Directory paths are programmatically constructed using the following token pattern:

> `ABC_{parameter_names}_uniform_sampling_{model_type}_{data_type}_{condition}`

* **`{parameter_names}`**: Parameters varied during ABC. 
    * `rm_rp` (Model 1), `rm_pint_rp` (Model 2), `rm_rp_a0_a1` (Model 3), or `rm_pint_rp_a0_a1` (Model 4).
* **`{model_type}`**: The modeling modality. 
    * `PDE` or `ABM`.
* **`{data_type}`**: The origin type of the dataset. 
    * `WH` (Wound Healing data) or `artificial` (Synthetic ABM baseline).
* **`{condition}`**: The specific profile ID being evaluated. 
    * `dens_10`, `dens_12`, ..., `dens_20` (for `WH`) or `simulation_0`, `simulation_1`, ..., `simulation_7` (for `artificial`).