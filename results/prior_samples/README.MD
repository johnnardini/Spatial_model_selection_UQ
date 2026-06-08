# Prior sampling directories

This directory contains the prior sampling directories from the ABC pipeline under various scenarios. Note that these directories are empty, but will be populated after running `MPI_uniform_sampling.py` in `../../scripts/ABC_pipeline`.

## File Naming Convention Breakdown
Directory paths are programmatically constructed using the following token pattern:

> `ABC_{parameter_names}_uniform_sampling_{model_type}_{data_type}_{condition}`

* **`{parameter_names}`**: Parameters varied during ABC. 
    * `rm_rp` (Model 1),
    * `rm_pint_rp` (Model 2),
    * `rm_rp_a0_a1` (Model 3), or 
    * `rm_pint_rp_a0_a1` (Model 4).
* **`{model_type}`**: The modeling modality. 
    * `PDE` or
    * `ABM`.
* **`{data_type}`**: The origin type of the dataset. 
    * `WH` (Wound Healing data) or 
    * `artificial` (Synthetic data).
* **`{condition}`**: The specific profile ID being evaluated. 
    * `dens_10` (for Wound Healing Dataset 10K), 
    * `dens_12`(for Wound Healing Dataset 12K),
    * ..., 
    * `dens_20` (for Wound Healing Dataset 20K) or 
    * `simulation_0` (for Artificial Dataset 1), 
    * `simulation_1` (for Artificial Dataset 2), 
    * ..., 
    * `simulation_7` (for Artificial Dataset 8).
* **`{prediction_type}`**: The temporal constraint of the objective function. 
    * ` ` (empty) uses all data points during matching.
    * `_pred_final` executes out-of-sample forecasting (excludes the final time point).