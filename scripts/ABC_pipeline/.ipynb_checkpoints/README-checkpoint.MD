# ABC Pipeline simulation

This directory contains the `.py` files necessary to run the ABC pipeline.

## File breakdown
* **`MPI_uniform_sampling.py`**: Runs the prior sampling routine and saves the results in `../../results/prior_sampling/`
* **`ABC_histogram_computation.py`**: Imports the pre-computed prior samples, computes the posterior histogram, and saves the results in `../../results/histograms/`
* **`get_posterior_mean_info.py`**: Imports the pre-computed histogram and computes the mean and MAP. From that, it calculates the model (ABM and PDE) at these two parameter values and saves the results in `../../results/posterior_sampling/`

## Running the files
The four `.sh` files can be used to run the pipeline for the ABC/PDE models on the Artificial/Wound healing data. This code can be run as using `sbatch`, for example:

```
sbatch MPI_artificial_PDE_uniform_sampling.sh
```

Note that while the PDE codes complete in seconds, the ABM codes take on the order of days to compute.