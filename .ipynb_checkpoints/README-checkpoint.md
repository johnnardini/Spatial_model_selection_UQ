# Spatial Model Selection and Uncertainty Quantification

README for: "Spatial Model Selection and Uncertainty Quantification: Comparing Continuous and Discrete Wound Healing Models"  by John T. Nardini and Jana L. Gevertz.

System Requirements:
====================
* Python 3
* High performance computing is recommended for replicating the ABC pipeline.
* Open MPI
* ABC data generation was performed using 20-50 mpi tasks.

Installation:
=============
Following [The Good Research Code Handbook](https://goodresearch.dev/setup), you can use pip to install the `src` package for this project. Once you have downloaded this code, you can install this package in the main directory directory by entering 
```
pip install -e .
```

Running the ABC pipeline and notebooks:
=========================================
See the README.md files in `scripts/ABC_pipeline` to see how to run the ABC pipeline. All notebooks in the other directories in `scripts/` can already be run using the results stored in `results/` (the directores in `results/prior_sampling/` are empty, but can be populated by running the ABC pipeline, which takes seconds for the PDE models and days for the ABMs).

Contact:
========
Please contact John Nardini at nardinij@tcnj.edu if you have any questions.