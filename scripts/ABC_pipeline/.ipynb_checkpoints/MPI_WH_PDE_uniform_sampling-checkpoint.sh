#!/bin/bash

#SBATCH --chdir=./                     # Set the working directory
#SBATCH --mail-user=user@email.edu      # Who to send emails to
#SBATCH --array=0-5
#SBATCH --output=out_files/job.%j.out              # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks=47                   # Total number of mpi tasks requested
#SBATCH --partition=long
#SBATCH --time=29-00:00:00                  # Max run time (days-hh:mm:ss) ... adjust as necessary

MODEL_TYPE="PDE"
DATA_TYPE="WH"
models=("rm_rp" "rm_pint_rp" "rm_rp_a0_a1" "rm_pint_rp_a0_a1")

echo "Starting on "`date`

for model in "${models[@]}"
do
    echo "starting model $model"

    PRED_TYPE="fit_all"
    #prior sampling
    mpirun python MPI_uniform_sampling.py "$model" "$MODEL_TYPE" "$DATA_TYPE" "$SLURM_ARRAY_TASK_ID" "$PRED_TYPE"
    #histogram generation    
    python ABC_kernel_computation.py "$model" "$MODEL_TYPE" "$DATA_TYPE" "$SLURM_ARRAY_TASK_ID" "$PRED_TYPE"
    #posterior information    
    python get_posterior_mean_info.py "$model" "$MODEL_TYPE" "$DATA_TYPE" "$SLURM_ARRAY_TASK_ID" "$PRED_TYPE"
    
    PRED_TYPE="pred_final"
    #histogram generation     
    python ABC_kernel_computation.py "$model" "$MODEL_TYPE" "$DATA_TYPE" "$SLURM_ARRAY_TASK_ID" "$PRED_TYPE"
    #posterior information    
    python get_posterior_mean_info.py "$model" "$MODEL_TYPE" "$DATA_TYPE" "$SLURM_ARRAY_TASK_ID" "$PRED_TYPE"
    
    echo "Finished $model on "`date`
done