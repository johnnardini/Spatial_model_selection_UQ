#!/bin/bash

#SBATCH --chdir=./                     # Set the working directory
#SBATCH --mail-user=nardinij@tcnj.edu      # Who to send emails to
#SBATCH --mail-type=none                  # Send emails on start, end and failure
#SBATCH --account=nlscience
#SBATCH --job-name=Timing       # Name to show in the job queue
#SBATCH --output=out_files/job.%j.out              # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks=100                   # Total number of mpi tasks requested
#SBATCH --partition=normal
#SBATCH --time=0-4:00:00                  # Max run time (days-hh:mm:ss) ... adjust as necessary

source activate WH_ABM

export OMPI_MCA_btl=tcp,vader,self
export OMPI_MCA_mtl=psm2
export OMPI_MCA_pml=ob1

module list


echo "Starting on "`date`

mpirun python time_ABM_simulations.py
