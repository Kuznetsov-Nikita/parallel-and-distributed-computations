#!/bin/bash
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=RT_study
#SBATCH --job-name=MIPT-PD-Homework
#SBATCH --comment="Run mpi from config"
#SBATCH --output=output.txt
#SBATCH --error=error.txt
mpiexec ./a.out $1
