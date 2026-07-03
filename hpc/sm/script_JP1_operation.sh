#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm # c23mm or devel # devel only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=96             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=op_100
#SBATCH --output=output_op_100.%J.txt
### #SBATCH --error=error_op.%J.txt
#SBATCH --time=24:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 50 \
--case JP1 \
--project project_ID10_100.yaml \
--input_csv_name operational_opt_var_ts_lhs_sample_100000.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name op_var_100000
