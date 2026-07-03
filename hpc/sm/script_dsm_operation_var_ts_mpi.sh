#!/usr/local_rwth/bin/zsh

### #SBATCH --account=your_project_ID
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm # devel # only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=48             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=dsm_operation_var_ts
#SBATCH --output=output.%J.txt
#SBATCH --time=24:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment
srun \
--ntasks=48 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_mpi.py \
--n_batch 2000 \
--case dsm_sumo_small \
--project project.yaml \
--input_csv_name operational_opt_var_ts_lhs_sample.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name operation_var_ts
