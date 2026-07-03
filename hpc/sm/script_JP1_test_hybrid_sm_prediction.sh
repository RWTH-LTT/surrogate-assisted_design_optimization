#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm # c23mm or devel # devel only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=48             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=test_hsm_pred
#SBATCH --output=output_test_hsm_pred.%J.txt
#SBATCH --error=error_test_hsm_pred.%J.txt
#SBATCH --time=3:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

### LostLoad HSM
### Run max overcapacity optimization
srun \
--ntasks=48 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project project.yaml \
--input_csv_name design_hybrid_sm_lostload_cache_filtered.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name design_hybrid_sm_lostload_cache_filtered_max_oc \
--optimization_type max_overcapacity

### Run operational optimization
srun \
--ntasks=48 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project project.yaml \
--input_csv_name design_hybrid_sm_lostload_cache_filtered.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name design_hybrid_sm_lostload_cache_filtered_opex \
--optimization_type operation


### Overcapacity HSM
### Run max overcapacity optimization
srun \
--ntasks=48 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project project.yaml \
--input_csv_name design_hybrid_sm_overcapacity_cache_filtered.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name design_hybrid_sm_overcapacity_cache_filtered_max_oc \
--optimization_type max_overcapacity

### Run operational optimization
srun \
--ntasks=48 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project project.yaml \
--input_csv_name design_hybrid_sm_overcapacity_cache_filtered.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name design_hybrid_sm_overcapacity_cache_filtered_opex \
--optimization_type operation

