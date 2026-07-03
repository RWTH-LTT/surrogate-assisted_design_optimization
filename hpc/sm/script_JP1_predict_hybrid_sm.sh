#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm # only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=96             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=predict_hybrid_sm
#SBATCH --output=output.%J.txt
#SBATCH --error=error.%J.txt
#SBATCH --time=3:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment
# Hybrid SM for lostload:
srun \
--ntasks=96 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_design_var_ts_hybrid_sm.py \
--n_batch 10 \
--case JP1 \
--project project.yaml \
--input_csv_name design_opt_var_ts_lhs_sample_10800.csv \
--sm_name_config ANN_8L \
--opex_sm_name NN_OPEX_8L \
--feas_sm_name NN_lostload_8L \
--feas_sm_type lost_load \
--output_name design_hybrid_sm_lostload

# Hybrid SM for overcapacityy
srun \
--ntasks=96 \
--cpus-per-task=1 \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_design_var_ts_hybrid_sm.py \
--n_batch 10 \
--case JP1 \
--project project.yaml \
--input_csv_name design_opt_var_ts_lhs_sample_10800.csv \
--sm_name_config ANN_8L \
--opex_sm_name NN_OPEX_8L \
--feas_sm_name NN_overcapacity_8L \
--feas_sm_type overcapacity \
--output_name design_hybrid_sm_overcapacity
