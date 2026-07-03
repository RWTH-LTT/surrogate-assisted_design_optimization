#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm # c23mm or devel # devel only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=48             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=data5
#SBATCH --output=output_data_5.%J.txt
### #SBATCH --error=error_op.%J.txt
#SBATCH --time=4:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

# Define User Variables:
PROJECT="project_ID14_5.yaml"

### Conduct operational optimizations
echo "Step 1/4: Generate OPEX Training Data"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 50 \
--case JP1 \
--project $PROJECT \
--input_csv_name operational_opt_var_ts_lhs_sample_100000.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name op_var_100000

### Conduct operational optimizations
echo "Step 2/4: Generate OC Training Data"
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 50 \
--case JP1 \
--project $PROJECT \
--input_csv_name operational_opt_var_ts_lhs_sample_100000.csv \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name op_var_max_overcapacity_100000 \
--optimization_type max_overcapacity

echo "Step 3/4: Generate design Training Data"
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_design_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name design_opt_var_ts_lhs_sample_10800.csv \
--ignore_ts_name design_opt_var_ts_ts_ignore.yaml \
--output_name design_10800

echo "Step 4/4: max oc optimizations for optimal designs"
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name 'design_results_for_maxoc.csv' \
--ignore_ts_name design_opt_var_ts_ts_ignore.yaml \
--output_name "design_opt_max_oc" \
--optimization_type max_overcapacity

echo "Done!"
