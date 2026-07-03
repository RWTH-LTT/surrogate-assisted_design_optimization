#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm #devel # only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=48             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=phsm5_N64
#SBATCH --output=output_phsm5_N64.%J.txt
### #SBATCH --error=error.%J.txt
#SBATCH --time=6:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

# Define use variables:
OUTPUT_NAME=""
PROJECT="project_ID11_5_N64.yaml"
OC_MIN="0.0"

### Hybrid SM (HSM) without feasiblity predictor
echo "Step 1.1: Predict optimal design with HSM_noFP:"
 
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_design_var_ts_hybrid_sm_no_fp.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name design_opt_var_ts_lhs_sample_10800.csv \
--opex_sm_name_config ANN_4L \
--opex_sm_name NN_opex \
--output_name "design_hybrid_sm_noFP${OUTPUT_NAME}"

echo "Step 1.2: Test predictions"

echo "Step 1.2.A: Test predictions with operational optimization with HSM_noFP:"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name "design_hybrid_sm_noFP${OUTPUT_NAME}_filtered.csv" \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name "design_hybrid_sm_noFP${OUTPUT_NAME}_filtered_opex" \
--optimization_type operation

echo "Step 1.2.B: Test predictions with max_overcapacity optimization with HSM_noFP:"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name "design_hybrid_sm_noFP${OUTPUT_NAME}_filtered.csv" \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name "design_hybrid_sm_noFP${OUTPUT_NAME}_filtered_max_oc" \
     --optimization_type max_overcapacity

### Hybrid SM (HSM) for lostload
echo "Step 2.1: Predict optimal design with HSM_LL:"
 
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_design_var_ts_hybrid_sm.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name design_opt_var_ts_lhs_sample_10800.csv \
--opex_sm_name_config ANN_4L \
--feas_sm_name_config ANN_4L \
--opex_sm_name NN_opex \
--feas_sm_name NN_lostload \
--feas_sm_type lost_load \
--output_name "design_hybrid_sm_lostload${OUTPUT_NAME}"

echo "Step 2.2: Test predictions"

echo "Step 2.2.A: Test predictions with operational optimization with HSM_LL:"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name "design_hybrid_sm_lostload${OUTPUT_NAME}_filtered.csv" \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name "design_hybrid_sm_lostload${OUTPUT_NAME}_filtered_opex" \
--optimization_type operation

echo "Step 2.2.B: Test predictions with max_overcapacity optimization with HSM_LL:"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name "design_hybrid_sm_lostload${OUTPUT_NAME}_filtered.csv" \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name "design_hybrid_sm_lostload${OUTPUT_NAME}_filtered_max_oc" \
--optimization_type max_overcapacity

### Hybrid SM (HSM) for overcapacity
echo "Step 3.1: Predict optimal design with HSM_OC:"
 
srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_design_var_ts_hybrid_sm.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name design_opt_var_ts_lhs_sample_10800.csv \
--opex_sm_name_config ANN_4L \
--feas_sm_name_config ANN_4L \
--opex_sm_name NN_opex \
--feas_sm_name NN_overcapacity \
--feas_sm_type overcapacity \
--output_name "design_hybrid_sm_overcapacity${OUTPUT_NAME}" \
--oc_min $OC_MIN

echo "Step 3.2: Test predictions"

echo "Step 3.2.A: Test predictions with operational optimization with HSM_OC:"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name "design_hybrid_sm_overcapacity${OUTPUT_NAME}_filtered.csv" \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name "design_hybrid_sm_overcapacity${OUTPUT_NAME}_filtered_opex" \
--optimization_type operation

echo "Step 3.2.B: Test predictions with max_overcapacity optimization with HSM_OC:"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_operation_var_ts_dynamic_mpi.py \
--n_batch 25 \
--case JP1 \
--project $PROJECT \
--input_csv_name "design_hybrid_sm_overcapacity${OUTPUT_NAME}_filtered.csv" \
--ignore_ts_name operational_opt_var_ts_ts_ignore.yaml \
--output_name "design_hybrid_sm_overcapacity${OUTPUT_NAME}_filtered_max_oc" \
--optimization_type max_overcapacity

echo "Done!"
