#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm #devel # only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=24           # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=CH_EX5
#SBATCH --output=output_CH_EX5.%J.txt
### #SBATCH --error=error.%J.txt
#SBATCH --time=2:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

# Define use variables:
OUTPUT_NAME="convex_hull_vertex_results"
PROJECT="project_ID11_5.yaml"

srun \
--ntasks=$SLURM_NTASKS \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_feasible_hull_exploration.py \
--n_batch 200 \
--case JP1 \
--project $PROJECT \
--input_csv_name convex_hull_vertex_200020.csv \
--output_name $OUTPUT_NAME

echo "filter for unique vertex points"

python3 ~/LTT/problemdriven_clustering/hpc/sm/build_unique_vertex_points.py \
--case JP1 \
--project $PROJECT \
--vertex_input_name $OUTPUT_NAME \
--vertex_output_name $OUTPUT_NAME \
--n_unique 5000

echo "Done!"
