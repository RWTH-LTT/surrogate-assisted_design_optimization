#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23mm # c23mm or devel # devel only for development
#SBATCH --nodes=1               # NEW Number of nodes
#SBATCH --ntasks=1             # NEW Number of tasks (processes)
#SBATCH --cpus-per-task=1       # FROM 8
#SBATCH --job-name=feas_lp
#SBATCH --output=output_feas_lp.%J.txt
### #SBATCH --error=error_op.%J.txt
#SBATCH --time=5:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

# Define User Variables:
PROJECT="project_ID12_5.yaml"

### Conduct feasibility_lp classification
echo "Feasibility LP classification"

python3 ~/LTT/problemdriven_clustering/hpc/sm/dsm_feasibility_lp.py \
--case JP1 \
--project $PROJECT \
--vertex_in convex_hull_vertex_results \
--vertex_out convex_hull_vertex_results \
--test_in op_var_100000_test \
--test_out feasibility_lp_prediction

echo "Done!"
