#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23g # ML node
#SBATCH --nodes=1        # Number of nodes
#SBATCH --ntasks=1       # Number of tasks (processes)
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=hp_tuning
#SBATCH --output=output_hp_tuning.%J.txt
### #SBATCH --error=error_ann.%J.txt
#SBATCH --time=05:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

python3 ~/LTT/problemdriven_clustering/hpc/sm/hp_tuning.py \
--case JP1 \
--project project_ID8_5.yaml \
--csv_hp_name "hp_tuning.csv" \
--csv_train_name "op_var_train_10000_hp.csv" \
--sm_name_config "hp_tuning" \
--output_name lost_load \
--af True \
--export_name "hp_tuning_lostload"
