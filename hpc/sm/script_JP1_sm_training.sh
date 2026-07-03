#!/usr/local_rwth/bin/zsh

#SBATCH --account=rwth1889
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyung.juergens@ltt.rwth-aachen.de
#SBATCH --partition=c23g # ML node
#SBATCH --nodes=1        # Number of nodes
#SBATCH --ntasks=1       # Number of tasks (processes)
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --job-name=ann100_E10k
#SBATCH --output=output_ann100_E10k.%J.txt
### #SBATCH --error=error_ann.%J.txt
#SBATCH --time=10:00:00
### beginning of executable commands
module load Python/3.12.3
module load GCCcore/.13.3.0
module load Gurobi/12.0.0

# Define user variabels:
SM_NAME=""
CONFIG_NAME="ANN_4L"
TRAIN_FILE="op_var_100000_train.csv"
RES_NAME_OP="op_var_100000.csv"
RES_NAME_MAXOC="op_var_max_overcapacity_100000.csv"
PROJECT="project_ID11_100_E10k.yaml"

cd ~/LTT/problemdriven_clustering ### setze Working Directory
source ~/LTT/problemdriven_clustering/venv/venv_dsm/bin/activate ### aktiviere die Virtuelle Environment

echo "Generate Training and Test Sets:"
python3 ~/LTT/problemdriven_clustering/hpc/sm/build_training_and_test_set.py \
--case JP1 \
--project $PROJECT \
--res_name_op $RES_NAME_OP \
--res_name_maxoc $RES_NAME_MAXOC \
--sm_name_config $CONFIG_NAME

echo "Train Lost load surrogate with ${TRAIN_FILE}:"

python3 ~/LTT/problemdriven_clustering/hpc/sm/train_ann.py \
--case JP1 \
--project $PROJECT \
--csv_train_name $TRAIN_FILE \
--sm_name_config $CONFIG_NAME \
--sm_name "NN${SM_NAME}_lostload" \
--output_name lost_load \
--af True

echo "Train Overcapacity surrogate with ${TRAIN_FILE}:"

python3 ~/LTT/problemdriven_clustering/hpc/sm/train_ann.py \
--case JP1 \
--project $PROJECT \
--csv_train_name $TRAIN_FILE \
--sm_name_config $CONFIG_NAME \
--sm_name "NN${SM_NAME}_overcapacity" \
--output_name overcapacity \
--af True

echo "Train OPEX surrogate with ${TRAIN_FILE}:"

python3 ~/LTT/problemdriven_clustering/hpc/sm/train_ann.py \
--case JP1 \
--project $PROJECT \
--csv_train_name $TRAIN_FILE \
--sm_name_config $CONFIG_NAME \
--sm_name "NN${SM_NAME}_opex" \
--output_name OPEX_woLL \
--af False
