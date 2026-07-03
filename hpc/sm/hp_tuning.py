import os
import argparse
import pandas as pd
from core.helpers import get_experiment_id, get_process_names, get_config_file
from sm.surrogate_model import SurrogateModelNN
from core.ts_scaling import get_ts_scaling_names

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case',           help='the name to case for the input data', required=True)
parser.add_argument('--project_file',   help='name of the .yaml file in data_in_path containing the input information', required=True)
parser.add_argument('--csv_hp_name',    help='name of the csv file containing hyperparameters', required=True)
parser.add_argument('--csv_train_name', help='name of the csv file containing the training data', required=True)
parser.add_argument('--sm_name_config', help='name of the sm paramters in config.yaml', required=True)
parser.add_argument('--output_name',    help='name variable to be fitted', required=True)
parser.add_argument('--af',             help= 'if only input variables which affect feasibility should be used', required=True)
parser.add_argument('--export_name',    help= 'name of the exported results', required=True)


ARGS = parser.parse_args()

CASE           = ARGS.case
PROJECT_FILE   = ARGS.project_file
CSV_HP_NAME    = ARGS.csv_hp_name
CSV_TRAIN_NAME = ARGS.csv_train_name

ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')

EXPERIMENT_ID = get_experiment_id(DATA_IN, PROJECT_FILE)

DATA_COMPUTED_IN = os.path.join(ROOT, 'data', CASE, 'ComputedInput_Data', EXPERIMENT_ID)
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', EXPERIMENT_ID, )
EXPORT_NAME = os.path.join(DATA_OUT, ARGS.export_name)

os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

ONLY_FEASIBILTIY_AFFECTING = ARGS.af == "True"

def main():

    # Load hp, training and test data
    df_hp = pd.read_csv(os.path.join(DATA_COMPUTED_IN, CSV_HP_NAME), index_col='Unnamed: 0')
    # add new colomns for test, training and validation loss:
    df_hp['train_loss'] = None
    df_hp['valid_loss'] = None

    config = get_config_file(DATA_IN, PROJECT_FILE)
    df_train = pd.read_csv(os.path.join(DATA_OUT, CSV_TRAIN_NAME))

    process_names = get_process_names(
        DATA_IN, PROJECT_FILE,
        only_feasibility_affecting = ONLY_FEASIBILTIY_AFFECTING,
        storage_or_production      = 'both')
    ts_scaling_names = get_ts_scaling_names(
        DATA_IN, PROJECT_FILE,
        only_feasibility_affecting = ONLY_FEASIBILTIY_AFFECTING
    )

    # Loop over df_hp:
    for idx in df_hp.index:
        df_row = df_hp.loc[[idx]]
        print(f"{idx}/{len(df_hp)}: {df_row}")
        # Setup SM
        sm_export_name = f"NN_{idx}"

        sm = SurrogateModelNN(
            DATA_IN,
            PROJECT_FILE,
            input_names = process_names + ts_scaling_names,
            output_name = ARGS.output_name,
            sm_name     = ARGS.sm_name_config,
            hp_df_row   = df_row.reset_index())

        sm.generate_train_val_test_sets(df_train, split_test_data=False)

        checkpoint_path = (
            None
            if config['surrogate_models'][ARGS.sm_name_config]['checkpoints']
            else os.path.join(EXPORT_NAME, sm_export_name)
            )

        sm.train_model(checkpoint_path=checkpoint_path)
        sm.save_model(
            output_path = os.path.join(EXPORT_NAME, sm_export_name),
            filename    = sm_export_name)

        # Add losses to list:
        for i in range(len(sm.sm.history) - 1, -1, -1):
            if sm.sm.history[i]['valid_loss_best']:
                # Save Training and validation loss (if its the best)
                df_hp.loc[idx, 'train_loss'] = sm.sm.history[i]['train_loss']
                df_hp.loc[idx, 'valid_loss'] = sm.sm.history[i]['valid_loss']
                break

        df_hp.to_csv(
            os.path.join(EXPORT_NAME, CSV_HP_NAME.replace('.csv', '_output.csv')),
            index=True)

if __name__=='__main__':
    main()
