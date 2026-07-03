import os
import argparse
import pandas as pd
from core.helpers import get_experiment_id, get_process_names
from sm.surrogate_model import SurrogateModelNN
from core.ts_scaling import get_ts_scaling_names

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case',           help='the name to case for the input data', required=True)
parser.add_argument('--project_file',   help='name of the .yaml file in data_in_path containing the input information', required=True)
parser.add_argument('--csv_train_name', help='name of the csv file containing the timeseries, e.g. df_in.csv', required=True)
parser.add_argument('--sm_name_config', help='name of the sm paramters in config.yaml', required=True)
parser.add_argument('--sm_name',        help='name of the surrogate model', required=True)
parser.add_argument('--output_name',    help='name variable to be fitted', required=True)
parser.add_argument('--af',             help= 'if only input variables which affect feasibility should be used', required=True)

ARGS = parser.parse_args()

CASE             = ARGS.case
PROJECT_FILE     = ARGS.project_file
CSV_TRAIN_NAME   = ARGS.csv_train_name
ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')

EXPERIMENT_ID = get_experiment_id(DATA_IN, PROJECT_FILE)

DATA_COMPUTED_IN = os.path.join(ROOT, 'data', CASE, 'ComputedInput_Data', EXPERIMENT_ID)
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', EXPERIMENT_ID, )

os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

ONLY_FEASIBILTIY_AFFECTING = ARGS.af == "True"

def main():

    # Load training data
    print(f"Reading training data from {os.path.join(DATA_OUT, CSV_TRAIN_NAME)}")
    df_train = pd.read_csv(os.path.join(DATA_OUT, CSV_TRAIN_NAME))
    print("\tDone reading training data.")

    process_names = get_process_names(
        DATA_IN, PROJECT_FILE,
        only_feasibility_affecting = ONLY_FEASIBILTIY_AFFECTING,
        storage_or_production      = 'both')
    ts_scaling_names = get_ts_scaling_names(
        DATA_IN, PROJECT_FILE,
        only_feasibility_affecting = ONLY_FEASIBILTIY_AFFECTING
    )

    sm = SurrogateModelNN(
        data_in_path = DATA_IN,
        project_file = PROJECT_FILE,
        input_names  = process_names + ts_scaling_names,
        output_name  = ARGS.output_name,
        sm_name      = ARGS.sm_name_config)

    sm.generate_train_val_test_sets(
        df_unscaled     = df_train,
        split_test_data = False # Test data is seperated before
    )
    sm.train_model(checkpoint_path=os.path.join(DATA_OUT, ARGS.sm_name))
    sm.save_model(output_path=os.path.join(DATA_OUT, ARGS.sm_name), filename=ARGS.sm_name)

if __name__=='__main__':
    main()
