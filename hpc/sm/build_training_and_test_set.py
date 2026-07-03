import argparse
import os
import pandas as pd
from core.helpers import get_experiment_id, get_value_of_lost_load, get_config_file

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', help='the name to case for the input data', required=True)
parser.add_argument('--project_file', default='project.yaml', help='name of the .yaml file in data_in_path containing the input information')
parser.add_argument('--res_name_op',  help='name of the csv file containing the results of the operational optimization', required=True)
parser.add_argument('--res_name_maxoc',  help='name of the csv file containing the results of the max overcapacity optimization',required=True)
parser.add_argument('--sm_name_config', help='name of the sm paramters in config.yaml', required=True)

ARGS = parser.parse_args()

CASE = ARGS.case
PROJECT_FILE = ARGS.project_file
ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', get_experiment_id(DATA_IN, PROJECT_FILE))
CONFIG = get_config_file(DATA_IN, PROJECT_FILE)

def main():
    # Read results from operational optimization
    df_train = pd.read_csv(os.path.join(DATA_OUT, ARGS.res_name_op))

    # Load overcapacity data
    df_overcapacity = pd.read_csv(os.path.join(DATA_OUT, ARGS.res_name_maxoc))

    # Overwrite the 'overcapacity' column in df_train_raw with the values from df_overcapacity
    df_train['overcapacity'] = df_overcapacity['overcapacity']

    # value of loss load dict (with products as keys):
    voll_dict = get_value_of_lost_load(DATA_IN, PROJECT_FILE)

    # Calculate OPEX minus loss load for each product
    df_train['OPEX_woLL'] = df_train['OPEX'] - sum(
        voll_dict[product] * df_train[f'lost_load_{product}']
        for product in voll_dict.keys()
    )

    # Read test size:
    test_size = CONFIG['surrogate_models'][ARGS.sm_name_config]['test_size']

    # Sanity check on test size:
    if not 0 <= test_size < 1:
        raise ValueError(f"Test size must be between 0 and 1, got {test_size}")

    # Split data into training and testing sets
    train_size = int((1 - test_size) * len(df_train))

    # Randomly shuffle the data
    df_train_shuffle = df_train.sample(
        frac=1, random_state=CONFIG['seed']).reset_index(drop=True).copy()

    # Split the data
    df_train = df_train_shuffle.iloc[:train_size].copy(deep=True)
    df_test = df_train_shuffle.iloc[train_size:].copy(deep=True)

    # Save the training and testing data
    df_train.to_csv(
        os.path.join(DATA_OUT, ARGS.res_name_op.replace('.csv', '_train.csv')), index=False)
    df_test.to_csv(
        os.path.join(DATA_OUT, ARGS.res_name_op.replace('.csv', '_test.csv')), index=False)

    # print lengths
    print(f"Training set length: {len(df_train)}, Testing set length: {len(df_test)}")

if __name__=='__main__':
    main()
