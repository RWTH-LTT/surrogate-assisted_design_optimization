"""
This module tracks the prediction time of the feasibility LP depending on the number of unique
vertices. Note: This process dos not require the HPC (as it is relatively cheap).
It is only used to achieve comparable results with the other calculations performed on HPC.
"""
import argparse
import os
import pandas as pd
from core.helpers import get_experiment_id
from sm.surrogates_feasible_space import (Feasibility_LP, get_result_column_names)


# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', default='dsm_LP_demo', help='the name to case for the input data', required=True)
parser.add_argument('--project_file', default='project.yaml', help='name of the .yaml file in data_in_path containing the input information', required=True)
parser.add_argument('--vertex_in',  help='input: name of csv files with all vertices (w/o .csv)', required=True)
parser.add_argument('--vertex_out',  help='output: name of csv files with unique vertices (w/o .csv)', required=True)
parser.add_argument('--test_in',  help='input: name of csv files with test samples (w/o .csv)', required=True)
parser.add_argument('--test_out',  help='output: name of csv files with tested samples (w/o .csv)', required=True)

args = parser.parse_args()

CASE = args.case
PROJECT_FILE = args.project_file
VERTEX_IN_CSV_NAME  = args.vertex_in
VERTEX_OUT_CSV_NAME = args.vertex_out
TEST_IN_CSV_NAME    = args.test_in
TEST_OUT_CSV_NAME   = args.test_out

ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')
EXPERIMENT_ID = get_experiment_id(DATA_IN, PROJECT_FILE)

DATA_COMPUTED_IN = os.path.join(ROOT, 'data', CASE, 'ComputedInput_Data', EXPERIMENT_ID)
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', EXPERIMENT_ID, )
os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

def main():

    # generate smaller sets of unique points
    df_sample = pd.read_csv(os.path.join(DATA_OUT, f"{VERTEX_IN_CSV_NAME}.csv"), index_col='id')

    # Filter out doublicates:
    column_names_norm = get_result_column_names(DATA_IN, PROJECT_FILE, True)
    df_unique = df_sample.drop_duplicates(
        subset=column_names_norm, keep='first'
    )

    # Build list of unique data points:
    n_unique_list = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
    n_unique_list.append(len(df_unique))

    for n_unique in n_unique_list:
        print(20*'#' + f"\n\t\t{n_unique}\n" + 20*'#', flush=True)

        # Save unique vertives
        df_unique[:n_unique].to_csv(os.path.join(
            DATA_OUT, f'{VERTEX_OUT_CSV_NAME}_{int(n_unique)}.csv'),index=False)

        # Read test_df which contains the files which are testet for feasibility
        test_df = pd.read_csv(os.path.join(DATA_OUT, f"{TEST_IN_CSV_NAME}.csv"))

        feasiblity_lp = Feasibility_LP(
            DATA_IN, PROJECT_FILE,
            df_vertices=df_unique[:n_unique]
        )

        (feasibility, dt) = feasiblity_lp.predict_feasibility(test_df)

        # Write results:
        test_df['convex_hull_feasibility'] = feasibility
        test_df['dt_feasibility_lp'] = dt

        test_df.to_csv(os.path.join(
            DATA_OUT, f'{TEST_OUT_CSV_NAME}_{int(n_unique)}.csv'), index=False)

if __name__=='__main__':
    main()
