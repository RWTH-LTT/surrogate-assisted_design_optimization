"""
This module tracks the prediction time of the feasibility LP depending on the number of unique
vertices. Note: This process dos not require the HPC (as it is relatively cheap).
It is only used to achieve comparable results with the other calculations performed on HPC.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from core.helpers import get_data_out, get_process_names, get_config_file
from core.ts_scaling import get_ts_scaling_names
from sm.surrogate_model import SurrogateModelNN

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', help='the name to case for the input data', required=True)
parser.add_argument('--in_test', help='name of the csv file with the test data', required=True)

args = parser.parse_args()

PROJECT_FILES = [
    "project_ID11_5.yaml",
    "project_ID11_10.yaml",
    "project_ID11_50.yaml",
    "project_ID11_100.yaml"
    ]

CASE = args.case
ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')

def get_sm(data_in, project, output_name, sm_name_config, sm_name, af:bool):

    data_out = get_data_out(data_in, project)
    # Load surrogate
    ts_scaling_names = get_ts_scaling_names(
        data_in, project, only_feasibility_affecting=af)
    processes = get_process_names(
        data_in, project, only_feasibility_affecting=af, storage_or_production='both')

    sm = SurrogateModelNN(
        data_in,
        project,
        input_names =  ts_scaling_names + processes,
        output_name=output_name,
        sm_name=sm_name_config)
    sm.setup_sm()
    sm.load_model(os.path.join(data_out, sm_name), sm_name)
    return sm

def get_prediction_times(sm, df_in):

    pred_time = np.zeros_like(range(len(df_in)), dtype=float)
    for i in range(len(df_in)):
        df_row = df_in.loc[[i]]
        x_in_scaled = sm.get_x_in_scaled(df_row)

        t0 = time.perf_counter_ns()
        _ = sm.sm.predict(x_in_scaled)
        dt = time.perf_counter_ns() - t0

        pred_time[i]=dt

    return pred_time * 1e-9

def main():

    df_dt = pd.DataFrame()

    # Loop over project files
    for project_file in PROJECT_FILES:
        data_out = get_data_out(DATA_IN, project_file)
        config = get_config_file(DATA_IN, project_file)
        n_day = int(config['time_series_aggregation']['noTypicalPeriods'])

        # Load test set:
        df_test = pd.read_csv(os.path.join(data_out, f"{args.in_test}.csv"))

        # OPEX predictor
        sm_opex = get_sm(DATA_IN, project_file, "OPEX_woLL", "ANN_4L", "NN_OPEX", False)
        df_dt[f"N{n_day}_OPEX_woLL"] = get_prediction_times(sm_opex, df_test)

        # OC predictor
        sm_oc = get_sm(DATA_IN, project_file, "overcapacity", "ANN_4L", "NN_overcapacity", True)
        df_dt[f"N{n_day}_OC"] = get_prediction_times(sm_oc, df_test)

        # ENS predictor
        sm_ens = get_sm(DATA_IN, project_file, "lost_load", "ANN_4L", "NN_lostload", True)
        df_dt[f"N{n_day}_ENS"] = get_prediction_times(sm_ens, df_test)

    #  save computation times to the last encountered data_out
    df_dt.to_csv(os.path.join(
        data_out, 'ann_prediction_times.csv'), index=False)

if __name__=='__main__':
    main()
