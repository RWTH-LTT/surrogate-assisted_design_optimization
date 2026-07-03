""" This module is used for generating training data for operational optimizations with variable
timeseries and design capacities. Indended use is on the HPC with dynamic multi processing.
Module is called by script_dsm_operation_var_ts_dmpi.sh
"""

import argparse
import os
import sys
import yaml
import traceback
import pandas as pd
from mpi4py import MPI  # multi processing interface
from core.helpers import get_experiment_id, UniqueKeyLoader
from core.hpc_helpers import get_input_df_for_rerun
from sm.routines import generate_aggregated_input_data, operational_optimization_var_ts

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', help='the name to case for the input data', required=True)
parser.add_argument('--n_batch', help='id of the current step', required=True)
parser.add_argument('--project_file', default='project.yaml', help='name of the .yaml file in data_in_path containing the input information')
parser.add_argument('--input_csv_name',  help='name of the csv file containing the timeseries, e.g. df_in.csv', required=True)
parser.add_argument('--ignore_ts_name',  help='name of the yaml file containing the timeseries which will be ignored, e.g. ts_ingore.yaml', required=True)
parser.add_argument('--output_name',  help='name of the output csv file, e.g. df_out (without.csv)', required=True)
parser.add_argument('--optimization_type',  help='"operation" or "max_overcapacity"',
                    required=False, default='operation')
parser.add_argument('--rerun',  help='If an incomplete output file should be extended', default="True")

ARGS = parser.parse_args()

CASE = ARGS.case
BATCH_SIZE = int(ARGS.n_batch)
PROJECT_FILE = ARGS.project_file
CSV_INPUT_NAME = ARGS.input_csv_name
OUTPUT_NAME = ARGS.output_name
OPTIMIZATION_TYPE = ARGS.optimization_type
assert OPTIMIZATION_TYPE in {'operation', 'max_overcapacity'}

ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')

DATA_COMPUTED_IN = os.path.join(
    ROOT, 'data', CASE, 'ComputedInput_Data', get_experiment_id(DATA_IN, PROJECT_FILE))
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', get_experiment_id(DATA_IN, PROJECT_FILE))

RESULTS_FILE = os.path.join(DATA_OUT, OUTPUT_NAME + ".csv")
RERUN =  ARGS.rerun == "True"
header_written = os.path.exists(RESULTS_FILE)  # Only check once at start

if header_written and not RERUN: # Check that file does not exist
    raise FileExistsError(f"The file '{RESULTS_FILE}' already exists.")

def save_result(task_index, result_value, result_columns):
    """Atomically append a new result to the results file."""
    global header_written

    df_out = pd.DataFrame(result_value, columns= result_columns)
    df_out.insert(0, 'task_index', task_index)

    # Write to temporary file first
    df_out.to_csv(RESULTS_FILE, mode='a', index=False, header=not header_written)

    if not header_written:
        header_written = True  # Ensure header is only written once

    print(f"Saved results from {task_index} to {RESULTS_FILE}.", flush=True)

def sort_by_index():
    """ master sorts output.csv by index after all workers are done."""
    df_out = pd.read_csv(RESULTS_FILE)

    # Sort by index
    df_out = df_out.sort_values(by='id', ascending=True)
    df_out.reset_index(drop=True, inplace=True)

    # Save file again:
    df_out.to_csv(RESULTS_FILE, index=False)

def master():
    """Master process: Loads full DataFrame, distributes tasks, and collects results."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # Master makes sure the directories exist
    os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)

    # Master generates aggregated input data:
    print("Master process: Generating aggregated input data:", flush=True)
    generate_aggregated_input_data(DATA_IN, PROJECT_FILE)

    # Master loads the full DataFrame
    print("Master process: Reading csv file:", flush=True)
    if RERUN:
        df_in = get_input_df_for_rerun(
            df_in_full_path = os.path.join(DATA_COMPUTED_IN, CSV_INPUT_NAME),
            df_out_1_path   = RESULTS_FILE
            )
    else:
        df_in = pd.read_csv(os.path.join(DATA_COMPUTED_IN, CSV_INPUT_NAME))
    print("\nDone reading csv file.", flush=True)

    num_batches = (len(df_in) + BATCH_SIZE - 1) // BATCH_SIZE

    task_index = 0
    in_use = []  # workers that currently hold a task

    # Send timeseries_ignore to all workers
    with open(os.path.join(DATA_COMPUTED_IN, ARGS.ignore_ts_name), 'rb') as f:
        timeseries_ignore = yaml.load(f, Loader=UniqueKeyLoader)
    for worker_id in range(1, size):
        comm.send(timeseries_ignore, dest=worker_id, tag=3)  # Use tag=3 for timeseries_ignore

    # Send column names to all workers first
    column_names = df_in.columns.tolist()  # Get column names
    for worker_id in range(1, size):
        comm.send(column_names, dest=worker_id, tag=0)  # Use tag=0 for metadata

    # Assign initial tasks to workers
    for worker_id in range(1, min(size, num_batches + 1)):
        start_idx = task_index * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(df_in))
        batch_data = df_in.iloc[start_idx:end_idx].values

        print(f"\tSending taks {task_index} to worker {worker_id}.", flush=True)
        comm.send((task_index, batch_data), dest=worker_id, tag=1)
        in_use.append(worker_id)
        task_index += 1

    # Immediately stop surplus idle workers (those who never get work)
    idle_surplus = [wid for wid in range(1, size) if wid not in in_use]
    for wid in idle_surplus:
        print(f"\tStopping idle worker {wid}.", flush=True)
        comm.send(None, dest=wid, tag=1)

    while in_use:
        # Receive results
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker_id = status.Get_source()

        # Save result immediately after receiving
        task_index_old, result_value, result_columns = result
        save_result(task_index_old, result_value, result_columns)

        # print log
        print(
            f"Done with task_index {task_index_old}. " +
            f"Total progress: {task_index_old/num_batches*100:.2f}%", flush=True
            )

        if task_index < num_batches:
            start_idx = task_index * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(df_in))
            batch_data = df_in.iloc[start_idx:end_idx].values  # Send only the required batch

            print(f"\tSending taks {task_index} to worker {worker_id}.", flush=True)
            comm.send((task_index, batch_data), dest=worker_id, tag=1)
            task_index += 1
        else:
            comm.send(None, dest=worker_id, tag=1)  # Stop signal
            print(f"\tStopping worker {worker_id}.", flush=True)
            in_use.remove(worker_id)

    sort_by_index()

    print(f"Master: Finished processing {num_batches} batches.", flush=True)

def worker():
    """Worker process: Receives tasks and performs computation."""
    comm = MPI.COMM_WORLD

    # Receive timeseries_ignore from master
    timeseries_ignore = comm.recv(source=0, tag=3)

    # Receive column names from master
    column_names = comm.recv(source=0, tag=0)

    while True:
        task = comm.recv(source=0, tag=1)
        if task is None:
            break

        (task_index, batch_data) = task
        batch_df = pd.DataFrame(batch_data, columns=column_names)

        result_df = operational_optimization_var_ts(
            df_in              = batch_df,
            data_in_path       = DATA_IN,
            project_file       = PROJECT_FILE,
            timeseries_ignored = timeseries_ignore,
            optimization_type  = OPTIMIZATION_TYPE
            )

        result_value = result_df.values
        result_columns = result_df.columns.tolist()

        comm.send((task_index, result_value, result_columns), dest=0, tag=2)

if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        if rank == 0:
            master()

        else:
            worker()

    # Make sure, that the slurm-job is killed, if one process fails:
    except Exception as e:
        print(f"Rank {MPI.COMM_WORLD.Get_rank()} failed with: {e}")

    	# Print Error message to error file:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        MPI.COMM_WORLD.Abort(1)
