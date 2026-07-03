
import sys
import traceback
import argparse
import os
import pandas as pd
from mpi4py import MPI  # multi processing interface
from core.helpers import get_experiment_id
from sm.routines import (
    design_optimization_var_ts_hybrid_sm, export_hsm_predictions_for_operational_optimization)

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', default='dsm_LP_demo', help='the name to case for the input data', required=True)
parser.add_argument('--n_batch', default='2000', help='id of the current step', required=True)
parser.add_argument('--project', default='project.yaml', help='name of the .yaml file in data_in_path containing the input information', required=True)
parser.add_argument('--input_csv_name',  help='name of the csv file containing the timeseries, e.g. df_in.csv', required=True)
parser.add_argument('--opex_sm_name_config', help='name of this hyperparameter_dict in config.yaml', required=True)
parser.add_argument('--feas_sm_name_config', help='name of this hyperparameter_dict in config.yaml', required=True)
parser.add_argument('--opex_sm_name', help='name of the opex surrogate model, e.g., ANN_OPEX', required=True)
parser.add_argument('--feas_sm_name', help='name of the feasibility surrogate model, e.g., ANN_lostload', required=True)
parser.add_argument('--feas_sm_type', help="'lostload', 'overcapacity', or 'convex_hull'", required=True)
parser.add_argument('--output_name',  help='name of the output csv file, e.g. df_out (without.csv)', required=True)
parser.add_argument('--oc_min',  help='minimal allowed overcapacity', default=0)

ARGS = parser.parse_args()

CASE = ARGS.case
BATCH_SIZE = int(ARGS.n_batch)
PROJECT_FILE = ARGS.project
CSV_INPUT_NAME = ARGS.input_csv_name
OUTPUT_NAME = ARGS.output_name

ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')
EXPERIMENT_ID = get_experiment_id(DATA_IN, PROJECT_FILE)

DATA_COMPUTED_IN = os.path.join(ROOT, 'data', CASE, 'ComputedInput_Data', EXPERIMENT_ID)
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', EXPERIMENT_ID, )

RESULTS_FILE = os.path.join(DATA_OUT, OUTPUT_NAME + ".csv")

header_written = os.path.exists(RESULTS_FILE)  # Only check once at start

if header_written: # Check that file does not exist
    raise FileExistsError(f"The file '{RESULTS_FILE}' already exists.")

#####################
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process rank
numP = comm.Get_size()  # Total number of processes
print(f"rank={rank}")

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

def clean_results_file():
    """ master cleans results file in two ways:
    Step 1: Sort by id and save it to DATA_OUT
    Step 2: Set negative capacity values to zero and remove the row 'task_index'  
    """
    df_out = pd.read_csv(RESULTS_FILE)

    # Step 1: Sort by index
    df_out = df_out.sort_values(by='id', ascending=True)
    df_out.reset_index(drop=True, inplace=True)

    # Save file again:
    df_out.to_csv(RESULTS_FILE, index=False)

    # Step 2:Export also filtered .csv file (ready to be used in operatioanl optimization):
    export_hsm_predictions_for_operational_optimization(
        df_hsm_prediction=df_out,
        data_in_path=DATA_IN,
        project_file=PROJECT_FILE,
        filepath_out=os.path.join(DATA_COMPUTED_IN, OUTPUT_NAME + "_filtered.csv")
    )

def master():
    """Master process: Loads full DataFrame, distributes tasks, and collects results."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # Master makes sure the directories exist
    os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)

    # Master loads the full DataFrame
    print("Master process: Reading csv file:", flush=True)
    df_in = pd.read_csv(os.path.join(DATA_COMPUTED_IN, CSV_INPUT_NAME))
    print("\nDone reading csv file.", flush=True)

    num_batches = (len(df_in) + BATCH_SIZE - 1) // BATCH_SIZE

    task_index = 0
    active_workers = 0

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
        task_index += 1
        active_workers += 1

    while active_workers > 0:
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
            f"Total progress: {task_index_old/num_batches*100:.2f}%", flush=True)

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
            active_workers -= 1

    clean_results_file()

    print(f"Master: Finished processing {num_batches} batches.", flush=True)

def worker():
    """Worker process: Receives tasks and performs computation."""
    comm = MPI.COMM_WORLD

    # Receive column names from master
    column_names = comm.recv(source=0, tag=0)

    while True:
        task = comm.recv(source=0, tag=1)
        if task is None:
            break

        (task_index, batch_data) = task 
        batch_df = pd.DataFrame(batch_data, columns=column_names)  

        result_df = design_optimization_var_ts_hybrid_sm(
            df_in               = batch_df,
            data_in_path        = DATA_IN,
            project_file        = PROJECT_FILE,
            opex_sm_name_config = ARGS.opex_sm_name_config,
            feas_sm_name_config = ARGS.feas_sm_name_config,
            opex_sm_name        = ARGS.opex_sm_name,
            feas_sm_name        = ARGS.feas_sm_name,
            feas_sm_type        = ARGS.feas_sm_type,
            oc_min              = float(ARGS.oc_min)
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
