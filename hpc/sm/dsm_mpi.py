import pandas as pd
from core.helpers import get_experiment_id
from core.hpc_helpers import combine_batch_csv_files
from sm.routines import calculate_OPEX_for_batch, generate_aggregated_input_data
from mpi4py import MPI  # multi processing interface
import argparse
import os

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', default='dsm_LP_demo', help='the name to case for the input data', required=True)
parser.add_argument('--n_batch', default='2000', help='id of the current step', required=True)
parser.add_argument('--project_file', default='project.yaml', help='name of the .yaml file in data_in_path containing the input information', required=True)
parser.add_argument('--input_csv_name',  help='name of the csv file containing the uncertain parameter combinations, e.g. df_in.csv', required=True)
parser.add_argument('--output_name',  help='name of the output csv file, e.g. df_out (without.csv)', required=True)

args = parser.parse_args()

def main():
    
    CASE = args.case
    N_BATCH = int(args.n_batch)
    PROJECT_FILE = args.project_file
    CSV_INPUT_NAME = args.input_csv_name
    OUTPUT_NAME = args.output_name

    ROOT = os.getcwd()
    DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')
    EXPERIMENT_ID = get_experiment_id(DATA_IN, PROJECT_FILE)

    DATA_COMPUTED_IN = os.path.join(ROOT, 'data', CASE, 'ComputedInput_Data', EXPERIMENT_ID)
    DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', EXPERIMENT_ID, )
    os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Current process rank
    numP = comm.Get_size()  # Total number of processes
    print(f"rank={rank}")
    # Code to be executed only by the process with rank 0
    if rank == 0:
        print("Master process: Generating aggregated input data:")
        # Generate aggregated_input_data:
        generate_aggregated_input_data(DATA_IN, PROJECT_FILE)

    design_df = pd.read_csv(os.path.join(DATA_COMPUTED_IN, CSV_INPUT_NAME))  # Only rank 0 reads the data

    # Synchronize all processes to ensure the data is available to all before proceeding
    comm.Barrier()

    # Distribute batches among processes
    for idx_start in range(rank * N_BATCH, len(design_df), numP * N_BATCH):
        print(f"Rank {rank} processing batch starting at index {idx_start} ...")
        calculate_OPEX_for_batch(design_df, idx_start, N_BATCH, DATA_IN, PROJECT_FILE, os.path.join(DATA_OUT, OUTPUT_NAME))

    # Optional: Synchronize processes at the end
    comm.Barrier()

    if rank == 0:
        print("All batches processed.")
        combine_batch_csv_files(
            design_df, os.path.join(DATA_OUT, OUTPUT_NAME), N_BATCH,
            data_export_path=os.path.join(DATA_OUT, OUTPUT_NAME))
if __name__=='__main__':
    main()