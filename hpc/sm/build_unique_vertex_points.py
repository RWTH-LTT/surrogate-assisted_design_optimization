import argparse
import os
from core.helpers import get_experiment_id
from sm.surrogates_feasible_space import save_unique_vertex_points

# read inputs from command line
parser = argparse.ArgumentParser()
parser.add_argument('--case', help='the name to case for the input data', required=True)
parser.add_argument('--project_file', default='project.yaml', help='name of the .yaml file in data_in_path containing the input information')
parser.add_argument('--vertex_input_name',  help='name of the csv file containing all vertex points', required=True)
parser.add_argument('--vertex_output_name',  help='name of the output csv file containing unique vertex points', required=True)
parser.add_argument('--n_unique', help='number of unique vertex points', required=True)

ARGS = parser.parse_args()

CASE = ARGS.case
PROJECT_FILE = ARGS.project_file
ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', get_experiment_id(DATA_IN, PROJECT_FILE))


def main():
    save_unique_vertex_points(
        DATA_IN, PROJECT_FILE,
        full_vertex_csv_name    = ARGS.vertex_input_name,
        unique_vertex_name      = ARGS.vertex_output_name,
        n_unique_list           = [int(ARGS.n_unique)],
        add_all_unique_vertices = True
    )

if __name__=='__main__':
    main()
