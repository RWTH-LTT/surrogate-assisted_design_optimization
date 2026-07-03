import os
import pickle
import time
import pandas as pd
from scipy.spatial import ConvexHull
from sm.surrogates_feasible_space import get_result_column_names, Feasibility_LP
from core.helpers import get_experiment_id

CASE             = "JP1"
PROJECT_FILE     = "project.yaml"
ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')

EXPERIMENT_ID = get_experiment_id(DATA_IN, PROJECT_FILE)

DATA_COMPUTED_IN = os.path.join(ROOT, 'data', CASE, 'ComputedInput_Data', EXPERIMENT_ID)
DATA_OUT = os.path.join(ROOT, 'data', CASE, 'Output_Data', EXPERIMENT_ID, )

os.makedirs(DATA_COMPUTED_IN, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

n_unique_list = [50, 100, 1000, 2000, 5000, 12746]

for n_unique in n_unique_list:

    print(f"{n_unique} unique points:")
    df_sample = pd.read_csv(os.path.join(DATA_OUT, f'convex_hull_vertex_results_{n_unique}.csv'))

    column_names_norm = get_result_column_names(
        DATA_IN, PROJECT_FILE, normed_keys=True)

    df_unique = df_sample.drop_duplicates(
        subset=column_names_norm, keep='first'
    )

    feasiblity_lp = Feasibility_LP(
        DATA_IN, PROJECT_FILE,
        df_vertices=df_unique
    )
    vertices = df_unique[column_names_norm].to_numpy()

    t0 = time.time()
    hull = ConvexHull(vertices, qhull_options='QJ')
    total_time = time.time() - t0

    print(f"\tdt = {total_time} s.")

    with open(os.path.join(DATA_OUT, f'hull_equations_{n_unique}.pkl'), 'wb') as f:

        pickle.dump(hull.equations, f)
