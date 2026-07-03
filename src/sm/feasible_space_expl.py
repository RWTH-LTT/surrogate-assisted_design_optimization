""" Module for exploring the feasible space of a given energy system model using MGA."""

import os
import numpy as np
import pandas as pd
import yaml
from core.preprocessing import UnscaledInputDictGenerator, ScaledInputDictGenerator
from core.optimization import Optimization_LP
from core.helpers import UniqueKeyLoader
from sm.surrogates_feasible_space import get_min_max_values

def sample_min_max(ts_scaling_names:list, process_names:list) -> pd.DataFrame:
    """ Generates a DataFrame with vectors for min-max sampling.
    Args:
        ts_scaling_names (list): list of time series scaling names
        process_names (list): list of process names
    Returns:
        pd.DataFrame: DataFrame with vectors for min-max sampling.
    """
    # Calculate the number of vectors needed
    n = len(ts_scaling_names) + len(process_names)

    # Create vectors for each time series scaling and process
    vectors = []
    for i in range(n):
        vec = np.zeros(n)
        vec[i] = 1
        vectors.append(vec.copy())
        vec[i] = -1
        vectors.append(vec.copy())

    # Create a DataFrame with the vectors
    df = pd.DataFrame(np.array(vectors), columns=ts_scaling_names + process_names)
    # Add a column for the search mode
    df['search_mode'] = 'min_max'
    # Reorder columns to have 'search_mode' first
    return df[['search_mode'] + [c for c in df.columns if c != 'search_mode']]

def sample_most_critical_and_feasible(ts_scaling_names:list, process_names:list) -> pd.DataFrame:
    """ Generates a DataFrame with vectors for most critical and feasible sampling.
    Args:
        ts_scaling_names (list): list of time series scaling names
        process_names (list): list of process names
    Returns:
        pd.DataFrame: DataFrame with vectors for most critical and feasible sampling.
    """
    # Create vectors for most critical and feasible sampling
    most_crit = np.array([1]*len(ts_scaling_names) + [-1]*len(process_names))
    most_feas = -most_crit

    vectors = np.vstack([most_crit, most_feas])
    vectors = vectors / np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)

    # Create a DataFrame with the vectors
    df = pd.DataFrame( vectors, columns=ts_scaling_names + process_names)
    # Add a column for the search mode
    df['search_mode'] = ['most_crit', 'most_feas']
    # Reorder columns to have 'search_mode' first
    return df[['search_mode'] + [c for c in df.columns if c != 'search_mode']]

def sample_random(ts_scaling_names:list, process_names:list, n:int) -> pd.DataFrame:
    """ Generates a DataFrame with random vectors for sampling
    Args:
        ts_scaling_names (list): list of time series scaling names
        process_names (list): list of process names
        n (int): number of random vectors to generate
    Returns:
        pd.DataFrame: DataFrame with random vectors for sampling.
    """
    # Generate n random vectors with values between -1 and 1
    vectors = np.random.uniform(-1, 1, (n, len(ts_scaling_names) + len(process_names)))
    # Normalize each vector to have a norm of 1
    vectors = vectors / np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)

    # Create a DataFrame with the vectors
    df = pd.DataFrame(vectors, columns=ts_scaling_names + process_names)
    # Add a column for the search mode
    df['search_mode'] = 'random'
    # Reorder columns to have 'search_mode' first
    return df[['search_mode'] + [c for c in df.columns if c != 'search_mode']]

def running_unique_counts(arr:np.array) -> np.array:
    """ Returns the number of unique rows seen so far at each iteration.
    Args:
        arr (np.array):  2D numpy array of shape (M, N)
    Returns:
        np.array: 1D numpy array of length M
    """
    arr = np.ascontiguousarray(arr)
    M, N = arr.shape
    dtype = np.dtype((np.void, arr.dtype.itemsize * N))
    packed_rows = arr.view(dtype).reshape(M)

    seen = set()
    result = np.empty(M, dtype=np.int32)

    for i in range(M):
        row = packed_rows[i]
        # Make the row immutable & hashable
        seen.add(bytes(row))
        result[i] = len(seen)

    return result

class Convex_Hull_Explorer():
    """ Class for exploring the feasible space of a given energy system model using MGA.
    The class generates a Pyomo model for the convex hull optimization problem and provides
    methods to run the optimization for given search directions.
    """
    def __init__(self, data_in_path:str, project_file:str) -> None:
        """ 
        Args:
            data_in_path (str): path to the input files, where the project file lies
            project_file (str): name of the .yaml file in data_in_path containing the
                input information (e.g. project.yaml)
        """
        self.data_in_path = data_in_path
        self.project_file = project_file
        self._load_general_input_yamls()

        self.min_values, self.max_values = get_min_max_values(self.data_in_path, self.project_file)

        self.model, self.optimization = self._generate_model()

        self._normalize_objective_function()

    def _load_general_input_yamls(self):
        """ Loads the general input yamls from the data_in_path and stores them
        in the class attributes.
        """

        # Load project file:
        with open(os.path.join(self.data_in_path, self.project_file), 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)
        # Load config:
        with open(os.path.join(self.data_in_path,  self.project['config']),
                  'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=UniqueKeyLoader)
        # Load technologies data:
        with open(os.path.join(self.data_in_path,  self.project['technologies']),
                  'r', encoding='utf-8') as f:
            self.technologies = yaml.load(
                f, Loader=UniqueKeyLoader
                )['technologies']
        # Load timeseries scaling data:
        with open(os.path.join(self.data_in_path,  self.project['timeseries_scaling']),
                  'r', encoding='utf-8') as f:
            self.ts_scaling = yaml.load(
                f, Loader=UniqueKeyLoader
                )['timeseries_scaling'][self.config['timeseries_scaling']]

    def _generate_model(self) -> tuple:
        """ Generates a Pyomo model for the convex hull optimization problem amd sets the pyomo
        model as class variable.

        Returns:
            tuple: (pyomo.environ.ConcreteModel, core.optimization.Optimization_LP): the pyomo model
                and an optimization object
        """
        unscaled_input_dict = UnscaledInputDictGenerator(
            data_in_path = self.data_in_path,
            project_file = self.project_file,
            objective    = 'feasible_hull'
            ).generate_unscaled_input_dict()

        scaled_input_dict = ScaledInputDictGenerator(
            data_in_path        = self.data_in_path,
            project_file        = self.project_file,
            unscaled_input_dict = unscaled_input_dict
            ).generate_scaled_input_dict()

        optimization = Optimization_LP(
            data_in_path = self.data_in_path,
            project_file = self.project_file,
            input_dict   = scaled_input_dict
        )
        return (optimization.generate_model(), optimization)

    def _normalize_objective_function(self) -> None:
        """ Normalized the objective function by setting the min and max values.
        Theses values are either determined by min_max exploration before or are imported. 

        Raises:
            TypeError: If no min values or max values are set.
        """

        if self.min_values is None or self.max_values is None:
            raise TypeError("min and/or max values are not set yet.")

        # Production processes:
        for d in self.model.production_affect_feas:
            self.model.min_values_production[d] = self.min_values['production'][d]
            self.model.max_values_production[d] = self.max_values['production'][d]
        # Storage processes:
        for d in self.model.storage_affect_feas:
            self.model.min_values_storage[d] = self.min_values['storage'][d]
            self.model.max_values_storage[d] = self.max_values['storage'][d]
        # Demand mean values:
        for d in self.model.demand_mean_affect_feas:
            self.model.min_values_demand_mean[d] = self.min_values['demand_mean'][d]
            self.model.max_values_demand_mean[d] = self.max_values['demand_mean'][d]
        # Demand peak values:
        for d in self.model.demand_peak_affect_feas:
            self.model.min_values_demand_peak[d] = self.min_values['demand_peak'][d]
            self.model.max_values_demand_peak[d] = self.max_values['demand_peak'][d]

        print("\tNormalized search directions in objective function.")

    def run_optimization(self, df_sample:pd.DataFrame, idx, debug=False) -> pd.DataFrame:
        """ Runs one exploration optimization for the row 'idx' in df_sample and writes results
        (normalized and non normalized) to df_sample.

        Args:
            df_sample (pd.DataFrame): contains the search directions
            idx: name of the DataFrame index of the wanted row.
            debug (bool, optional): True activates debugging mode where the solver log is printed.
                In future version, the output_dict and plots are also generated in debugging mode.
                Defaults to False.

        Returns:
            pd.DataFrame: Contains the (normalized and non normalized) results.
        """

        # Update search direction in the model:
        # Production processes:
        for d in self.model.production_affect_feas:
            self.model.search_dir_production[d] = df_sample.loc[idx, d]
        # Storage processes:
        for d in self.model.storage_affect_feas:
            self.model.search_dir_storage[d] = df_sample.loc[idx, d]
        # Demand mean values:
        for d in self.model.demand_mean_affect_feas:
            self.model.search_dir_demand_mean[d] = df_sample.loc[idx, f"demand_{d}_mean"]
        # Demand peak values:
        for d in self.model.demand_peak_affect_feas:
            self.model.search_dir_demand_peak[d] = df_sample.loc[idx, f"demand_{d}_peak"]

        # Solve optimization:
        result = self.optimization.solve_model(
            model=self.model,
            tee = debug,
            symbolic_solver_labels=debug,
        )

        # Write unnormalized results:
        df_sample = self._add_unnormalized_results(df_sample, idx)

        # Write normalized results:
        df_sample = self._add_normalized_results(df_sample, idx)

        # Add solver time and termination condition:
        df_sample.loc[idx, 'solver_time'] = result['Solver'][0]['wall_time']
        df_sample.loc[
            idx, 'termination_condition'] = result['Solver'][0]['Termination condition'].value

        return df_sample

    def _add_unnormalized_results(self,  df_sample:pd.DataFrame, idx:str) -> pd.DataFrame:
        """ Adds the unnormalized results to a row of the df_sample DataFrame.

        Args:
            df_sample (pd.DataFrame): contains the input data
            idx (str): name of the index of the current DataFrame row

        Returns:
            pd.DataFrame: input data frame but with added unnormalized results
        """

        # Production processes:
        for d in self.model.production_affect_feas:
            df_sample.loc[idx, f'res_{d}'] = self.model.Production_nominal_capacity[d].value
        # Storage processes:
        for d in self.model.storage_affect_feas:
            df_sample.loc[idx, f'res_{d}'] = self.model.Storage_nominal_capacity[d].value
        # Demand mean values:
        for d in self.model.demand_mean_affect_feas:
            df_sample.loc[idx, f'res_demand_{d}_mean'] = self.model.Ts_demand_new_mean[d].value
        # Demand peak values:
        for d in self.model.demand_peak_affect_feas:
            df_sample.loc[idx, f'res_demand_{d}_peak'] = self.model.Ts_demand_new_peak[d].value

        return df_sample

    def _add_normalized_results(self, df_sample:pd.DataFrame, idx:str) -> pd.DataFrame:
        """ Adds the normalized results to a row of the df_sample DataFrame. The results are
        normalized using the min and max values from self.model.

        Args:
            df_sample (pd.DataFrame): contains the input data
            idx (str): name of the index of the current DataFrame row

        Returns:
            pd.DataFrame: input data frame but with added normalized results
        """

        # Production processes:
        for d in self.model.production_affect_feas:
            df_sample.loc[idx, f'res_{d}_norm'] = (
                (self.model.Production_nominal_capacity[d].value - self.model.min_values_production[d].value)/
                (self.model.max_values_production[d].value - self.model.min_values_production[d].value)
                )
        # Storage processes:
        for d in self.model.storage_affect_feas:
            df_sample.loc[idx, f'res_{d}_norm'] = (
                (self.model.Storage_nominal_capacity[d].value - self.model.min_values_storage[d].value)/
                (self.model.max_values_storage[d].value - self.model.min_values_storage[d].value)
                )
        # Demand mean values:
        for d in self.model.demand_mean_affect_feas:
            df_sample.loc[idx, f'res_demand_{d}_mean_norm'] = (
                (self.model.Ts_demand_new_mean[d].value - self.model.min_values_demand_mean[d].value)/
                (self.model.max_values_demand_mean[d].value - self.model.min_values_demand_mean[d].value)
                )
        # Demand peak values:
        for d in self.model.demand_peak_affect_feas:
            df_sample.loc[idx, f'res_demand_{d}_peak_norm'] = (
                (self.model.Ts_demand_new_peak[d].value - self.model.min_values_demand_peak[d].value)/
                (self.model.max_values_demand_peak[d].value - self.model.min_values_demand_peak[d].value)
                )

        return df_sample
