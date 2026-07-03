""" Module to generate samples for surrogate model training. """
import itertools
import os
from pathlib import Path
import warnings
from copy import deepcopy
import yaml
import pandas as pd
import numpy as np
from scipy.stats import qmc #Quasi-Monte-Carlo
from core.helpers import (
    UniqueKeyLoader, get_minimal_capacity, get_potential_capacity, get_experiment_id)
from core.ts_scaling import sanity_check_ts, get_scale_mode_ts
from sm.routines import generate_aggregated_input_data

class SampleGenerator():
    """ Generates samples which are intended to be used to train surrogate models.
    To generate samples for Design Surrogates:
        * Input: timeseries, Outputs: process_capacites and KPIs (OPEX, ...)
        * sample_processes must be False.
        * sample_timeseries must be True.
    To generate samples for OPEX Surroagte:
        * Inputs timeseries and process_capacites, Output: KPIs (OPEX, ...)
        * sample_processes must be True.
        * sample_timeseries CAN be True or False, depending on the use case.

    Attributes
    ----------
    general:
        data_in_path (str): Path to the input data
        project (dict): contains project.yaml.
        config (dict): contains config.yaml.
        technologies (dict): contains technologies.yaml.
        superstructure (dict): contains superstructure.yaml.
        ts_scaling (dict): contains timeseries_scaling.yaml.
        timeseries (dict): contains timeseries.yaml.
    processes:
        sample_processes (bool): If processes are sampled as input data.
        processes_fixed (dict): contains all processes with a constant capacity.
        processes_custom_bounds (dict): specifies custom bounds for specific processes.
    timeseries:
        sample_timeseries (bool): If timeseries are sampled as input data.
        timeseries_ignored (list): specifies which ts_names will not be sampled.
    

    Methods
    ----------
    sample_grid(points_per_dimension,resample=True, add_vertices=True, data_export_path=None):
        Generates grid sample.
    sample_lhs(n_points, seed=0, scramble=True, resample=True, add_vertices=True,
        data_export_path=None):
        Generates Latin Hypercube Sample.
    sample_random(n_points, seed=0, resample=True, add_vertices=True, data_export_path=None):
        Generates Random Uniform Sample.
    sample_vertices(debug=False):
        Samples the vertices of the sampling space.
    set_processes_fixed(processes_fixed):
        sets processes with fixed capacities, overwriting ranges in technologies.yaml.
    set_processes_custom_bounds(processes_custom_bounds):
        sets custom bound for processes, processes_custom_bounds (dict):
    set_timeseries_ignore_list(timeseries_ignored):
        sets ts_names which will be ignored.
    """
    def __init__(
        self,
        data_in_path,
        project_file:str,
        sample_processes:bool=True,
        sample_timeseries:bool=True
        ):
        """
        This class is used to generate the samples to train the surrogates.

        Args:
            data_in_path : Path to the input data
            project_file (str): name of the project file (e.g., 'project.yaml')
            sample_processes(bool): If processes are sampled as input data. Intended use is for
                operational optimization. The processes are taken from superstructure.yaml.
                Custom bounds can be used via set 'SampleGenerator.set_processes_custom_bounds()'.
                Processes, which should not be sampled and be fixed instead can be specified via
                'SampleGenerator.set_processes_fixed()'.
                Defaults to True.
            sample_timeseries (bool): If timeseries are sampled as input data. Can be used for
                design as well as operational optimization. The sampled timeseries are taken from
                timeseries_scaling.yaml. Custom bounds cannot be used at the moment, the function is
                not implemented yet. Timeseries, which should not be sampled (thus be ignored) can
                be specified via 'SampleGenerator.timeseries_ignored()'.
        """

        self.data_in_path      = data_in_path
        self.project_file      = project_file
        self.sample_processes  = sample_processes
        self.sample_timeseries = sample_timeseries

        self.processes_fixed = {}
        self.timeseries_ignored = []
        self.processes_custom_bounds = {'min': {}, 'max': {}}

        self._load_general_input_yamls(project_file)

        # Check, at least one of sample_processes and sample_timeseries is True:
        if not (self.sample_processes or self.sample_timeseries):
            raise ValueError(
                "At least one of 'sample_processes' and 'sample_timeseries' must be True.")

        # Prepare timeseries sampling, if sample_timeseries is selected:
        if self.sample_timeseries:
            sanity_check_ts(self.ts_scaling)

        # Sanity check, such that no technology shares the name with an ts_scaling
        if any(process in self.ts_scaling for process in self.technologies):
            raise ValueError(
                "At least one process name (superstructure.yaml) clashes with an " +
                "ts_scaling name (timeseries_scaling.yaml).")
    ####################################################################################
    ####################                General functions           ####################
    ####################################################################################

    def sample_grid(self, points_per_dimension:int,
        resample:bool=True, add_vertices:bool=True, data_export_path:str=None
        ):
        raise NotImplementedError("Grid sampling is not implemented yet.")

    def sample_lhs(
        self, n_points:int, seed:int=None, scramble:bool=True,
        resample:bool=True, add_vertices:bool=True, data_export_path:str=None
        ) -> pd.DataFrame:
        """ Generates a Latin Hypercube Sample.

        Args:
            n_points (int): number of points for the LHS sampling
            seed (int, optional): Seed for the LHS algorithm. Defaults to None.
            scramble (bool, optional): If LHS scrambles its outputs. Defaults to True.
            resample (bool, optional): If True, timeseries values are resampled, where 'mean'>'peak.
                Defaults to True.
            add_vertices (bool, optional): If True, all vertex points are added. Defaults to True.
            data_export_path (str, optional): If not None, the generated sample is exported to the
                given path. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the LHS samples
        """

        # Set dimension as sum of timeseries scalers and sampled processes:
        dimension = (
            len(self._generate_timeseries_splitted())+ len(self._generate_processes_sampled())
            )

        # Set lower and upper bounds:
        lower_bound, upper_bound = self._get_lower_and_upper_bound()

        sampler = qmc.LatinHypercube(d=dimension, scramble=scramble, seed=seed)
        sample = sampler.random(n=n_points)

        sample_scaled = qmc.scale(sample, lower_bound, upper_bound)

        sample_df = self._sample_postprocessing(
            sample_scaled, resample, add_vertices, data_export_path)

        return sample_df

    def sample_random(
        self, n_points:int, seed:int=None,
        resample:bool=True, add_vertices:bool=True, data_export_path:str=None
        ):
        """ Generates a random (uniform) samples.

        Args:
            n_points (int): number of points for the random 0sampling
            seed (int, optional): Seed for numpy.random. Defaults to None.
            resample (bool, optional): If True, timeseries values are resampled, where 'mean'>'peak.
                Defaults to True.
            add_vertices (bool, optional): If True, all vertex points are added. Defaults to True.
            data_export_path (str, optional): If not None, the generated sample is exported to the
                given path. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the uniform samples
        """

        # Set dimension as sum of timeseries scalers and sampled processes:
        dimension = (
            len(self._generate_timeseries_splitted())+ len(self._generate_processes_sampled())
            )

        # Set lower and upper bounds:
        lower_bound, upper_bound = self._get_lower_and_upper_bound()

        if seed is not None:
            np.random.seed(seed)

        sample = np.random.rand(n_points, dimension)
        sample_scaled = qmc.scale(sample, lower_bound, upper_bound)

        sample_df = self._sample_postprocessing(
            sample_scaled, resample, add_vertices, data_export_path)

        return sample_df

    def _sample_postprocessing(self, sample_scaled:np.array, resample:bool, add_vertices:bool,
                               data_export_path:str) -> pd.DataFrame:
        """ Function for conducting postprocessing steps after uniform or LHS sampling.

        Args:
            sample_scaled (np.array): random (LHS or uniform) sample
            resample (bool): If True, timeseries values are resampled, where 'mean'>'peak.
            add_vertices (bool): If True, all vertex points are added.
            data_export_path (str): If not None, the generated sample is exported to the given path.

        Returns:
            pd.DataFrame: Sampled data points as DataFrame.
        """
        # TS: Split timeseries into mean and peak values
        ts_dict_split = self._generate_timeseries_splitted()

        # PR: Generate processes sampled list:
        pr_list_sampled = self._generate_processes_sampled()

        sample_df = pd.DataFrame(
            sample_scaled,
            columns= list(ts_dict_split.keys()) + pr_list_sampled
        )

        # Resample samples with peak < mean
        if resample:
            sample_df = self.resample_peak_and_mean_ts(sample_df)

        # Add vertices:
        if add_vertices:
            sample_df = pd.concat([self.sample_vertices(), sample_df], ignore_index=True)

        # Add processes that are not sampled with their defined capacities:
        if self.processes_fixed and self.sample_processes:
            for process, capacity in self.processes_fixed.items():
                sample_df[process] = capacity

        # Add first row with id:
        sample_df.insert(0, 'id', range(len(sample_df)))

        if data_export_path is not None:
            export_file = Path(data_export_path).with_suffix('.csv')
            print(f"\tExporting Random sampling data {export_file}")
            sample_df.to_csv(export_file, index=False)

        return sample_df

    def _load_general_input_yamls(self, project_file:str,) -> None:
        """ Loads the general input yamls from the data_in_path and stores them
        in the class attributes.

        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
        """

        # Load project file:
        filename = os.path.join(self.data_in_path, project_file)
        with open(filename, 'r', encoding='utf-8') as file:
            self.project = yaml.load(file, Loader=UniqueKeyLoader)

        # Load config:
        filename = os.path.join(self.data_in_path,  self.project['config'])
        with open(filename, 'r', encoding='utf-8') as file:
            self.config = yaml.load(file, Loader=UniqueKeyLoader)

        # Load technologies:
        filename = os.path.join(self.data_in_path,  self.project['technologies'])
        with open(filename, 'r', encoding='utf-8') as file:
            self.technologies = yaml.load(file, Loader=UniqueKeyLoader)['technologies']

        # Load superstructure:
        filename = os.path.join(self.data_in_path,  self.project['superstructures'])
        with open(filename, 'r', encoding='utf-8') as file:
            self.superstructure = yaml.load(
                file, Loader=UniqueKeyLoader
                )['superstructures'][self.config['superstructure_name']]

        # Load ts_scaling:
        filename = os.path.join(self.data_in_path,  self.project['timeseries_scaling'])
        with open(filename, 'r', encoding='utf-8') as file:
            self.ts_scaling = yaml.load(
                file, Loader=UniqueKeyLoader
                )['timeseries_scaling'][self.config['timeseries_scaling']]

        # Load timeseries data:
        filename = os.path.join(self.data_in_path,  self.project['timeseries'])
        with open(filename, 'r', encoding='utf-8') as file:
            self.timeseries = yaml.load(file, Loader=UniqueKeyLoader)

    def sample_vertices(self, debug:bool=False) -> pd.DataFrame:
        """ Samples all vertices of the convex hull

        Args:
            debug (bool, optional): If True, the added vertices are printed.
                Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing all vertices
        """

        print("\tSampling vertices of the input space.")

        # PR: Generate processes sampled list: 
        pr_list_sampled = self._generate_processes_sampled()
        # TS: Generate timeseries sampled list:
        ts_list_sampled = self._generate_timeseries_sampled()

        # Get the possible variables:
        variables = (
            [ # variables combination for timeseries
                self._get_vertices_ts(ts_name, debug)[1]
                for ts_name in ts_list_sampled
            ]
            + [ # variables combination for processes
                [[self._get_process_lower_bound(process)], [self._get_process_upper_bound(process)]]
                for process in pr_list_sampled
            ]
        )

        # Get the column names:
        column_names = (
            [ # column names for timeseries
                self._get_vertices_ts(ts_name)[0]
                for ts_name in ts_list_sampled
            ]
            + [pr_list_sampled] # column names of processes
        )

        combinations = list(itertools.product(*variables))

        # Flatten lists:
        column_names_flattend = [item for sublist in column_names for item in sublist]
        variables_flattend = [
            [item  for subsublist in sublist for item in subsublist]
            for sublist in combinations]

        print(f"\t   Sampled {len(variables_flattend)} vertices.")

        # Return DataFrame:
        return pd.DataFrame(variables_flattend, columns=column_names_flattend)

    def _get_lower_and_upper_bound(self):
        """_summary_

        Returns:
            tuple: (lower_bound, upper_bound)
                lower_bound (list): list of lower bounds for each dimension
                upper_bound (list): list of upper bounds for each dimension
        """
        # TS: Split timeseries into mean and peak values
        ts_dict_split = self._generate_timeseries_splitted()

        # PR: Generate processes sampled list:
        pr_list_sampled = self._generate_processes_sampled()

        # Set lower bounds:
        lower_bound = (
            [val['min'] for val in ts_dict_split.values()] # timeseries
            + [self._get_process_lower_bound(process) for process in pr_list_sampled] # processes
        )

        # Set upper bounds:
        upper_bound = (
            [val['max'] for val in ts_dict_split.values()] # timeseries
            + [self._get_process_upper_bound(process) for process in pr_list_sampled] # processes
        )

        return (lower_bound, upper_bound)

    ####################################################################################
    ####################         Sample Processes (PR)              ####################
    ####################################################################################

    def set_processes_fixed(self, processes_fixed:dict) -> None:
        """ sets self.set_processes_fixed after conductin a sanity check.

        Args:
            processes_fixed (dict, optional): dict containing all processes that should
                have a constaint capacity. e.g. {'B': 400, 'HTHP': 0, 'battery': 1000}.
        """

        # Raise Error for non-empty processes_fixed, if self.sample_processes is False
        if processes_fixed and not self.sample_processes:
            raise ValueError("set_processes_fixed must not be used, if sample_processes is False.")

        # Ensure all processes in self.processes_fixed exist in the superstructure
        assert all(key in self.superstructure for key in processes_fixed), \
                "One or more processes in self.processes_fixed are not in the superstructure."

        # Ensure that the capacities in self.processes_fixed are numeric
        assert all(isinstance(value, (int, float)) for value in processes_fixed.values()), \
                "All capacities in self.processes_fixed must be numeric."

        # Validate that capacities fall within the defined minimal and potential capacities
        for process, capacity in processes_fixed.items():
            min_capacity = get_minimal_capacity(self.technologies[process])
            max_capacity = get_potential_capacity(self.technologies[process])

            if capacity < min_capacity and capacity!=0:
                warnings.warn(
                    f"'{process}' has a capacity of '{capacity}' kW (or kWh), " +
                    f"which is below the minimum of '{min_capacity}' kW (or kWh).")

            if capacity > max_capacity:
                warnings.warn(
                    f"'{process}' has a capacity of '{capacity}' kW (or kWh), " +
                    f"which exceeds the potential maximum of '{max_capacity}' kW (or kWh).")

        self.processes_fixed = processes_fixed

    def set_processes_custom_bounds(self, processes_custom_bounds:dict) -> None:
        """ Sets custom bounds for processes. If not bounds are given for 'min' or 'max', an empty
        dict is generated. Conducts santiy checks for min and max values

        Args:
            processes_custom_bounds (dict): contains the custom bounds in kW or kWh, e.g.,
                {
                    'max': {'B':15000, 'TES': 14000},
                    'min': {'B':0}
                }
        """
        # Copy processes_custom_bounds
        self.processes_custom_bounds = deepcopy(processes_custom_bounds)

        # add 'min' to custom bounds, if not already existend:
        if 'min' not in self.processes_custom_bounds:
            self.processes_custom_bounds['min'] = {}

        # add 'max' to custom bounds, if not already existend:
        if 'max' not in self.processes_custom_bounds:
            self.processes_custom_bounds['max'] = {}

        # Do sanity checks:
        self._sanity_check_processes_custom_bounds()

    def _sanity_check_processes_custom_bounds(self) -> None:
        """ Conducts sanity check for lower (min) or upper (max) bound of
        self.processes_custom_bounds.

        Raises:
            ValueError: If a process is not in self.technologies.
            ValueError: If a bound is negative.
            ValueError: If a bound is larger than the potential capacity.
            ValueError: If the lower bound is larger than the upper bound.
        """
        # Check, that processes_custom_bounds is a dict:
        if not isinstance(self.processes_custom_bounds, dict):
            raise TypeError("processes_custom_bounds must be a dict.")

        # Check, that only 'min' and 'max' are keys.
        invalid_keys = [k for k in self.processes_custom_bounds if k not in {'min', 'max'}]
        if invalid_keys:
            raise KeyError(
                f"Invalid key(s) in processes_custom_bounds: '{invalid_keys}'. " +
                "Only 'min' and 'max' are allowed.")

        for min_or_max in ['min', 'max']:
            for process, sizing in self.processes_custom_bounds[min_or_max].items():
                # 1. Check that the process exists:
                if process not in self.technologies:
                    raise ValueError(
                        f"process '{process}' used in processes_custom_bounds is not in " +
                        "self.technologies.")

                # 2. Check, that sizing is non-negative:
                if sizing < 0:
                    raise ValueError(
                        f"{min_or_max}_sizing of '{process}' in processes_custom_bounds " +
                        f"must not be negative (but is '{sizing}').")

                # 3. If process is in processes fixed: give warning:
                if process in self.processes_fixed:
                    warnings.warn(
                        f"'{process}' is in processes_custom_bounds and in processes_fixed. " +
                        "It will not be sampled.")

                # Only for max:                
                if min_or_max == 'max':
                    # 4. Check, that sizing <= potential_capacity:
                    pot_capacity = get_potential_capacity(self.technologies[process])
                    if sizing > pot_capacity:
                        raise ValueError(
                            f"max_sizing of '{process}' in processes_custom_bounds must not be " +
                            f"larger than its potential capacity ( '{sizing}' > '{pot_capacity}').")

                    # 5. Check, that min_sizing is not larger than max_sizing (if min_sizing exists)
                    min_sizing = self.processes_custom_bounds['min'].get(process, 0)
                    # Check only, if min_sizing exists
                    if min_sizing >= sizing:
                        raise ValueError(
                            f"process '{process}': min_sizing must be smaller than max_sizing. " +
                            f"('{min_sizing}' < '{sizing}')")

                # Only for min:
                if min_or_max == 'min':
                    # 6. Check, that sizing < potential_capacity:
                    pot_capacity = get_potential_capacity(self.technologies[process])
                    if sizing >= pot_capacity:
                        raise ValueError(
                            f"min_sizing of '{process}' in processes_custom_bounds must be " +
                            f"smaller than its potential capacity ( {sizing} >= {pot_capacity}).")

    def _generate_processes_sampled(self) -> list:
        """ Generates and returns processes_sampled.

        Returns:
            list: containes all process names from self.technologies that are not in processes_fixed
        """
        # If processes are not sampled, return empty list
        if not self.sample_processes:
            return  []

        if self.processes_fixed is None:
            # If no processes are excluded, return all processes in the superstructure
            return list(self.superstructure.keys())

        # Return all processes that are not excluded in self.processes_fixed:
        return [
            process for process in self.superstructure
            if process not in self.processes_fixed
        ]

    def _get_process_lower_bound(self, process) -> float:
        """ Returns the lower bound of a process sizing.
        Reads from self.processes_custom_bounds['min']

        Args:
            process (str): name of the process

        Returns:
            float: lower bound of a process sizing
        """

        return self.processes_custom_bounds['min'].get(process, 0)

    def _get_process_upper_bound(self, process) -> float:
        """ Returns the upper bound of a process sizing.
        Reads from self.processes_custom_bounds['max']. If process is not found there,
            the potential capacity is returned.

        Args:
            process (str): name of the process

        Returns:
            float: upper bound of a process sizing
        """

        return self.processes_custom_bounds['max'].get(
            process,
            get_potential_capacity(self.technologies[process])
        )

    ####################################################################################
    ####################        Sample Timeseries (TS)              ####################
    ####################################################################################

    def set_timeseries_ignore_list(self, timeseries_ignored:list) -> None:
        """ Sets self.timeseries_ignore after conducting sanity checks.
        The ts_names of timeseries_ignore will not be used in the scenario generation.
        Thus, they stay at their default value (which can be found in the respective .csv in
        InputData.)

        Args:
            timeseries_ignored (list): contains names of the timeseries, which should stay at the
                default value. e.g., timeseries_ignored = ['ts_demand_cold', 'ts_avail_pv']
        """
        # Sanity checks
        # 1. Check, that timeseries_ignored is either None or list
        if not isinstance(timeseries_ignored, list):
            raise TypeError(
                "Type of timeseries_ignored must be 'list'. Instead it is "+
                f"'{type(timeseries_ignored)}'.")

        # 2. Check, that every entry in timeseries_ignored is a string:
        if any(not isinstance(ts_name, str) for ts_name in timeseries_ignored):
            raise TypeError(
                "'timeseries_ignored' must be filled with str. Please check the list again.")

        # 3. Check, that every ts_name in timeseries_ignored is in self.ts_scaling:
        if any(ts_name not in self.ts_scaling for ts_name in timeseries_ignored):
            raise KeyError("At least one key of timeseries_ignored is not in self.ts_scaling.")

        # 4 Warn, if self.timeseries is not empty (and will be now overwritten):
        if len(self.timeseries_ignored) > 0:
            warnings.warn(
                "self.timeseries_ignored is not empty but will now be overwritten from " +
                f"{self.timeseries_ignored} to {timeseries_ignored}." +
                "\n\tKeep in mind, that old ignored timeseries are not considered.")

        self.timeseries_ignored = timeseries_ignored

    def resample_peak_and_mean_ts(self, lhs_sample:pd.DataFrame) -> pd.DataFrame:
        """ Resamples peak values, where peak < mean. 

        Args:
            lhs_sample (pd.DataFrame): DataFrame containing the sample without
                resampling

        Returns:
            pd.DataFrame: resampled DataFrame
        """
        print("\tResampling peak and mean...")

        # Return lhs_sample, if self.sample_timeseries is False:
        if not self.sample_timeseries :
            return lhs_sample

        # Copy input DataFrame:
        lhs_sample = lhs_sample.copy()

        # Initialize counter:
        counter = 0

        # Loop over list of sampled ts_names:
        for ts_name in self._generate_timeseries_sampled():
            ts_dict = self.ts_scaling[ts_name]

            # Continue if 'ts_type' is 'mean' or 'interpolate':
            if get_scale_mode_ts(ts_dict) != 'mean_and_peak':
                continue

            key_peak = ts_dict['param']['peak']['key_name']
            key_mean = ts_dict['param']['mean']['key_name']
            # Loop over all samples:
            for i in range(len(lhs_sample)):

                if lhs_sample.loc[i, key_peak] <  lhs_sample.loc[i, key_mean]:
                    counter += 1

                while lhs_sample.loc[i, key_peak] <  lhs_sample.loc[i, key_mean]:
                    # Randomly resample Peak
                    lhs_sample.loc[i, key_peak] =  np.random.uniform(
                        ts_dict['param']['peak']['min'],
                        ts_dict['param']['peak']['max'])
                     # Randomly resample Mean
                    lhs_sample.loc[i, key_mean] =  np.random.uniform(
                        ts_dict['param']['mean']['min'],
                        ts_dict['param']['mean']['max'])

        print(f"\tResampled {int(counter)} peak values.")

        return lhs_sample

    def report_extreme_ts(self, report_aggregated_timeseries:bool = True) -> None:
        """
            report_aggregated_timeseries (bool): True: Aggregated ts data is created and reported.
                Defaults to True.
        """

        if not self.sample_timeseries:
            print("No timeseries are scaled.")
            return

        # Generate aggregated data if necessary:
        if report_aggregated_timeseries:
            generate_aggregated_input_data(self.data_in_path, self.project_file)
            ts_path = os.path.join(
                self.data_in_path,
                'timeseries_tsa_export',
                get_experiment_id(self.data_in_path, self.project_file))
            # Load tsa_weights (to calculate the correct mean in later steps)
            with open(os.path.join(ts_path, 'time_slice_weight.yaml'), 'rb') as f:
                tsa_weights = yaml.load(f, Loader=UniqueKeyLoader)
            assert tsa_weights is not None

        else:
            print("Reporting non-aggregated data:")
            ts_path = self.data_in_path
            # Set tsa_weights to None, as they are not needed for non-aggregated data
            tsa_weights = None


        for timeseries in self.ts_scaling:

            if timeseries in self.timeseries_ignored:
                self._report_ignored_timeseries(timeseries, ts_path, tsa_weights)
                continue

            self._report_nonignored_timeseries(timeseries, ts_path, tsa_weights)

    def _report_ignored_timeseries(self, timeseries:str, ts_path:str, tsa_weights:list) -> None:
        """ Reports min, max and meand values for ignored timeseries

        Args:
            timeseries (str): name of the timeseries
            ts_path (str): path to the timeseries data
        """
        assert timeseries in self.timeseries_ignored

        # Read csv:
        ts_data = self._load_ts_csv(
            ts_path,
            self.ts_scaling[timeseries]['ts_type'],
            self.ts_scaling[timeseries].get(
                'process', self.ts_scaling[timeseries].get('product'))
            )

        # If Timeseries are Non-aggregated:
        if tsa_weights is None:
            ts_data_mean = ts_data.mean()
        # Else: Timeseries are aggregated:
        else:
            ts_data_mean = sum(
                ts_data[i] * tsa_weights[i] for i in tsa_weights
                ) / sum(tsa_weights.values())

        # Report
        print(f"\n{timeseries}: ignored (= timeseries is not scaled):")
        print(f"\tmax: {ts_data.max():.3f}, mean: {ts_data_mean:.3f}, min: {ts_data.min():.6f}")

    def _report_nonignored_timeseries(
            self, ts_of_interest:str, ts_path:str, tsa_weights:list) -> None:
        """ Reports min, max and meand values for each vertex point of the
        timeseries of interest (ts_of_interest)

        Args:
            ts_of_interest (str): name of the timeseries of interest
            ts_path (str): path to the timeseries data
        """
        assert ts_of_interest not in self.timeseries_ignored

        scale_mode = get_scale_mode_ts(self.ts_scaling[ts_of_interest])
        # skip, if scale_mode == interpolate
        if scale_mode == 'interpolate':
            print(f"\n'interpolate' is currently not reported but present for '{ts_of_interest}'.")
            return

        print(f"\n{ts_of_interest}: {scale_mode} are scaled.")

        # Read csv:
        ts_data = self._load_ts_csv(
            ts_path,
            self.ts_scaling[ts_of_interest]['ts_type'],
            self.ts_scaling[ts_of_interest].get(
                'process', self.ts_scaling[ts_of_interest].get('product'))
            )

        min_old = ts_data.min()
        peak_old = ts_data.max()

        # If Timeseries are Non-aggregated:
        if tsa_weights is None:
            mean_old = ts_data.mean()
        # Else: Timeseries are aggregated:
        else:
            mean_old = sum(
                ts_data[i] * tsa_weights[i] for i in tsa_weights
                ) / sum(tsa_weights.values())

        print("\tOriginal timeseries:")
        print(f"\tmax: {peak_old:.4f}, mean: {mean_old:.4f}, min: {min_old:.6f}")

        # Create temporary Object of SampleGenerator, to create vertex points
        # only for ts_of_interest:
        temp_sg = SampleGenerator(
            self.data_in_path,
            self.project_file,
            sample_processes=False,
            sample_timeseries=True
        )
        # In temp_sg: ignore each timeseries except for the timeseries of interest:
        temp_sg.set_timeseries_ignore_list(
            [ts for ts in self.ts_scaling if ts != ts_of_interest])
        # In temp_sg: generate sample with vertex points
        temp_sample = temp_sg.sample_lhs(
            n_points=1,
            resample=False,
            add_vertices=True
        ).iloc[:-1]

        # Iterate of temp_sample:
        for (_, temp_series) in temp_sample.iterrows():

            # Read mean_new:
            mean_new = temp_series[self._get_key_name_ts(ts_of_interest, 'mean')]

            if scale_mode == 'mean':
                a = mean_new / mean_old
                b = 0
            elif scale_mode == 'mean_and_peak':
                # Read peak_new:
                peak_new = temp_series[self._get_key_name_ts(ts_of_interest, 'peak')]
                a = (peak_new - mean_new) / (peak_old - mean_old)
                b = mean_new - a * mean_old
            elif scale_mode == 'mean_and_relpeak':
                # Read peak_new:
                relpeak_new = temp_series[self._get_key_name_ts(ts_of_interest, 'relpeak')]
                peak_new = relpeak_new * mean_new
                a = (peak_new - mean_new) / (peak_old - mean_old)
                b = mean_new - a * mean_old
            else:
                raise ValueError(f"Invalid scalemode '{scale_mode}'.")

            print(f"\tmax: {peak_old *a + b:.2f}, mean: {mean_new:.2f}, min: {min_old * a + b:.6f}")

    def _generate_timeseries_sampled(self) -> list:
        """ Generates and returns timeseries_sampled.

        Returns:
            list: containes all ts_names from self.ts_scaling which are not in timeseries_ignored
        """
        # If timeseries are not sampled, return empty list
        if not self.sample_timeseries:
            return  []

        if self.timeseries_ignored is None:
            print("'timeseries_ignored' should not be None, please doublecheck")
            # If no timeseries are excluded, return all timeseries in ts_scaling
            return list(self.ts_scaling.keys())

        # Return all processes that are not excluded in self.processes_fixed:
        return [
            ts for ts in self.ts_scaling
            if ts not in self.timeseries_ignored
        ]

    def _get_key_name_ts(self, ts_name:str, mean_or_peak:str) -> str:
        """ Get the key name for the timeseries in the dataframe.

        Args:
            ts_name (str): name of the timeseries
            mean_or_peak (str): 'mean' or 'peak'

        Returns:
            str: key name which will we used as column name in the dataframe
        """
        if mean_or_peak not in {'mean', 'peak', 'relpeak', 'interpolate'}:
            raise ValueError(
                f"mean_or_peak = '{mean_or_peak}' is not 'mean', 'peak', 'relpeak', 'interpolate'."
            )
        ts_dict = self.ts_scaling[ts_name]

        # If ts_type is availability, use process in the key:
        if ts_dict['ts_type'] == 'availability':
            process_or_product = 'process'
            assert mean_or_peak == 'interpolate'

        # Else, use product in the key:
        else:
            process_or_product = 'product'

        return f"{ts_dict['ts_type']}_{ts_dict[process_or_product]}_{mean_or_peak}"

    def _get_vertices_ts(self, ts_name:str, debug:bool=False) -> tuple:
        """ Returns all vertices of a given timeseries.

        Args:
            ts_name (str): name of the timeseries in ts_scaling.yaml
            debug (bool, optional): If True, the added vertices are printed.
                Defaults to False.

        Returns:
            (list, list): (nested list of column_names, nested list of all vertex combination)
        """

        scale_mode = get_scale_mode_ts(self.ts_scaling[ts_name])
        # First case: scale_mode is 'mean':
        if scale_mode == 'mean':
            return self._get_vertices_ts_only_mean(ts_name, debug)

        # Second case: scale_mode is 'mean_and_peak':
        if scale_mode == 'mean_and_peak':
            return self._get_vertices_ts_mean_and_peak(ts_name, debug)

        # Third case: scale_mode is 'mean_and_relpeak':
        if scale_mode == 'mean_and_relpeak':
            return self._get_vertices_ts_mean_and_relpeak(ts_name, debug)

        # Forth case: scale_mode is 'interpolate':
        if scale_mode == 'interpolate':
            return self._get_vertices_ts_interpolate(ts_name, debug)

        # Raise error if there is no match:
        raise ValueError(
            f"'{ts_name}': scale_mode must be in 'mean', 'peak', or 'interpolate'. "+
            f"Got '{scale_mode}' instead."
        )

    def _get_vertices_ts_only_mean(self, ts_name:str, debug:bool) -> tuple:
        """ Returns all vertices of a given timeseries.
        The indened use is for scale_mode == 'mean'

        Args:
            ts_name (str): name of the timeseries in ts_scaling.yaml
            debug (bool): If True, the added vertices are printed.

        Returns:
            (list, list): (nested list of column_names, nested list of all vertex combination)
        """
        ts_dict = self.ts_scaling[ts_name]
        mean_min = ts_dict['param']['mean']['min']
        mean_max = ts_dict['param']['mean']['max']

        key_names = [self._get_key_name_ts(ts_name, 'mean')]

        # If max > min, return [min, max], else [max]
        if mean_min < mean_max:
            # Print infos in debug-mode
            if debug:
                print(f"\t   Getting mean vertices for '{ts_name}':")
                print("\t       Adding min and max")
            return (key_names, [[mean_min], [mean_max]])

        if mean_min == mean_max:
            # Print infos in debug-mode
            if debug:
                print(f"\t   Getting mean vertices for '{ts_name}':")
                print("\t       Adding min")
            return (key_names, [[mean_min]])

        raise ValueError(
            f"min(mean) > max(mean) for '{ts_name}'"
        )

    def _get_vertices_ts_mean_and_peak(self, ts_name:str, debug:bool) -> tuple:
        """ Returns all vertices of a given timeseries.
        The indened use is for scale_mode == 'mean_and_peak'

        Args:
            ts_name (str): name of the timeseries in ts_scaling.yaml
            debug (bool): If True, the added vertices are printed.

        Returns:
            (list, list): (nested list of column_names, nested list of all vertex combination)
        """

        ts_dict = self.ts_scaling[ts_name]
        mean_min = ts_dict['param']['mean']['min']
        mean_max = ts_dict['param']['mean']['max']
        peak_min = ts_dict['param']['peak']['min']
        peak_max = ts_dict['param']['peak']['max']

        # Initialize empty list with point A
        # Point A: [mean: min(mean) | peak: max(peak)]
        vertices = [ [mean_min, peak_max] ]
        if debug:
            print(f"\t   Getting mean_and_peak vertices for '{ts_name}':")
            print(f"\t       Adding point A: {vertices[-1]}")

        # Point B: [mean: max(mean) | peak: max(peak)]
        # Is only added, if max(mean) > min(mean)
        if mean_max > mean_min:
            vertices.append([mean_max, peak_max])
            if debug:
                print(f"\t       Adding point B: {vertices[-1]}")

        # Point C: [mean: max(mean) | peak: max(mean)]
        # Is only added, if max(peak) > max(mean) > min(peak)
        if peak_max > mean_max > peak_min:
            vertices.append([mean_max, mean_max])
            if debug:
                print(f"\t       Adding point C: {vertices[-1]}")

        # Point D: [mean: min(peak) | peak: min(peak)]
        # Is only added, if max(mean) > min(peak) > min(mean)
        if mean_max > peak_min > mean_min:
            vertices.append([peak_min, peak_min])
            if debug:
                print(f"\t       Adding point D: {vertices[-1]}")

        # Point E: [mean: min(mean) | peak: min(peak)]
        # Is only added, if max(peak) > min(peak)
        if peak_max > peak_min:
            vertices.append([mean_min, peak_min])
            if debug:
                print(f"\t       Adding point E: {vertices[-1]}")

        # Point F: [mean: max(mean) | peak: min(peak)]
        # Is only added, if:
        #       ( min(peak) > max(mean) )
        # AND   ( max(peak) > min(peak) )
        # AND   ( max(mean) > min(mean) )
        if (peak_min > mean_max) and (peak_max > peak_min) and (mean_max > mean_min):
            vertices.append([mean_max, peak_min])
            if debug:
                print(f"\t       Adding point F: {vertices[-1]}")

        key_names = [self._get_key_name_ts(ts_name, 'mean'), self._get_key_name_ts(ts_name, 'peak')]

        return (key_names, vertices)

    def _get_vertices_ts_mean_and_relpeak(self, ts_name:str, debug:bool) -> tuple:
        """ Returns all vertices of a given timeseries.
        The indened use is for scale_mode == 'mean_and_relpeak'

        Args:
            ts_name (str): name of the timeseries in ts_scaling.yaml
            debug (bool): If True, the added vertices are printed.

        Returns:
            (list, list): (nested list of column_names, nested list of all vertex combination)
        """

        ts_dict = self.ts_scaling[ts_name]
        mean_min = ts_dict['param']['mean']['min']
        mean_max = ts_dict['param']['mean']['max']
        relpeak_min = ts_dict['param']['relpeak']['min']
        relpeak_max = ts_dict['param']['relpeak']['max']

        # Initialize empty list with point A
        # Point A: [mean: min(mean) | relpeak: min(relpeak)]
        vertices = [ [mean_min, relpeak_min] ]
        if debug:
            print(f"\t   Getting mean_and_relpeak vertices for '{ts_name}':")
            print(f"\t       Adding point A: {vertices[-1]}")

        # Point B: [mean: max(mean) | relpeak: min(relpeak)]
        # Is only added, if max(mean) > min(mean)
        if mean_max > mean_min:
            vertices.append([mean_max, relpeak_min])
            if debug:
                print(f"\t       Adding point B: {vertices[-1]}")

        # Point C: [mean: max(mean) | relpeak: max(relpeak)]
        # Is only added, if  max(mean) > min(mean) and max(relpeak) > min(relpeak)
        if mean_max > mean_min and relpeak_max > relpeak_min:
            vertices.append([mean_max, relpeak_max])
            if debug:
                print(f"\t       Adding point C: {vertices[-1]}")

        # Point D: [mean: min(mean) | relpeak: max(relpeak)]
        # Is only added, if max(relpeak) > min(relpeak)
        if relpeak_max > relpeak_min:
            vertices.append([mean_min, relpeak_max])
            if debug:
                print(f"\t       Adding point D: {vertices[-1]}")

        key_names = [
            self._get_key_name_ts(ts_name, 'mean'), self._get_key_name_ts(ts_name, 'relpeak')]

        return (key_names, vertices)

    def _get_vertices_ts_interpolate(self, ts_name:str, debug:bool) -> tuple:
        """ Returns all vertices of a given timeseries.
        The indened use is for scale_mode == 'interpolate'

        Args:
            ts_name (str): name of the timeseries in ts_scaling.yaml
            debug (bool): If True, the added vertices are printed.

        Returns:
            (list, list): (nested list of column_names, nested list of all vertex combination)
        """
        ts_dict = self.ts_scaling[ts_name]
        interpolate_min = ts_dict['param']['interpolate']['min']
        interpolate_max = ts_dict['param']['interpolate']['max']

        key_names = [self._get_key_name_ts(ts_name, 'interpolate')]

        # If max > min, return [min, max], else [max]
        if interpolate_min < interpolate_max:
            # Print infos in debug-mode
            if debug:
                print(f"\t   Getting 'interpolate' vertices for '{ts_name}':")
                print("\t       Adding min and max")
            return (key_names, [[interpolate_min], [interpolate_max]])

        if interpolate_min == interpolate_max:
            # Print infos in debug-mode
            if debug:
                print(f"\t   Getting mean vertices for '{ts_name}':")
                print("\t       Adding min")
            return (key_names, [[interpolate_min]])

        raise ValueError(
            f"min(interpolate) > max(interpolate) for '{ts_name}'"
        )

    def _generate_timeseries_splitted(self) -> dict:
        """ Generates a dict containing all parameters to be sampled.
        Splits 'peak' and 'mean' into different parameters

        Returns:
            dict: containing sample parameters as
                {key : {'min': _, 'max': _, 'ts_scaling':_}}
        """

        # Initialize empty dict:
        param_dict = {}

        if not self.sample_timeseries:
            return param_dict

        # Loop over list of sampled ts_names:
        for ts_name in self._generate_timeseries_sampled():
            ts_dict = self.ts_scaling[ts_name]

            for key, val in ts_dict['param'].items():
                key_name = self._get_key_name_ts(ts_name, key)
                param_dict[key_name] = {
                    'min': val['min'],
                    'max': val['max'],
                    'ts_scaling': ts_name
                }
                # Add key_name to ts_scaling.ts_dict.param.val:
                val['key_name'] = key_name

        return param_dict

    def _load_ts_csv(self, ts_path:str, category: str, process_or_product:str) -> pd.DataFrame:
        """ Loads the csv file for the given ts category and process/product

        Args:
            ts_path (str): path where the .csv file lies
            category (str): category of the ts (e.g., 'demand' or 'availability')
            process_or_product (str): the process (e.g., 'PV') or product (e.g., 'heat') 

        Returns:
            pd.DataFrame: data loaded from the .csv file.
        """
        keys_for_category = [
            key for key, val in self.timeseries.items()
            if category in val['categories']
        ]
        correct_ts_key = [
            key for key in keys_for_category
            if (
                self.timeseries[key].get('process', self.timeseries[key].get('product'))
                == process_or_product
                )
        ]

        if not correct_ts_key:
            raise ValueError(f"Found no {category} for {process_or_product}.")
        if len(correct_ts_key)>1:
            raise ValueError(
                f"Found {len(correct_ts_key)} matching entries for {category} for " +
                f"{process_or_product} (Should only be one).")
        correct_ts_key = correct_ts_key[0]

        return pd.read_csv(os.path.join(
                ts_path, self.timeseries[correct_ts_key]['file']
            ))['data']
