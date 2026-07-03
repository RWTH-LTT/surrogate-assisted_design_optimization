""" This module contains functions and classes to manipulate pyomo models.
    The main functionalities are:
    - generate_initial_model(): Generate a blank pyomo model with zero nominal capacities for all
        processes in the superstructure.
    - mutate_nominal_capacities(): Mutate the nominal capacities of a pyomo model according to a
        design_dict.
    - EnergySystemDesignConverter: Class to convert a pandas.Series representing an energy system
        design into a design_dict.
    - TimeSeriesScaler: Class to scale timeseries data in a pyomo model according to a pandas.Series
"""

import os
import warnings
from copy import deepcopy
import yaml
import pandas as pd
import numpy as np
from core.helpers import (UniqueKeyLoader, get_minimal_capacity, get_potential_capacity,
                         get_config_file)
from core.preprocessing import UnscaledInputDictGenerator, ScaledInputDictGenerator
from core.optimization import Optimization_LP, Optimization_MILP
from core.ts_scaling import get_scale_mode_ts, get_products_or_processes_scaling_ts

class EnergySystemDesignConverter():
    """ This class converts a pandas.DataFrame parameterizing an energy system design into 
    the energy system design stored as dict.
    """
    def __init__(self, data_in_path, project_file:str):
        """

        Args:
            data_in_path : Path to the input data
            project_file (str): name of the project file (e.g., 'project.yaml')
        """
        self.data_in_path = data_in_path
        self._load_general_input_yamls(project_file)
        self._generate_superstructure_production_and_storage()

    def _load_general_input_yamls(self, project_file:str,):
        """ Loads the general input yamls from the data_in_path and stores them
        in the class attributes.
        Args:
            project_file (str): name of the .yaml file in data_in_path containing the
                input information (e.g. project.yaml)
        """

        # Load project file:
        with open(os.path.join(self.data_in_path, project_file), 'r', encoding='utf-8') as file:
            self.project = yaml.load(file, Loader=UniqueKeyLoader)
        # Load config:
        with open(
            os.path.join(self.data_in_path,  self.project['config']), 'r', encoding='utf-8'
            ) as file:
            self.config = yaml.load(file, Loader=UniqueKeyLoader)
        # Load technologies:
        with open(
            os.path.join(self.data_in_path,  self.project['technologies']), 'r', encoding='utf-8'
            ) as file:
            self.technologies = yaml.load(file, Loader=UniqueKeyLoader)['technologies']
        # Load superstructure:
        with open(
            os.path.join(self.data_in_path,  self.project['superstructures']), 'r', encoding='utf-8'
            ) as file:
            self.superstructure = yaml.load(
                file, Loader=UniqueKeyLoader
                )['superstructures'][self.config['superstructure_name']]

    def _generate_superstructure_production_and_storage(self):
        """ Loops over all components in self.design and creates list of production_processes"""
        self.superstructure_production = {}
        self.superstructure_storage = {}

        for component_name, property_dict in self.superstructure.items():
            if self.technologies[component_name]['category'] == 'production':
                self.superstructure_production[component_name] = property_dict
            elif self.technologies[component_name]['category'] == 'storage':
                self.superstructure_storage[component_name] = property_dict
            else:
                raise ValueError(
                    f"category '{property_dict['category']}' is invaild. "+
                    "Select 'production' or 'storage'.")

    def get_design_dict_from_series(self, design_series: pd.Series):
        """ Converts an energy system design given as a pd.Series to a dict.

        Args:
            design_series (pd.Series): pandas.Series representing the design
        """
        # Check, that design_series is a pd.Series:
        assert isinstance(design_series, pd.Series), "'design_series must be a pandas.Series."

        # Remove column 'id' if it exists
        if 'id' in design_series.index:
            design_series = design_series.drop('id')

        # Check, that all indices are part of the superstructure:
        assert all(process in self.superstructure for process in design_series.index), (
            "Invalid process name was used in 'design_series'. "+
            "Make sure, all indices in 'design_series' occur in the superstructure."
            )

        # Check, that all values are non-negative: (with a tolerance of 1e-10)
        if any(design_series.values < -1e-10):
            min_value = design_series.min()
            process = design_series.index[design_series.argmin()]
            raise ValueError(
            "All Design values must be non-negative, but the process "+
            f"'{process}' has a value of '{min_value}'.")

        processes = {
            **{p: ('nominal_capacity_kW', self.superstructure[p])
               for p in self.superstructure_production},
            **{p: ('nominal_capacity_kWh', self.superstructure[p])
               for p in self.superstructure_storage}
        }

        return {
            process: {**details, capacity_key: float(design_series.get(process, 0))}
            for process, (capacity_key, details) in processes.items()
            if design_series.get(process, 0) > 0
        }

    def export_series_as_yaml(self, design_series: pd.Series, export_path):
        """ Exports a pandas.Series representing an energy system design to a yaml file.        

        Args:
            design_series (pd.Series): pandas.Series representing the design
            export_path: Path to the file, where the design should be exported (w/o .yaml ending)
        """
        design_dict = self.get_design_dict_from_series(design_series)
        export_path = f"{export_path}.yaml"
        print(f"Exported design to {export_path}.")

        with open(export_path,'w', encoding='utf-8') as f:
            yaml.dump(design_dict, f, default_flow_style=False, sort_keys=False)

    def get_blank_design_dict(self):
        """ Generates a blank design_dict. The intended use of the design_dict is in generation of
        an initial pyomo model in generate_initial_model(). 

        Returns:
            dict: blank design_dict, containing all production and storage processes from the
                superstructure but each with a nominal_capacity of 0.
        """
        production_dict = {
            process: process_dict | {'nominal_capacity_kW' : 0} 
            for process, process_dict in self.superstructure_production.items()
        }
        storage_dict = {
            process: process_dict | {'nominal_capacity_kWh' : 0} 
            for process, process_dict in self.superstructure_storage.items()
        }

        return production_dict | storage_dict

    def get_bounds(self, return_type:str='list', minimal_capacity_is_zero:bool=True):
        """ Returns the bounds of all technologies of the superstructure.

        Args:
            return_type (str, optional): Sets the return type (either 'list' or 'dict').
                Defaults to 'list'.
            minimal_capacity_is_zero (bool, optional): If True, the minimal capacity is set to zero.
                Defaults to True.

        Raises:
            ValueError: Raised if `return_type` is not 'list' or 'dict'.

        Returns:
            dict or list: Returns the bounds of the uncertain parameters.
        """
        # Initialize bounds dictionary
        bounds = {}

        # Loop over all processes in the superstructure
        for process, process_dict in self.superstructure.items():

            # Set minimal capacity to 0 if minimal_capacity_is_zero is True,
            # else use the value from the technology dictionary
            min_capacity = 0 if minimal_capacity_is_zero else get_minimal_capacity(
                self.technologies[process_dict['type']]
                )
            pot_capacity = get_potential_capacity(self.technologies[process_dict['type']])

            # Store the bounds in the dictionary
            bounds[process] = [min_capacity, pot_capacity]

        # Return bounds in the requested format
        if return_type == 'dict':
            return bounds
        elif return_type=='list':
            return list(bounds.values())

        # Raise an error for invalid return_type
        raise ValueError(f"Unknown return type '{return_type}'. Use 'list' or 'dict'.")

class TimeSeriesScaler():
    """
    This class is intended to be used to scale timeseries data up and down.
        The scaling is by ts_series, which is created from timeseries_scaling.yaml using the 
        SampleGenerator(). The scaling takes place AFTER a blank pyomo model is initiliazed.
        Thus, this scaling offers an alternative way of scaling inputs than using
        UnscaledInputDictGenerator. TimeSeriesScaler is intended to be used on the HPC to allow for
        efficient input parameter manipulation.
    """
    def __init__(
            self, data_in_path:str, project_file:str, blank_model, timeseries_ignored:list=None):
        """
        Args:
            data_in_path (str): path to the input data (containing the project file)
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
            blank_model: unsolved blank pyomo model generated
                by sm.model_manipulation.generate_initial_model
            timeseries_ignored (list): list of strings defining which timeseries are not scaled.
                Defaults to [].
        """

        if timeseries_ignored is None:
            timeseries_ignored = []

        self.data_in_path       = data_in_path
        self.project_file       = project_file
        self.blank_model        = deepcopy(blank_model)
        self.timeseries_ignored = timeseries_ignored
        self.max_overcapacity   = hasattr(blank_model, 'demand_scaled')

        self._load_general_input_yamls(project_file)


        # Build helper dicts:
        self.ts_demand_helper   = self._get_helper_dict_demand()
        self.ts_pricebuy_helper = self._get_helper_dict_pricebuy()

        # Sanity Check:
        self._sanity_check()

    def scale_ts(self, model, ts_series:pd.Series):
        """ Scales the timeseries in the given pyomo model according to the given ts_series.

        Args:
            model (ConcreteModel): Initialized pyomo model
            ts_series (pd.Series): Contains the scaling parameters for the current sample

        Returns:
            ConcreteModel: Mutated pyomo model with scaled timeseries
        """
        # Scale demand timeseries:
        model = self._scale_demand_timeseries(model, ts_series)

        # Scale price buy timeseries:
        model = self._scale_pricebuy_timeseries(model, ts_series)

        # Scale availability time series:
        model = self._interpolate_availability_timeseries(model, ts_series)

        return model

    def _load_general_input_yamls(self, project_file:str,):
        """ Loads the general input yamls from the data_in_path and stores them
        in the class attributes.

        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
        """

        # Load project file:
        with open(os.path.join(self.data_in_path, project_file), 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)
        # Load config:
        with open(os.path.join(self.data_in_path,  self.project['config']),
                  'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=UniqueKeyLoader)
        # Load ts_scaling:
        with open(os.path.join(self.data_in_path,  self.project['timeseries_scaling']),
                  'r', encoding='utf-8') as f:
            self.ts_scaling = yaml.load(
                f, Loader=UniqueKeyLoader
                )['timeseries_scaling'][self.config['timeseries_scaling']]
        # Load timeseries data:
        with open(os.path.join(self.data_in_path,  self.project['timeseries']),
                  'r', encoding='utf-8') as f:
            self.timeseries = yaml.load(
                f, Loader=UniqueKeyLoader
                )
        # Load products data:
        with open(os.path.join(self.data_in_path,  self.project['products']),
                  'r', encoding='utf-8') as f:
            self.products = yaml.load(
                f, Loader=UniqueKeyLoader
                )

    def _get_helper_dict_demand(self):
        """ Returns a helper dict for the demand timeseries. The dict contains
        the 'scale_mode', 'mean_old' and 'peak_old' for each demand timeseries.

        Returns:
            dict: Helper dict for demand timeseries
        """
        # Get all demand timeseries names:
        ts_names = [
            key
            for key, val in self.ts_scaling.items()
            if val['ts_type'] == 'demand'
        ]

        # Initiate empty dict:
        helper_dict = {}

        for ts_name in ts_names:

            # Skip, if ts_name should be ignored:
            if ts_name in self.timeseries_ignored:
                print(f"Ignoring {ts_name}.")
                continue

            # Get the product name:
            product = self.ts_scaling[ts_name]['product']

            # Add the helper dict:
            helper_dict[product] = {
                'scale_mode': get_scale_mode_ts(self.ts_scaling[ts_name]),
                'mean_old'  : self.blank_model.ts_demand_old_mean[product],
                'peak_old'  : self.blank_model.ts_demand_old_peak[product]
            }

        return helper_dict

    def _get_helper_dict_pricebuy(self):
        """ Returns a helper dict for the pricebuy timeseries. The dict contains
        the 'scale_mode', 'mean_old' and 'peak_old' for each pricebuy timeseries.

        Returns:
           dict: Helper dict for pricebuy timeseries 
        """
        # Get all pricebuy timeseries names:
        ts_names = [
            key
            for key, val in self.ts_scaling.items()
            if val['ts_type'] == 'pricebuy'
        ]

        # Initiate empty dict:
        helper_dict = {}

        for ts_name in ts_names:

            # Skip, if ts_name should be ignored:
            if ts_name in self.timeseries_ignored:
                print(f"Ignoring {ts_name}.")
                continue

            # Get the product name:
            product = self.ts_scaling[ts_name]['product']

            # Add the helper dict:
            helper_dict[product] = {
                'scale_mode': get_scale_mode_ts(self.ts_scaling[ts_name]),
                'mean_old'  : self.blank_model.ts_pricebuy_old_mean[product],
                'peak_old'  : self.blank_model.ts_pricebuy_old_peak[product]
            }

        return helper_dict

    def _scale_demand_timeseries(self, model, ts_series:pd.Series):
        """ 
        Args:
            model (pyomo.environ.ConcreteModel): Initialized pyomo model
            ts_series (pd.Series): Contains the scaling parameters for the current sample
        """
        # If the optimization type is 'max_overcapacity', the demand cannot be scaled inside
        # optimization.py, since " model.Demand * (1 + model.Overcapacity) " in the energy balance
        # constaint would result in a quadratic constraint. To avoid this, every element (product,
        # timestep, and scenario) is scaled with this class (TimeSeriesScaler)
        if self.max_overcapacity:
            return self._scale_demand_timeseries_max_overcapacity(model, ts_series)

        # If the optimization type is NOT 'max_overcapacity', the demand scaling takes place INSIDE
        # optimization.py. This is not mandatory for standard optimizations (operational and design
        # optimizations). For the exploraiton of the feasibility hull, however, this is mandatory,
        # as model.Demand becomes a decision variable and cannot be treated as a paramter anymore.
        return self._scale_demand_timeseries_in_optimization(model, ts_series)

    def _scale_demand_timeseries_in_optimization(self, model, ts_series:pd.Series):
        """ scales the timeseries for energy demand with y_new = a * y_old + b.
        If only the 'mean' shoud be scaled, 'a' = mean and 'b' = 0.
        If the 'mean' and 'peak' should be scaled independtly, calculate 'a' and 'b' to satisfy:
            peak_new = a * peak_old + b
            mean_new = a * mean_old + b
        Note: The scaling with a and b takes place in optimization.py. In this function,
        peak_new and mean_new are provided
        Args:
            model (pyomo.environ.ConcreteModel): Initialized pyomo model
            ts_series (pd.Series): Contains the scaling parameters for the current sample

        Raises:
            ValueError: If an invalid 'scale_mode' is provided
        """

        for product, ts_dict in self.ts_demand_helper.items():
            if ts_dict['scale_mode'] == 'mean':
                # a = mean_new / mean_old
                # b = 0

                mean_new = ts_series[f'demand_{product}_mean']

                # Fix Variable: Ts_demand_new_mean:
                model.Ts_demand_new_mean[product].fix(mean_new)

                # Ts_demand_new_peak must be scaled aswell:
                a = mean_new / ts_dict['mean_old']
                peak_old = model.ts_demand_old_peak[product]
                peak_new = a * peak_old

                # Fix Variable: Ts_demand_new_mean:
                model.Ts_demand_new_peak[product].fix(peak_new)

            elif ts_dict['scale_mode'] == 'mean_and_peak':
                # a = (peak_new - mean_new) \ (peak_old - mean_old)
                # b = mean_new - a * mean_old

                # Fix Variable: Ts_demand_new_mean:
                model.Ts_demand_new_mean[product].fix(ts_series[f'demand_{product}_mean'])
                # Fix Variable: Ts_demand_new_peak:
                model.Ts_demand_new_peak[product].fix(ts_series[f'demand_{product}_peak'])

            else:
                raise ValueError(f"Unknown 'scale_mode': {ts_dict['scale_mode']}.")

        return model

    def _scale_demand_timeseries_max_overcapacity(self, model, ts_series:pd.Series):
        """ scales the timeseries for energy demand with y_new = a * y_old + b.
        If only the 'mean' shoud be scaled, 'a' = mean and 'b' = 0.
        If the 'mean' and 'peak' should be scaled independtly, calculate 'a' and 'b' to satisfy:
            peak_new = a * peak_old + b
            mean_new = a * mean_old + b
        Args:
            model (pyomo.environ.ConcreteModel): Initialized pyomo model
            ts_series (pd.Series): Contains the scaling parameters for the current sample

        Raises:
            ValueError: If an invalid 'scale_mode' is provided
        """
        for product, ts_dict in self.ts_demand_helper.items():
            if ts_dict['scale_mode'] == 'mean':
                # a = mean_new / mean_old
                a = ts_series[f'demand_{product}_mean'] / ts_dict['mean_old']
                b = 0

            elif ts_dict['scale_mode'] == 'mean_and_peak':
                # warn, if mean > peak:
                if ts_series[f'demand_{product}_peak'] < ts_series[f'demand_{product}_mean']:
                    warnings.warn(
                        f"peak < mean ({ts_series[f'demand_{product}_peak']} "+
                        f"< {ts_series[f'demand_{product}_mean']}) for demand of '{product}')"
                    )

                # a = (peak_new - mean_new) \ (peak_old - mean_old)
                a = (ts_series[f'demand_{product}_peak'] - ts_series[f'demand_{product}_mean']) \
                    / (ts_dict['peak_old'] - ts_dict['mean_old'])

                # b = mean_new - a * mean_old
                b = ts_series[f'demand_{product}_mean'] - a * ts_dict['mean_old']

            else:
                raise ValueError(f"Unknown 'scale_mode': {ts_dict['scale_mode']}.")

            for t in model.t:
                for scenario in model.scenarios:
                    # Access the parameter value and scale it
                    model.demand_scaled[product, t, scenario] = (
                        a * model.demand_unscaled[product, t, scenario] + b
                        )

        return model

    def _scale_pricebuy_timeseries(self, model, ts_series:pd.Series):
        """ scales the timeseries for energy prices with y_new = a * y_old + b
        If only the 'mean' shoud be scaled, 'a' = mean and 'b' = 0.
        If the 'mean' and 'peak' should be scaled independtly, calcalate 'a' and 'b' to satisfy:
            peak_new = a * peak_old + b
            mean_new = a * mean_old + b

        Args:
            model (pyomo.environ.ConcreteModel): Initialized pyomo model
            ts_series (pd.Series): Contains the scaling parameters for the current sample

        Raises:
            ValueError: If an invalid 'scale_mode' is provided
        """

        for product, ts_dict in self.ts_pricebuy_helper.items():
            if ts_dict['scale_mode'] == 'mean':
                # a = mean_new / mean_old
                a = ts_series[f'pricebuy_{product}_mean'] / ts_dict['mean_old']
                b = 0

            elif ts_dict['scale_mode'] == 'mean_and_peak':
                # warn, if mean > peak:
                if ts_series[f'pricebuy_{product}_peak'] < ts_series[f'pricebuy_{product}_mean']:
                    warnings.warn(
                        f"peak < mean ({ts_series[f'pricebuy_{product}_peak']} "+
                        f"< {ts_series[f'pricebuy_{product}_mean']}) for pricebuy of '{product}')"
                    )

                # a = (peak_new - mean_new) \ (peak_old - mean_old)
                a = (ts_series[f'pricebuy_{product}_peak'] - ts_series[f'pricebuy_{product}_mean'])\
                    / (ts_dict['peak_old'] - ts_dict['mean_old'])

                # b = mean_new - a * mean_old
                b = ts_series[f'pricebuy_{product}_mean'] - a * ts_dict['mean_old']

            elif ts_dict['scale_mode'] == 'mean_and_relpeak':

                mean_new = ts_series[f'pricebuy_{product}_mean']
                peak_new = mean_new * ts_series[f'pricebuy_{product}_relpeak']
                # a = (peak_new - mean_new) \ (peak_old - mean_old)
                a = (peak_new - mean_new) / (ts_dict['peak_old'] - ts_dict['mean_old'])

                # b = mean_new - a * mean_old
                b = mean_new - a * ts_dict['mean_old']

            else:
                raise ValueError(f"Unknown 'scale_mode': {ts_dict['scale_mode']}.")

            for t in model.t:
                for scenario in model.scenarios:
                    # Access the parameter value and scale it
                    model.pricebuy_scaled[product, t, scenario] = (
                        a * model.pricebuy_unscaled[product, t, scenario] + b
                        )

            # Scale pricesell if it is coubpled to pricebuy:
            if self.products[product].get('price_sell_follows_price_buy', False):
                price_sell_buy_ratio = self.products[product].get('price_sell_buy_ratio', 0)

                for t in model.t:
                    for scenario in model.scenarios:
                        price_buy = model.pricebuy_scaled[product, t, scenario]
                        # Access the parameter value and scale it
                        model.pricesell_scaled[product, t, scenario] = (
                            price_buy - (1 - price_sell_buy_ratio) * np.abs(price_buy)
                            )

        return model

    def _interpolate_availability_timeseries(self, model, ts_series:pd.Series):
        """ interpolates the timeseries for availability between a first timeseries y_1st and a
        second timeseries y_2nd:
        y_new(t) = y_1st(t) * (1 - scaler) + y_2nd(t) * scaler

        Args:
            model (pyomo.environ.ConcreteModel): Initialized pyomo model
            ts_series (pd.Series): Contains the scaling parameters for the current sample

        Raises:
            ValueError: If an invalid 'scale_mode' is provided
        """

        processes, ts_key_names = get_products_or_processes_scaling_ts(
            ts_scaling         = self.ts_scaling,
            ts_type            = 'availability',
            product_or_process = 'process'
        )
        for process, ts_key in zip(processes, ts_key_names):

            # Skip, if ts_name should be ignored:
            if ts_key in self.timeseries_ignored:
                print(f"Ignoring {ts_key}.")
                continue

            scale_mode = get_scale_mode_ts(self.ts_scaling[ts_key])
            if scale_mode != 'interpolate':
                raise ValueError(
                    f"Only 'scale_mode'='interpolate' is allowed. Instead got: '{scale_mode}'.")

            scaler = ts_series[f"availability_{process}_interpolate"]

            # Mutate the scaler:
            for scenario in model.scenarios:
                model.availability_interpolation_scaler[process, scenario] = scaler

                # Interpolate the availability:
                for t in  model.t.data():
                    model.production_availability[process, t, scenario] = (
                        model.production_availability_interpolate_1st[process, t] * (1 - scaler)
                        + model.production_availability_interpolate_2nd[process, t] * scaler
                    )

        return model

    def _get_number_of_timesteps(self):

        number_of_timesteps = sum(
            list(self.blank_model.time_slice_weight.values())
        )
        if number_of_timesteps != 8760:
            raise NotImplementedError(
                "TimeSeriesScaler is currently implemented to work only with 8760 hours. "+
                f"Instead got '{number_of_timesteps}'."
            )
        return number_of_timesteps

    def _sanity_check(self):

        # Check, that all ts_types are valid:
        if any(
            val['ts_type'] not in ['demand', 'pricebuy', 'availability']
            for val in self.ts_scaling.values()
        ):
            raise ValueError(
                "Invalid 'ts_type' is used in ts_scaling.yaml. +"
                "Only ['demand', 'pricebuy', 'availability'] are allowed."
            )

        # Check, that all keys in self.ignored_timeseries are also in self.ts_scaling:
        if any(ts_name not in self.ts_scaling for ts_name in self.timeseries_ignored):
            raise ValueError(
                "At least one key in 'timeseries_ignored' is not in 'ts_scaling'."
            )

def generate_initial_model(data_in_path:str, project_file:str, objective:str='operation'):
    """ Generates a pyomo model with blank nominal capacities for the given data and project file.

    Args:
        data_in_path (str): Path to the input data
        project_file (str): name of the project file (e.g., 'project.yaml')
        objective (str): 'operation' or 'design'. Defaults to 'operation'

    Returns:
        (pyomo model, core.optimization.Optimization):
        pyomo.core.base.PyomoModel.ConcreteModel: model with blank nominal capacities.
            Intended use is that the nominal capacities are mutated with mutate_nominal_capacity(),
            before the model is solved.
        core.Optimization.Optimization_LP or Optimization_MILP: Optimization object.
    """

    # For operational optimization, generate blank_design_dict:
    if objective in {'operation', 'max_overcapacity'}:
        # Generate blank design_dict:
        blank_design_dict = EnergySystemDesignConverter(
            data_in_path = data_in_path,
            project_file = project_file
        ).get_blank_design_dict()
    # For design optimization, set blank_design_dict to None
    # (so it is not used by UnscaledInputDictGenerator())
    elif objective in {'design', 'near_optimal'}:
        blank_design_dict = None
    else:
        raise ValueError(f"Unexpected objective: '{objective}'.")

    # Do an optimization preprocessing:
    unscaled_input_dict = UnscaledInputDictGenerator(
        data_in_path  = data_in_path,
        project_file  = project_file,
        objective     = objective,
        custom_design = blank_design_dict
    ).generate_unscaled_input_dict()

    scaled_input_dict = ScaledInputDictGenerator(
        data_in_path        = data_in_path,
        project_file        = project_file,
        unscaled_input_dict = unscaled_input_dict,
        scenario_df         = None
    ).generate_scaled_input_dict()

    # Use either LP or MILP, depending on config.yaml
    optimizations = {'LP': Optimization_LP, 'MILP': Optimization_MILP}[get_config_file(
        data_in_path, project_file)['model_formulation']]

    optimization = optimizations(
        data_in_path = data_in_path,
        input_dict   = scaled_input_dict,
        project_file = project_file
    )

    # Build and return the pyomo model
    return optimization, optimization.generate_model()

def mutate_nominal_capacities(model, design_dict):
    """ Mutate the nominal capacities of the model according to the design_dict.
    Intended use is before an operational optimization NOT a design optimization.

    Args:
        model (pyomo.core.base.PyomoModel.ConcreteModel): Unmutated model
        design_dict (dict): Dictionary containing the new nominal capacities for each process.
            The keys are the process names and the values are dictionaries containing the new
            nominal capacities for each process.
            The dictionary must contain the keys 'nominal_capacity_kW' or 'nominal_capacity_kWh'
            and 'type' for each process.
    Returns:
        pyomo.core.base.PyomoModel.ConcreteModel: Mutated model
    """
    # Check that design_dict is a dict:
    assert isinstance(design_dict, dict), (
        f"design_dict must be a dict, is '{type(design_dict)}' instead.")
    # Check if the design_dict is valid
    assert all(
        set(process_dict.keys())
        in ({'nominal_capacity_kW', 'type'},{'nominal_capacity_kWh', 'type'})
        for process_dict in design_dict.values()), (
            "The design_dict is not valid. mutate_nominal_capacities it only intended for the " + 
            "keys 'nominal_capacity_kW' or 'nominal_capacity_kWh' and 'type' for each process.")

    process_list =list(model.production_potential_capacity) + list(model.storage_potential_capacity)
    assert all(key in process_list for key in design_dict.keys()), (
        'The design_dict contains processes that are not in the model')

    # Check that model is an operational optimization: if
    # constraint_production_operational_optimization_max_rule is not set, its a design optimization
    if len(list(model.constraint_production_operational_optimization_max_rule.keys())) == 0:
        raise ValueError(
            "'mutate_nominal_capacities' is intended to be used for operational optimization. "+
            "Got model for design optimization instead.")

    # Mutate nominal_capacity of production:
    for index in model.production_potential_capacity:
        # Mutate potential_capacity
        model.production_potential_capacity[index] = design_dict.get(index, {}).get(
            'nominal_capacity_kW', 0)
        # Mutate minimal_capacity, if existing:
        if hasattr(model, 'production_minimal_capacity'):
            model.production_minimal_capacity[index] = design_dict.get(index, {}).get(
                'nominal_capacity_kW', 0)

    # Mutate nominal capacity of storage:
    for index in model.storage_potential_capacity:
        # Mutate potential_capacity
        model.storage_potential_capacity[index] = design_dict.get(index, {}).get(
            'nominal_capacity_kWh', 0)
        # Mutate minimal_capacity, if existing:
        if hasattr(model, 'storage_minimal_capacity'):
            model.storage_minimal_capacity[index] = design_dict.get(index, {}).get(
                'nominal_capacity_kWh', 0)

    return model
