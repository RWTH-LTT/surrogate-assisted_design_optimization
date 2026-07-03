""" This module contains the classes and functions for preprocessing the input data."""
import os
import pickle
from copy import deepcopy
import itertools
import warnings
import pandas as pd
import numpy as np
import yaml
import tsam.timeseriesaggregation as tsam
import bokeh.plotting as bplt
from bokeh.palettes import Category10_10, Category20  # Importing a color palette
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.layouts import column
from core.helpers import UniqueKeyLoader, get_tsa_path, calculate_weighted_mean, get_data_out
from core.ts_scaling import (
    sanity_check_ts, get_scale_mode_ts, get_products_or_processes_scaling_ts,
    sanity_check_ts_scaling_bounds, sanity_check_ts_scaling_nonnegative,
    add_ts_scaling_old_values)


class UnscaledInputDictGenerator():
    """ Class to generate the unscaled input dictionary for the optimization model.
    Intended to be used before the AggregatedInputDictGenerator and before ScaledInputDictGenerator.
    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)
        objective (str): 
            'design' for a superstructure-based design optimization
            'operation' for an operational optimization with given designs
            'max_overcapacity' for an overcapacity maximization
            'feasible_hull' for determination of the feasible space
            'near_optimal' for determining the near-optimal space
        name_design (str, optional): name of the chosen design, must only be given for operational
            optimization
        custom_design_path (str, optional): Path to the yaml file containing the design.
            If None, the design is loaded from the input yamls. Defaults to None.
        custom_design (dict, optional): dict containing the design. 
        ignore_tsa (bool, optional): Forces this class to load the full timeseries. If True, the
            parameter self.config['load_tsa_inputs'] is overwritten to False. Defaults to False.
    """
    def __init__(self, data_in_path, project_file:str, objective:str, name_design:str=None,
                 custom_design_path:str=None, custom_design:dict=None, ignore_tsa:bool=False):

        print("Unscaled_InputDict_Generator:")
        assert objective in {'design', 'operation', 'max_overcapacity', 'feasible_hull', 'near_optimal'}, (
            f"Unkown objective: '{objective}'. objective must be in " +
            "['design', 'operation', 'max_overcapacity', 'feasible_hull', 'near_optimal']")
        self.objective = objective

        self.data_in_path = data_in_path
        assert os.path.exists(self.data_in_path)
        self._load_general_input_yamls(project_file, ignore_tsa)

        # Initialize input dict:
        self.input_dict = {}

        # Initialize attributes:
        self.product_list = None
        self.processes_production = None
        self.processes_storage = None
        self.design = None

        self._load_design(name_design, custom_design_path, custom_design)
        # Generate designs, processes_production and processes_storage, and product_list:
        self._generate_design()
        self._generate_processes_production_and_storage_lists()
        self._generate_product_list()

        # Check that the model is either an LP or a MILP:
        if self.config['model_formulation'] not in ('LP', 'MILP'):
            raise ValueError("'model_formulation' must be either 'LP' or 'MILP'.")

        # In the case of LP Formulation, do LP sanity check on the input data:
        if self.config['model_formulation'] == 'LP':
            self._LP_sanity_check()

        # If objective is operation, fix minimal and potential capacities:
        if self.objective in {'operation', 'max_overcapacity'}:
            self._fix_minimal_and_potential_capacities()

    def generate_unscaled_input_dict(self, data_export_path:str=None):
        """Main function of UnscaledInputDictGenerator()

        Args:
            data_export_path (str, optional): path where the unscaled input dict
                should be saved. Defaults to None.

        Returns:
            dict: unscaled_input_dict
        """
        print(f"\tGenerating unscaled input_dict for '{self.objective}' optimization.")

        # Add miscellaneous parameters:
        self._add_misc_to_input_dict()

        self._add_importable_exportable_products_to_input_dict()
        self._add_timeslices_to_input_dict()

        if self.config['time_series_aggregation']['load_tsa_inputs']:
            self._add_exported_tsa_parameters_to_input_dict()
        else:
            self._add_dummy_tsa_parameters_to_input_dict()

        self._add_availability_timeseries_to_input_dict()
        self._add_product_timeseries_to_input_dict('demand', 'demand')
        self._add_product_timeseries_to_input_dict('pricebuy', 'pricebuy')
        self._add_product_timeseries_to_input_dict('pricesell', 'pricesell')
        self._add_temperature_timeseries_to_input_dict()

        self._couple_timeseries_price_sell_and_buy()

        # Add timeseries scaling:
        self._add_timeseries_scaling_demand()
        self._add_timeseries_scaling_pricebuy()
        self._add_timeseries_scaling_availability()

        self._add_parameters_for_product_slack_term_to_input_dict()
        self._add_components_to_input_dict()

        # Add sets and parameters for feasible_hull exploration:
        if self.objective == 'feasible_hull':

            self._add_feasibility_affecting_sets()
            self._add_feasibility_extreme_values()
            self._add_default_feasibility_search_directions()

        # Add default values for slack, TAC, process boundaries, and search direction
        if self.objective == 'near_optimal':
            self._add_default_slack()
            self._add_default_optimal_tac()
            self._add_processes_extreme_values()
            self._add_default_near_optimal_search_directions()

        if data_export_path is not None:
            print(f"\tExporting unscaled input_dict to {data_export_path}.pkl")
            with open(f"{data_export_path}.pkl", 'wb') as outfile:
                pickle.dump(self.input_dict, outfile)

        print("\tDone: Generated unscaled input_dict.")
        return self.input_dict

    def _load_design(self, name_design:str, custom_design_path, custom_design):
        """ Loads the design. If custom_design_path is None, the design path is taken from 
        the project file. If a custom_design_path is given the custom_design_path is used. 

        Args:
            name_design (str): name of the design, must only be given for operational operation
            custom_design_path (str): custom path to design input which overwrites path from
                the project file
        """
        if self.objective in {'design', 'feasible_hull', 'near_optimal'}:
            assert custom_design_path is None, (
                "'custom_design_path' is not None. " +
                "The design optimization is not intended to use custom_design_path.")

            assert name_design is None, (
                "'name_design' is not None. "+
                "The design optimization is not intended to use name_design.")

            assert custom_design is None, (
                "'custom_design' is not None. "+
                "The design optimization is not intended to use custom_design.")

            superstructure_path = os.path.join(self.data_in_path,  self.project['superstructures'])

            with open(superstructure_path, 'r', encoding='utf-8') as f:
                superstructures = yaml.load(f, Loader=UniqueKeyLoader)['superstructures']
                self.design_short = superstructures[self.config['superstructure_name']]

        elif self.objective in {'operation', 'max_overcapacity'}:
            assert (
                sum(x is not None for x in (custom_design, custom_design_path)) <= 1
            ), "Only one of 'custom_design' and 'custom_design_path' can be not None."
            if custom_design is None:
                if custom_design_path is None:
                    design_path = os.path.join(self.data_in_path,  self.project['designs'])
                else:
                    design_path = custom_design_path

                with open(design_path, 'r', encoding='utf-8') as f:
                    designs = yaml.load(f, Loader=UniqueKeyLoader)['designs']
                    self.design_short = designs[name_design]
            else:
                assert name_design is None, (
                    "If a custom design is given, it is not intended that a design name is given")
                self.design_short = custom_design

    def _load_general_input_yamls(self, project_file:str, ignore_tsa:bool):
        """Loads the general input yamls from the data_in_path and stores them
        in the class attributes.
        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
            ignore_tsa (bool): Forces this class to load the full timeseries.
                If True, the parameter self.config['time_series_aggregation']['load_tsa_inputs']
                is overwritten to False.
        """

        # Load project file:
        with open(os.path.join(self.data_in_path, project_file), 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)
        # Load config:
        with open(
            os.path.join(self.data_in_path,  self.project['config']), 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=UniqueKeyLoader)
        # Load technologies:
        with open(os.path.join(self.data_in_path,  self.project['technologies']),
                  'r', encoding='utf-8') as f:
            self.technologies = yaml.load(f, Loader=UniqueKeyLoader)['technologies']
        # Load products:
        with open(os.path.join(self.data_in_path,  self.project['products']),
                  'r', encoding='utf-8') as f:
            self.products = yaml.load(f, Loader=UniqueKeyLoader)
        # Load general_param:
        with open(os.path.join(self.data_in_path,  self.project['general_param']),
                  'r', encoding='utf-8') as f:
            self.general_param = yaml.load(f, Loader=UniqueKeyLoader)
        # Load timeseries:
        self._load_timeseries(project_file, ignore_tsa)
        # Load timeseries scaling:
        with open(os.path.join(self.data_in_path,  self.project['timeseries_scaling']),
            'r', encoding='utf-8') as f:
            self.ts_scaling = yaml.load(f, Loader=UniqueKeyLoader)['timeseries_scaling'][
                self.config['timeseries_scaling']
                ]

    def _load_timeseries(self, project_file:str, ignore_tsa:bool):
        """Loads the timeseries data and stores it as class attributes.
        TSA inputs are loaded, if wanted. If no TSA inputs are available, generate them.
        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
            ignore_tsa (bool): Forces this class to load the full timeseries.
                If True, the parameter self.config['time_series_aggregation']['load_tsa_inputs']
                is overwritten to False.
        """
        # If ignore_tsa==True, self.config['time_series_aggregation']['load_tsa_inputs']
        # is overwritten to False:
        if ignore_tsa:
            print("\tForcing UnscaledInputDictGenerator not to execute tsa.")
            self.config['time_series_aggregation']['load_tsa_inputs'] = False

        # If load_tsa_inputs from config.yaml is active, set respective timeseries_path from the
        # given path:
        if self.config['time_series_aggregation']['load_tsa_inputs']:
            self.data_path_timeseries = get_tsa_path(
                data_in_path = self.data_in_path,
                project_file = project_file)
            # Load tsa_export_project
            with open(os.path.join(self.data_path_timeseries, 'project_tsa_export.yaml'),
                'r', encoding='utf-8') as f:
                ts_filename = yaml.load(f, Loader=UniqueKeyLoader)['timeseries']
        else:
            self.data_path_timeseries = self.data_in_path

            ts_filename = self.project['timeseries']

        with open(
            os.path.join(self.data_path_timeseries,  ts_filename), 'r', encoding='utf-8') as f:
            self.timeseries = yaml.load(f, Loader=UniqueKeyLoader)

    def _generate_design(self):
        """Loops over all components in self.design_short and inserts properties from 
        self.technologies. If a property is already written in self.design_short,
        the property is skipped"""

        self.design = {}
        # Loop over self.design:
        for component_name, property_dict in self.design_short.items():
            self.design[component_name] = property_dict

            technology_dict = self.technologies[property_dict['type']]
            for tech_key, tech_value in technology_dict.items():
                if tech_key in self.design[component_name]:
                    continue
                else:
                    self.design[component_name][tech_key] = tech_value

    def _generate_processes_production_and_storage_lists(self):
        """ Loops over all components in self.design and creates list of production_processes"""
        self.processes_storage = []
        self.processes_production = []

        for component_name, property_dict in self.design.items():
            if property_dict['category'] == 'production':
                self.processes_production.append(component_name)
            elif property_dict['category'] == 'storage':
                self.processes_storage.append(component_name)
            else:
                raise ValueError(
                    f"category '{property_dict['category']}' is invaild. " +
                    "Select 'production' or 'storage'.")

    def _generate_product_list(self):
        """Loops over all components in self.design and creates a list of all products
        that are produced or stored in the system. Also adds all products from time_series, 
        if not already added."""
        self.product_list = list(self.products.keys())

        # Check that all products from processes are in product_list:
        for process_values in self.design.values():

            if process_values['category'] == 'production':
                # Get keys from efficiency_dict:
                if 'product_influenced_by_carnot_efficiency' not in process_values:
                    assert all(
                        product in self.products for product in process_values['efficiency'].keys()
                        )
            elif process_values['category'] == 'storage':
                # Get key from reference product:
                assert process_values['reference'] in self.products

        # Add products from timeseries:
        for ts_dict in self.timeseries.values():
            # Skip, if no entry "product" exists in the timeseries dict:
            if 'product' not in ts_dict:
                continue

            assert ts_dict['product'] in self.products

    def _add_components_to_input_dict(self):
        """Adds all components to the input_dict. The input_dict is a dictionary of dictionaries."""

        # Add lifetimes:
        self._add_component_property('production_lifetime', 'lifetime_a', 'production')
        self._add_component_property('storage_lifetime', 'lifetime_a', 'storage')
        # Add investment cost:
        self._add_component_property('production_investment_cost', 'investment_cost_EUR_kW',
                                     'production')
        self._add_component_property('storage_investment_cost', 'investment_cost_EUR_kWh',
                                     'storage')
        # Add maintenance cost:
        self._add_component_property('production_maintenance_cost', 'maintenance_cost_percent',
                                     'production')
        self._add_component_property('storage_maintenance_cost', 'maintenance_cost_percent',
                                     'storage')
        # Add operational cost:
        self._add_component_property('production_operational_cost', 'operational_cost_per_kWh',
                                     'production')
        # Add potential capacity:
        self._add_component_property('production_potential_capacity', 'potential_capacity_kW',
                                     'production')
        self._add_component_property('storage_potential_capacity', 'potential_capacity_kWh',
                                     'storage')
        # Add area per peak power
        self._add_component_property(
            'production_area_per_peak_power', 'area_per_peak_power', 'production',
            default_value=0) # if no area per peak power is given in technologies,  0 is assumed

        if self.config['model_formulation'] == 'MILP':
            # Add minimal capactiy:
            self._add_component_property('production_minimal_capacity', 'minimal_capacity_kW',
                                         'production')
            self._add_component_property('storage_minimal_capacity', 'minimal_capacity_kWh',
                                         'storage')
            # Add efficiencies for MILP formulation:
            self._add_efficiency_production_MILP()
        elif self.config['model_formulation'] == 'LP':
            # Add efficiencies for LP formulation:
            self._add_efficiency_production_LP()

        self._add_component_property('storage_efficiency_deposit', 'efficiency_deposit', 'storage')
        self._add_component_property('storage_efficiency_withdrawal', 'efficiency_withdrawal',
                                     'storage')
        self._add_component_property('storage_loss_percent', 'storage_loss_percent', 'storage')
        self._add_component_property('storage_reference', 'reference', 'storage')
        # Add storage minimal level
        self._add_component_property('storage_minimal_level', 'minimal_level', 'storage')
        # Add storage capacity ratio
        self._add_storage_capacity_ratio_to_input_dict()
        # Add process index lists:
        self.input_dict['processes_production'] = self.processes_production
        self.input_dict['processes_storage'] = self.processes_storage
        # Do sanity check for all components:
        for component in self.design:
            self._component_sanity_check(component)

    def _add_component_property(self, input_dict_name:str, design_name:str, process_type:str,
                                              default_value:float=None):
        """ adds the values of a property to self.input_dict for all processes of the given
        process_type. A property may have different names (keys) in self.input_dict and self.design.
        Therefore, there are two Args for the name of the property.

        Args:
            input_dict_name (str): Name (or key) of the property in self.input_dict
            design_name (str): Name (or key) of the property in technologies.yaml
            process_type (str): 'production' or 'storage'
            default_value (float): if design_name is not a key in technologies.yaml, the 
                default value will be used. Defaults to None.
        """
        if process_type == 'production':
            self.input_dict[input_dict_name] = {
                process: self.design[process].get(design_name, default_value)
                for process in self.processes_production
            }
        elif process_type == 'storage':
            self.input_dict[input_dict_name] = {
                process: self.design[process].get(design_name, default_value)
                for process in self.processes_storage
            }
        else:
            raise ValueError(f"process_type {process_type} is unknown.")

    def _add_efficiency_production_MILP(self):
        """ adds the efficiency parameters to the input_dict for the MILP formulation.
        The efficiency parameters are parametrized like in Reinert2023:
        https://doi.org/10.1016/j.compchemeng.2023.108176 """
        theta_dict =  {}
        theta_dot_dict = {}
        minload_dict = {}
        maxload_dict = {}

        # Loop over all production process:
        for process in self.processes_production:
            reference_product = self.design[process]['reference']
            eff_dict = self.design[process]['efficiency']
            eff_list_len = [len(value) for value in eff_dict.values()]

            # Make sure, that the reference product is in the effiency_dict:
            assert reference_product in eff_dict, (
                f"reference_product '{reference_product}' is not in {eff_dict.keys()}")
            # Make sure, all lists have the same length:
            assert len(set(eff_list_len)) == 1, (
                f"The efficiencies for {process} do not have the same length.")
            # Method is only implemented for 1 or two points:
            assert eff_list_len[0] in [1, 2], (
                "Partload-Efficiency is only implemented for one or two points.")
            # If only one point is given, add the point (0,0)
            if eff_list_len[0] == 1:
                for key in eff_dict.keys():
                    eff_dict[key].append(0)

            # Set properties for the reference product:
            coefficients = np.array(eff_dict[reference_product])
            minload_reference = float(np.min(np.abs(coefficients)))
            maxload_reference = float(np.max(np.abs(coefficients)))
            assert maxload_reference > 0

            # Update min/max-load and theta and theta_dict for all timeslices for reference product
            minload_dict.update({
                (process, ts): minload_reference for ts in self.input_dict['time_slices']})
            maxload_dict.update({
                (process, ts): maxload_reference for ts in self.input_dict['time_slices']})
            theta_dict.update({
                (reference_product, process, ts): float(coefficients[
                    np.argmin(np.abs(coefficients))])
                for ts in self.input_dict['time_slices']})
            theta_dot_dict.update({
                (reference_product, process, ts): float(
                    np.sign(coefficients[np.argmax(np.abs(coefficients))]) * 1)
                for ts in self.input_dict['time_slices']})

            # Loop over all products (except for the reference):
            for product_key, values in eff_dict.items():
                if product_key == reference_product:
                    continue
                values = np.array(values)
                minload = float(values[np.argmin(np.abs(values))])
                maxload = float(values[np.argmax(np.abs(values))])

                # Check if Product is influenced by carnot Efficiency (Heat Pumps)
                # -> Calculate Heat output (COP(T)) for each time step
                if (self.design[process].get('product_influenced_by_carnot_efficiency')
                    and product_key == 'carnot_efficiency'):
                    # Update the theta and theta_dot matrices for HTHP Heat
                    self._update_effiency_for_carnot_MILP(
                            minload, maxload, minload_reference, maxload_reference, process,
                            theta_dict, theta_dot_dict
                    )
                # In the normal case, calculate the theta and theta_dot value once
                # and use them for all timeslices
                else:
                    theta = minload
                    theta_dot = (maxload- minload) / (maxload_reference - minload_reference)
                    theta_dict.update({
                        (product_key, process, ts): theta for ts in self.input_dict['time_slices']})
                    theta_dot_dict.update({
                        (product_key, process, ts): theta_dot
                        for ts in self.input_dict['time_slices']})

        self.input_dict['production_minload'] = minload_dict
        self.input_dict['production_maxload'] = maxload_dict
        self.input_dict['production_theta'] = theta_dict
        self.input_dict['production_theta_dot'] = theta_dot_dict

        # Add zeros to not-added indices in input_dict expanded to all timeslices:
        self._fill_empty_entries_in_input_dict(
            target_key='production_theta',
            index_list=[
                self.product_list, self.processes_production, self.input_dict['time_slices']
                ])
        self._fill_empty_entries_in_input_dict(
            target_key='production_theta_dot',
            index_list=[
                self.product_list, self.processes_production, self.input_dict['time_slices']])

    def _update_effiency_for_carnot_MILP(
            self, min_carnot_eff, max_carnot_eff, minload_reference, maxload_reference, process,
            theta_dict, theta_dot_dict):
        """ Calculates the temperature-dependent COP and adds the theta and theta_dot values to
        the matrix
        
        In the Efficiency Matrix the Carnot Efficiency is not scaled with the reference product.
        Thats why the Reference (Electricity) has to be multiplied aswell
            -> heat_efficiency = Carnot_COP * Carnot_Efficency * Electricity_Efficiency
            -> Heat_Out_max = 0.5 * 5.4 * 1   = 2.7
            -> Heat_Out_min = 0.4 * 5.4 * 0.1 = 0.216
        """
        # Only implemented for HTHP:
        if process not in ['HTHP']:
            raise NotImplementedError("Process with canrot efficiency is not yet implemented.")
        # Check if value for Carnot efficiency is between 0 and 1
        if not (0 < min_carnot_eff < 1) and not 0 < max_carnot_eff < 1:
            raise ValueError("Carnot efficiency not between 0 and 1.")

        # Uses ambient Temperature (from csv) and Sink Temperature (from products.yaml)
        # to calculate Carnot COP
        T_ambient = np.array([
            value for (key, value) in self.input_dict['temperature'].items()
            if key[0] == 'temperature'])
        product_key = self.design[process]['product_influenced_by_carnot_efficiency']
        T_sink = np.array(
            [float(self.products[product_key]['temperature'])] *
            len(self.input_dict['time_slices']))
        COP_Carnot = (273.15 + T_sink) / (T_sink - T_ambient)

        # Calculates the Heat_Out Values described at the top of the function
        values_min = min_carnot_eff * COP_Carnot * minload_reference
        values_max = max_carnot_eff * COP_Carnot * maxload_reference

        theta_dot = (values_max - values_min) / (maxload_reference - minload_reference)

        # Add efficiency for products influenced by the Carnot efficiency to the matrices
        for ts, value in zip(self.input_dict['time_slices'], values_min):
            theta_dict[
                (self.design[process]['product_influenced_by_carnot_efficiency'], process, ts)
            ] = value

        for ts, value in zip(self.input_dict['time_slices'], theta_dot):
            theta_dot_dict[
                (self.design[process]['product_influenced_by_carnot_efficiency'], process, ts)
            ] = value

    def _add_efficiency_production_LP(self):
        """ adds the efficiency parameters to the input_dict for the LP formulation.
        The efficiency parameters are parametrized like in Reinert2022:
        https://doi.org/10.3389/fenrg.2022.884525 """
        technology_matrix =  {}

        # Loop over all production process:
        for process in self.processes_production:
            reference_product = self.design[process]['reference']
            eff_dict = self.design[process]['efficiency']

            # Make sure, that the reference product is in the effiency_dict:
            assert reference_product in eff_dict, (
                f"reference_product '{reference_product}' is not in {eff_dict.keys()}")
            # Make sure, all lists have the same length:
            assert (
                len({len(value) for value in eff_dict.values()}) == 1
            ), f"The efficiencies for {process} do not have the same length."

            # Set efficiency for the reference product:
            coefficients = np.array(eff_dict[reference_product])
            index_eff_ref = np.argmax(np.abs(coefficients))
            # For LP, only the data point with the maximal absolute value of the reference product
            # is selected
            eff_ref = float(coefficients[index_eff_ref])
            # Add efficiency for reference product expanded to all timeslices
            technology_matrix.update(
                {(reference_product, process, ts): eff_ref for ts in self.input_dict['time_slices']}
            )

            assert eff_ref != 0, (
                f"The efficiency for the reference product (here: '{reference_product}' in "+
                f" '{process}') must be != 0.")

            # Loop over all products:
            for product_key, values in eff_dict.items():
                values = np.array(values)

                # Check, that all entries are either non-positive or non-negative
                if not (np.all(values >= 0) or np.all(values <= 0)):
                    warnings.warn(
                        f"The technology matrix of '{process}' for '{product_key}' containts "+
                        "positive and negative values.")
                # Check, that no values occur double
                if len(set(values))!=len(values):
                    warnings.warn(
                        f"Double occuring values for '{product_key}' in the technology matrix of " +
                        f"'{process}'.")

                # Add effiency to technology-matrix for non-reference products:
                if product_key == reference_product:
                    continue

                # Check if Product is influenced by carnot Efficiency (Heat Pumps)
                # -> Calculate COP for each time step
                if (self.design[process].get('product_influenced_by_carnot_efficiency', False)
                    and product_key == 'carnot_efficiency'):
                    # Update the technology matric for HTHP Heat
                    self._update_effiency_for_carnot_LP(
                            values, index_eff_ref, eff_ref, process, technology_matrix
                    )
                # In the normal case input it the efficiency value for all timeslices
                else:
                    technology_matrix.update({
                        (product_key, process, ts): float(values[index_eff_ref])
                        for ts in self.input_dict['time_slices']})

                # Check, that the entry with the biggest absolute value is also the entry matching
                # the index of the reference products efficiency
                if values[np.argmax(np.abs(values))] != values[index_eff_ref]:
                    warnings.warn(
                        f"The maximum absolute value for '{product_key}' in the technology " +
                        f"matrix of '{process}' does not match. \n\t the position of the maximum "+
                        f"absolute value of the reference product '{reference_product}'."+
                        "\n\t Please doublecheck 'production_effiency' if this it intentional.")

        self.input_dict['production_technology_matrix'] = technology_matrix

        # Add zeros to not-added indices in input_dict:
        self._fill_empty_entries_in_input_dict(
            target_key='production_technology_matrix',
            index_list=[
                self.product_list, self.processes_production, self.input_dict['time_slices']]
        )

    def _update_effiency_for_carnot_LP(self, values, index_eff_ref, eff_ref, process,
                                       technology_matrix):
        """Calculates the temperature-dependent COP and adds the efficiency values to the
        technology-matrix.

        In the Efficiency Matrix the Carnot Efficiency is not scaled with the reference Product.
        Thats why the Reference (Electricity) has to be multiplied aswell (-1 because of positiv
        heat and negative electricity)
            -> values = Carnot_COP * Carnot_Efficency * (-1) * Electricity_Efficiency
            -> values = 0.5 * 5.4 * 1   = 2.7
        """
        # Only implemented for HTHP:
        if process not in ['HTHP']:
            raise NotImplementedError("Process with canrot efficiency is not yet implemented.")
        # Check if value for Carnot efficiency is between 0 and 1
        if not 0 < float(values[index_eff_ref]) < 1:
            raise ValueError("Carnot efficiency not between 0 and 1.")

        # Uses ambient Temperature (from csv) and Sink Temperature (from products.yaml) to
        # calculate Carnot COP
        T_ambient = np.array([
            value for (key, value) in self.input_dict['temperature'].items()
            if key[0] == 'temperature'])
        # Check that not all T_ambient values are zero, as this indicated a faulty time series:
        if set(T_ambient) == {0}:
            raise ValueError(
                f"Error for calculating the temperature dependent efficiency of '{process}': " +
                "Ambient temperature is zero for all timesteps.")
        T_sink = np.array(
            [float(self.products['heat']['temperature'])] * len(self.input_dict['time_slices']))
        COP_Carnot = (273.15 + T_sink) / (T_sink - T_ambient)

        # Calculates the Heat_Out Values described at the top of the function
        values = eff_ref * COP_Carnot * (-1) * float(values[index_eff_ref])

        # Add efficiency for products influenced by the Carnot efficiency to the Technology-Matrix
        for ts, value in zip(self.input_dict['time_slices'], values):
            technology_matrix[
                (self.design[process]['product_influenced_by_carnot_efficiency'], process, ts)
            ] = value

    def _add_storage_capacity_ratio_to_input_dict(self):
        """ Method creates the dict 'storage_capacity_ratio' and adds it to 
        self.input_dict. For the reference product of the storage_process, the
        capacity ratio is set. For all other products, the capacity ratios are 
        set to zero."""

        # Initialize dict:
        self.input_dict['storage_capacity_ratio'] = {}

        # Loop over all storage processes:
        for process in self.processes_storage:
            # Add capacity ratio for the reference product:
            reference_product = self.design[process]['reference']
            self.input_dict['storage_capacity_ratio'][
                (reference_product, process)] = self.design[process]['capacity_ratio']

        # Fill all other entries with zero:
        self._fill_empty_entries_in_input_dict(
            target_key='storage_capacity_ratio',
            index_list=[self.product_list, self.processes_storage])

    def _fill_empty_entries_in_input_dict(self, target_key:str, index_list:list, number:float=0):

        # Get keys of target_dict from input_dict:
        keys = self.input_dict[target_key].keys()

        # Check, if all keys have the same size as index_list:
        assert all(len(key) == len(index_list) for key in keys), (
            f"index_list (size {len(index_list)}) has not the same size as the target_dict.")

        # Generate list containing all possible key tuples:
        all_keys = list(itertools.product(*index_list))
        # # Check, if all keys from target_dict are in all_keys:
        # assert all(key in all_keys for key in keys), (
        #     "Invalid index_list provided: Index_list does not contain all indices from "+
        #     f"input_dict['{target_key}']")

        # Loop over all_keys, and add value 0, if the key is not already in keys:
        for key in all_keys:
            if key in keys:
                continue
            self.input_dict[target_key][key] = number

    def _add_availability_timeseries_to_input_dict(self):
        """ creates list 'limited_availability_processes' containing all production processes
        for which an availability timeseries exists and adds that list to input_dict.
        For each process in limited_availability_processes, the availability timeseries
        is added to self.input_dict
        """

        # Create 'limited_availability_processes'
        limited_avail_process_types = {
            ts_dict['process']: key for key, ts_dict in self.timeseries.items()
            if 'availability' in ts_dict['categories']
        }
        # Check, that all processes for which availabilities are given, are unique:
        assert len(set(limited_avail_process_types)) == len(limited_avail_process_types)

        # Get all production processes from the given category:
        limited_availability_processes = [
            process for process in self.processes_production
            if self.design[process]['type'] in limited_avail_process_types
        ]

        self.input_dict['production_availability'] = {}
        # Set availability time series:
        for process in self.processes_production:
            if process in limited_availability_processes:
                ts_name = limited_avail_process_types[self.design[process]['type']]
                input_df = pd.read_csv(
                    os.path.join(self.data_path_timeseries, self.timeseries[ts_name]['file']))

                for index, value in input_df['data'].items(): #ts_name
                    self.input_dict['production_availability'][(process, index)] = value

            else:
                self._fill_empty_entries_in_input_dict(
                    target_key = 'production_availability',
                    index_list = [[process], self.input_dict['time_slices']],
                    number=1
                )

    def _generate_product_list_from_timeseries(self, category:str):
        """ generates a list for a given category string which containts all products
        for which timeseries exist for the given category.
        Args:
            category(str): category of the timeseries, e.g. 'demand' or 'pricebuy'

        Returns:
            list: list of all products of this category
            list: list of the corresponding time_series_names:
        """
        product_list = []
        timeseries_name_list = []
        for ts_name, ts_dict in self.timeseries.items():
            if category in ts_dict['categories']:
                assert ts_dict['product'] not in product_list
                product_list.append(ts_dict['product'])
                timeseries_name_list.append(ts_name)
        return product_list, timeseries_name_list

    def _add_product_timeseries_to_input_dict(self, input_dict_key, category):
        """ adds a timeseries to the input_dict. 

        Args:
            input_dict_key (str): key name of the parameter in input_dict
            category (str): category of the timeseries, e.g. 'demand' or 'pricebuy'
        """
        # Generate demand list:
        product_list, ts_name_list = self._generate_product_list_from_timeseries(category)

        self.input_dict[input_dict_key] = {}
        for product, ts_name in zip(product_list, ts_name_list):
            input_df = pd.read_csv(
                os.path.join(self.data_path_timeseries, self.timeseries[ts_name]['file']))
            for index, value in input_df['data'].items():
                self.input_dict[input_dict_key] [(product, index)] = value

            assert list(input_df.index) == self.input_dict['time_slices']
            # Fill empty entries with 0:
            self._fill_empty_entries_in_input_dict(
                target_key = input_dict_key,
                index_list = [self.product_list, self.input_dict['time_slices']]
            )
        # If no entry for the category is given:
        if len(product_list) == 0:
            self._fill_empty_entries_in_input_dict(
                target_key = input_dict_key,
                index_list = [self.product_list, self.input_dict['time_slices']]
            )

    def _add_temperature_timeseries_to_input_dict(self):
        """Adds temperaute timeseries data to the input_dict
        """
        self.input_dict['temperature'] = {}
        ts_name_list = [
            ts_name
            for ts_name, ts_dict in self.timeseries.items()
            if 'timeseries' in ts_dict['categories']
        ]

        for ts_name in ts_name_list:
            ts_dict = self.timeseries[ts_name]
            input_df = pd.read_csv(os.path.join(self.data_path_timeseries, ts_dict['file']))
            for index, value in input_df['data'].items():
                self.input_dict['temperature'][('temperature', index)] = value

            assert list(input_df.index) == self.input_dict['time_slices']

        # Fill empty entries with 0:
        self._fill_empty_entries_in_input_dict(
            target_key='temperature',
            index_list=[['temperature'], self.input_dict['time_slices']]
        )

    def _couple_timeseries_price_sell_and_buy(self):
        """ This function couples the selling price with the purchase price. The coupling only
        occurs for products, which have price_sell_follows_price_buy=True in products.yaml and is
        defined by the price_sell_buy_ratio. The coupling assures, that its never profitable to buy
        and sell electricity in the same timestep. This is done, by subtracting the absolute value
        of the purchase price multiplied with (1- price_sell_buy_ratio).
        Therefore, price_sell < price_buy (even for negative purchase prices)
        """
        for product, product_dict in self.products.items():

            # Continue, if price_sell_follows_price_buy = False
            if not product_dict.get('price_sell_follows_price_buy', False):
                continue

            price_sell_buy_ratio = product_dict['price_sell_buy_ratio']
            assert price_sell_buy_ratio >= 0, (
                f"price_sell_buy_ratio for {product} must be >= 0, but is {price_sell_buy_ratio}.")

            # Otherwise, replace each entry with scaled entry
            for t in self.input_dict['time_slices']:
                price_buy = self.input_dict['pricebuy'][(product, t)]
                self.input_dict['pricesell'][(product, t)] = \
                    price_buy - (1 - price_sell_buy_ratio) * np.abs(price_buy)

    def _add_timeseries_scaling_demand(self):
        ### 1. Add Sets:
        self._add_timeseries_scaling_nonproportional_set('demand')

        ### 2. Add Parameters:
        # Add old values for mean and peak
        self.input_dict = add_ts_scaling_old_values('demand', self.input_dict)
        # Add bounds for mean and peak
        self._add_ts_scaling_bounds('demand')

        ### 3. Do Sanity Check:
        # Do general sanity check for the structure of timeseries_scaling.yaml:
        sanity_check_ts(self.ts_scaling)
        # Do specific sanity check for demand:
        sanity_check_ts_scaling_bounds(self.ts_scaling, self.input_dict, 'demand')
        sanity_check_ts_scaling_nonnegative(self.ts_scaling, self.input_dict, 'demand')

    def _add_timeseries_scaling_nonproportional_set(self, ts_type:str):
        """ Add set of products, where the timeseries can be scaled non-proportionally
                
        Args:
            ts_type (str): name of the timeseries
                e.g., 'demand' or 'pricebuy'
        """
        # Initialize empty list
        products_nonprop_scaling = []

        products, ts_scaling_keys = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type=ts_type, product_or_process='product'
        )

        for product, ts_scaling_key in zip(products, ts_scaling_keys):

            mean_and_peak = get_scale_mode_ts(self.ts_scaling[ts_scaling_key])

            # Add only product for 'mean_and_peak':
            if mean_and_peak == 'mean_and_peak':
                products_nonprop_scaling.append(product)

            elif mean_and_peak == 'mean':
                continue
            else:
                raise KeyError(f"Expected 'mean' or 'mean_and_peak'. Got {mean_and_peak} instead.")

        # Add to input dict:
        self.input_dict[f'products_{ts_type}_nonprop_scaling'] = products_nonprop_scaling

    def _add_ts_scaling_old_values(self, ts_type:str):
        """ Adds the old values (from the input .csv files) for a product timeseries
        to the input_dict

        Args:
            ts_type (str): name of the timeseries
                e.g., 'demand' or 'pricebuy'
        """
        ts_old_mean = {}
        ts_old_peak = {}

        for product in self.product_list:
            # Fill in mean:
            ts_old_mean[product] = calculate_weighted_mean(
                timeseries_dict = self.input_dict[ts_type],
                product_or_process= product,
                weights_dict=self.input_dict['time_slice_weight']
            )
            # Fill in peak:
            ts_old_peak[product] = max(
                self.input_dict[ts_type][k]
                for k in self.input_dict[ts_type].keys()
                if k[0] == product
            )

        # Add to input_dict:
        self.input_dict[f"ts_{ts_type}_old_mean"] = ts_old_mean
        self.input_dict[f"ts_{ts_type}_old_peak"] = ts_old_peak

    def _add_ts_scaling_bounds(self, ts_type:str):
        """ Add the min/max bounds for mean and peak.
        If mean and peak are not given in timeseries_scaling.yaml,
        the values from the .csv timeseries are taken to avoid scaling.

        Args:
            ts_type (str): name of the timeseries
                e.g., 'demand' or 'pricebuy'
        """
        mean_min = {}
        mean_max = {}
        peak_min = {}
        peak_max = {}

        products_for_ts_type, key_list = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type=ts_type, product_or_process='product'
        )

        ts_scaling_keys = dict(zip(products_for_ts_type, key_list))

        # Loop over products:
        for product in self.product_list:
            # If product is not in timeseries_scaling.yaml,
            # no scaling takes place. Add old values as bounds.
            if product not in products_for_ts_type:
                mean_min[product] = self.input_dict[f"ts_{ts_type}_old_mean"][product]
                mean_max[product] = self.input_dict[f"ts_{ts_type}_old_mean"][product]
                peak_max[product] = self.input_dict[f"ts_{ts_type}_old_peak"][product]
                peak_min[product] = self.input_dict[f"ts_{ts_type}_old_peak"][product]

                continue
            # Else: mean values are read from timeseries_scaling.yaml:
            mean_min[product] = self.ts_scaling[ts_scaling_keys[product]]['param']['mean']['min']
            mean_max[product] = self.ts_scaling[ts_scaling_keys[product]]['param']['mean']['max']

            # Get mean_or_peak
            mean_and_peak = get_scale_mode_ts(self.ts_scaling[ts_scaling_keys[product]])

            # Only mean: Use old values for peak:
            if mean_and_peak == 'mean':
                peak_max[product] = self.input_dict[f"ts_{ts_type}_old_peak"][product]
                peak_min[product] = self.input_dict[f"ts_{ts_type}_old_peak"][product]

            # Else: peak values are read from timeseries_scaling.yaml:
            elif mean_and_peak == "mean_and_peak":
                peak_min[product] = self.ts_scaling[
                    ts_scaling_keys[product]]['param']['peak']['min']
                peak_max[product] = self.ts_scaling[
                    ts_scaling_keys[product]]['param']['peak']['max']

        # Add dicts to input_dict:
        self.input_dict[f"ts_{ts_type}_mean_min"] = mean_min
        self.input_dict[f"ts_{ts_type}_mean_max"] = mean_max
        self.input_dict[f"ts_{ts_type}_peak_min"] = peak_min
        self.input_dict[f"ts_{ts_type}_peak_max"] = peak_max

    def _add_timeseries_scaling_pricebuy(self):
        ### 1. Add Parameters:
        # Add old values for mean and peak
        self.input_dict = add_ts_scaling_old_values('pricebuy', self.input_dict)

        ### 2. Do Sanity Check:
        # Do specific sanity check for pricebuy:
        sanity_check_ts_scaling_bounds(self.ts_scaling, self.input_dict, 'pricebuy')

    def _add_timeseries_scaling_availability(self):
        ### 1. Add Sets:
        self._add_availability_interpolation_set()

        ### 2. Add 1st Timeseries for interpolation:
        self._add_availability_interpolation_1st_timeseries()

        ### 3. Add 2nd Timeseries for interpolation:
        self._add_availability_interpolation_2nd_timeseries()

        ### 4. Add default interpolation scaler:
        self._add_default_availability_interpolation_scaler()

        ### 5. Overwrite production_availability with interpolation:
        self._overwrite_with_interpolated_availability()

    def _add_availability_interpolation_set(self):
        """ Adds the set of processes for which availabilities are interpolated
        """
        processes, _ = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type="availability", product_or_process='process'
        )
        # Only use processes which are part of the design:
        processes = [process for process in processes if process in self.design]

        self.input_dict['processes_availability_interpolation'] = processes

        # Warn, if first availability timeseries only contain ones:
        for process in processes:
            first_availability_ts = [
                val for i, val in self.input_dict['production_availability'].items()
                if i[0] == process]

            if min(first_availability_ts) == 1:
                warnings.warn(
                    "\nSomething might went wrong for availability interpolation " +
                    f"for '{process}' as the original timeseries only containts '1'."
                )

    def _add_availability_interpolation_1st_timeseries(self):
        """ Only for processes which interpolate availabilities: Copy the original availability from
        'production_availability' to 'availability_interpolation_1st_ts'.

        'availability_interpolation_1st_ts' is not a parameter used for the optimization. Instead,
        it is used for mutating 'production_availability'.
        """
        # Initialize empty dict:
        self.input_dict['availability_interpolation_1st_ts'] = {}

        # Get all ts_scaling keys with 'availability_interpolation' as ts_key
        processes, _ = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type="availability", product_or_process='process'
        )
        # Only use processes which are part of the design:
        processes = [process for process in processes if process in self.design]

        # Loop over the matching processes as keys:
        for process in processes:
            # # Copy values to dict:
            for t in self.input_dict['time_slices']:

                self.input_dict['availability_interpolation_1st_ts'][(process, t)] = float(
                    self.input_dict['production_availability'][(process, t)])

    def _add_availability_interpolation_2nd_timeseries(self):
        """ Only for processes which interpolate availabilities:
        Read .csv where the second timeseries is saved and add it to input_dict

        'availability_interpolation_2nd_ts' is not a parameter used for the optimization. Instead,
        it is used for mutating 'production_availability'.

        Raises:
            ValueError: If negative values occur in the availability timeseries
            ValueError: If values > 1 occur in the availability timeseries

        """
        # Initialize empty dict:
        self.input_dict['availability_interpolation_2nd_ts'] = {}

        # Get all ts_scaling keys with 'availability_interpolation' as ts_key
        processes, _ = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type="availability", product_or_process='process'
        )
        # Only use processes which are part of the design:
        processes = [process for process in processes if process in self.design]

        # Loop over the matching processes as keys:
        for process in processes:
            # Find entry in timeseries.yaml:
            timeseries_yaml = [
                val for val in self.timeseries.values()
                if val['categories'][0] == 'availability_interpolation'
            ]
            assert len(timeseries_yaml) == 1

            # Read .csv:
            filename = os.path.join(self.data_path_timeseries, timeseries_yaml[0]['file'])
            df_availability = pd.read_csv(filename)['data']

            # Sanity check:
            if df_availability.min() < 0:
                raise ValueError(
                    f"Availibility timeseries for {process} has negative " +
                    f"values: {df_availability.min()}"
                )

            if df_availability.max() > 1:
                raise ValueError(
                    f"Availibility timeseries for {process} has values " +
                    f"larger one: {df_availability.max()}"
                )

            # Add values to input dict:
            for index, value in df_availability.items():
                self.input_dict['availability_interpolation_2nd_ts'][(process, index)] = value

    def _add_default_availability_interpolation_scaler(self):
        """ adds the default scaler for the interpolation of availabilities. The default scaler is
        read from timeseries_scaling.yaml.

        Raises:
            ValueError: If the given default scaler is not between 0 and 1
        """
        # Initialize empty dict:
        self.input_dict['availability_interpolation_scaler'] = {}

        # Get all ts_scaling keys with 'availability_interpolate' as ts_key
        processes, ts_dict_keys = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type="availability", product_or_process='process'
        )
        # Only use processes which are part of the design:
        processes = [process for process in processes if process in self.design]

        # Loop over all matching processess:
        for process, ts_dict_key in zip(processes, ts_dict_keys):

            default_scaler = self.ts_scaling[ts_dict_key]['param']['interpolate']['default']

            # Sanity check: 0 <= default scaler <= 1
            if not 0 <= default_scaler <= 1:
                raise ValueError(
                    f"'{process}': 'default_scaler' must be between 0 and 1. " +
                    f"Instead, it is {default_scaler}."
                )

            # Add to input_dict:
            self.input_dict['availability_interpolation_scaler'][process] = default_scaler

    def _overwrite_with_interpolated_availability(self):
        """ Only for processes which interpolate availabilities: Overwrite 'production_availability'
        with interpolation from 'availability_interpolation_1st_ts' and 
        'availability_interpolation_2nd_ts'.
        """
        # Get all ts_scaling keys with 'availability_interpolation' as ts_key
        processes, _ = get_products_or_processes_scaling_ts(
            self.ts_scaling, ts_type="availability", product_or_process='process'
        )
        # Only use processes which are part of the design:
        processes = [process for process in processes if process in self.design]

        for process in processes:
            scaler = self.input_dict['availability_interpolation_scaler'][process]

            # continue if scaler is 0 (so no interpolation takes place)
            if scaler == 0:
                continue

            for t in self.input_dict['time_slices']:
                avail_1st = self.input_dict['availability_interpolation_1st_ts'][(process, t)]
                avail_2nd = self.input_dict['availability_interpolation_2nd_ts'][(process, t)]

                self.input_dict['production_availability'][(process, t)] = (
                    avail_1st * (1 - scaler) + avail_2nd * scaler
                )

    def _add_parameters_for_product_slack_term_to_input_dict(self):
        """
        Adds the parameters 'allow_lost_load' and 'value_of_lost_load' to the input_dict.
        'allow_lost_load' has one entry for each product. The entry can be either 0 or 1.
            0: an energy deficit is not allowed for that product. 
            1: an energy deficit is allowed for that product.
        'value_of_lost_load' has one entry for each product. The entries measure the cost of not 
        supplying the respective product in EUR/kWh.
        """

        if self.objective in {'design', 'operation'}:
            # Check, if 'allow_lost_load' exists in the products yaml for all products. If one
            # allow_lost_load is missing, the value is set to 0 (= not allowed) for all products:
            if all('allow_lost_load' in products_dict for products_dict in self.products.values()):

                # Make sure, all entries of 'allow_lost_load' are either 0 or 1
                assert all(
                    isinstance(val['allow_lost_load'], bool) for val in self.products.values()
                    ), 'Only binary values are allowed for allow_lost_load!'
                # Add allow_lost_load:
                self.input_dict['allow_lost_load'] = {
                    product_name: int(product_dict['allow_lost_load'])
                    for product_name, product_dict in self.products.items()
                }

                # Add value_of_lost_load:
                self.input_dict['value_of_lost_load'] = {
                    product_name: product_dict['value_of_lost_load']
                    for product_name, product_dict in self.products.items()
                }

                # Make sure, all entries of 'value_of_lost_load' are non-negative
                assert all(
                    val >= 0 for val in self.input_dict['value_of_lost_load'].values()
                    ), 'Only non-negative values are allowed for value_of_lost_load!'
            else:
                print("\t'allow_lost_load' is missing for at least one product.")
                print("\t'allow_lost_load' is set to 0 for all products.")
                self._disable_lost_load()

        # Lost load is not allowed in feasible_hull, set all values to 0
        elif self.objective in {'feasible_hull', 'near_optimal', 'max_overcapacity'}:
            self._disable_lost_load()

    def _disable_lost_load(self):
        """ There are cases, where you dont want lost_load to be possible. These cases are:
        1.) product.yaml is incomplete, 2.) objective is 'feasible_hull' or 'max_overcapacity'.
        In these cases, this method is called to deactive lost_load for all products.
        """
        # Do not allow lost load for any product:
        self.input_dict['allow_lost_load'] = {
            product_name: 0 for product_name in self.products.keys()}
        # Set default value of 100 EUR/kW:
        self.input_dict['value_of_lost_load'] = {
            product_name: 100 for product_name in self.products.keys()}

    def _add_importable_exportable_products_to_input_dict(self):
        # Check, that all products are unique:
        assert len(set(self.products.keys())) == len(self.products.keys()), (
            "products must be unique.")

        self.input_dict['importable_products'] = [
            product_name for product_name, product_dict in self.products.items()
            if product_dict['importable']]
        self.input_dict['exportable_products'] = [
            product_name for product_name, product_dict in self.products.items()
            if product_dict['exportable']]

    def _add_timeslices_to_input_dict(self):
        """ checks, if all timeseries have the same length
        """
        ts_list = []
        index_list = []
        # Loop over all timeseries:
        for ts_content in self.timeseries.values():

            input_df = pd.read_csv(os.path.join(self.data_path_timeseries, ts_content['file']))

            # Avoid error message if no renewable interpolation is used:
            if ts_content['categories'][0] == 'availability_interpolation' and len(input_df['data']) == 0:
                continue

            ts_list.append(list(input_df['data']))
            index_list.append(list(input_df.index))
        ts_length = [len(ts) for ts in ts_list]
        assert len(set(ts_length)) == 1
        assert all(index == index_list[0] for index in index_list)
        self.input_dict['time_slices'] = index_list[0]

    def _add_dummy_tsa_parameters_to_input_dict(self):
        """ adds dummy parameters to the input_dict. These parameters are:
        time_slice_weight, time_slice_duration, and next_time_slice. These parameters will be
        changed by the time series aggregation algorithm.
        """

        assert not self.config['time_series_aggregation']['load_tsa_inputs'], (
            "_add_dummy_tsa_parameters_to_input_dict must not be used, if 'load_tsa_inputs' in "+
            "config.yaml is active")

        # Add time_slice_weight (how many hours of the year are represented by this time_slice)
        self.input_dict['time_slice_weight'] = dict(
            zip(self.input_dict['time_slices'],
                [self.config['time_series_aggregation']['total_hours']
                 / len(self.input_dict['time_slices'])]*len(self.input_dict['time_slices']))
            )

        # Add time_slice_duration: without tsa, the time_slice_duration is 1
        self.input_dict['time_slice_duration'] = dict(
            zip(self.input_dict['time_slices'], [1]*len(self.input_dict['time_slices']))
            )

        # Add next_time_slice:
        self.input_dict['next_time_slice'] = dict(
            zip(self.input_dict['time_slices'], np.roll(self.input_dict['time_slices'],-1))
            )

    def _add_exported_tsa_parameters_to_input_dict(self):
        """
        Adds the exported tsa parameters to the input_dict. 
        Intended to be used, if 'load_tsa_inputs' in config.yaml is active. If the tsa_parameters
        are loaded, no subsequent time_series_aggregation is needed (and must not be used).
        """
        print(f"\tImporting tsa_input parameters from {self.data_path_timeseries}")
        # Assert that load_tsa_inputs is active
        assert self.config['time_series_aggregation']['load_tsa_inputs'], (
            "_add_exported_tsa_parameters_to_input_dict must not be used, if 'load_tsa_inputs' "+
            "in config.yaml is disabled")

        # Load tsa_project yaml:
        with open(os.path.join(self.data_path_timeseries, 'project_tsa_export.yaml'),
                  'r', encoding='utf-8') as f:
            tsa_project = yaml.load(f, Loader=UniqueKeyLoader)

        # Load time_slices from exported tsa data and assert, that it is equal to self.time_slices:
        with open(os.path.join(self.data_path_timeseries, tsa_project ['time_slices']),
                  'r', encoding='utf-8') as f:
            time_slices_import = yaml.load(f, Loader=UniqueKeyLoader)

        assert time_slices_import == self.input_dict['time_slices'], (
            f"time_slices importes from {self.data_path_timeseries} to not match time_slices " +
            "constructed from timeseries data.")

        # Import time_slice_weight and add to input_dict:
        with open(os.path.join(self.data_path_timeseries, tsa_project ['time_slice_weight']),
                  'r', encoding='utf-8') as f:
            self.input_dict['time_slice_weight'] = yaml.load(f, Loader=UniqueKeyLoader)

        # Import time_slice_duration and add to input_dict:
        with open(os.path.join(self.data_path_timeseries, tsa_project ['time_slice_duration']),
                  'r', encoding='utf-8') as f:
            self.input_dict['time_slice_duration'] = yaml.load(f, Loader=UniqueKeyLoader)

        # Import next_time_slice and add to input_dict:
        with open(os.path.join(self.data_path_timeseries, tsa_project ['next_time_slice']), 'r',
                  encoding='utf-8') as f:
            self.input_dict['next_time_slice'] = yaml.load(f, Loader=UniqueKeyLoader)

        # Sanity check: Ensure, that the exported tsa-data was created with the same tsa_parameters
        # as in the current config
        with open(os.path.join(self.data_path_timeseries, tsa_project ['tsa_config']), 'r',
                  encoding='utf-8') as f:
            tsa_import_config = yaml.load(f, Loader=UniqueKeyLoader)

        for key, val in tsa_import_config.items():
            config_value = self.config['time_series_aggregation'][key]
            if config_value != val:
                warnings.warn(
                    "Imported TSA data seems to be generated with a different config file." + 
                    f"\n\t{key}: '{config_value}' != '{val}' (expected value)")

    def _add_misc_to_input_dict(self):
        # Add discount rate:
        self.input_dict['discount_rate'] = self.general_param['discount_rate_percent']
        assert self.input_dict['discount_rate'] >= 0

        # Add time_horizon:
        self.input_dict['time_horizon'] = self.general_param['time_horizon_a']
        assert self.input_dict['time_horizon'] > 0

        # Add maximum_area (mostly used for PV and solar collector)
        self.input_dict['maximum_area'] = self.general_param['maximum_available_area']
        assert self.input_dict['maximum_area'] >= 0

        # Add products:
        self.input_dict['products'] = self.product_list

        # Add objective to config:
        self.input_dict['objective'] = self.objective

    def _fix_minimal_and_potential_capacities(self):
        """
        This function is intended to be used for operational optimizations. The parameters
        minimal_capacity and potential_capactiy are fixed to the nominal_capacities given in
        the design.yaml.
        If the model_formulation is 'LP', the parameters minimal_capacity_kW and
        minimal_capacity_kWh are not set.

        Raises:
            ValueError: If an unknown component_category (not production or storage) is used.
        """
        components = self.processes_production + self.processes_storage
        for component in components:
            component_category = self.design[component]['category']

            if component_category == 'production':
                nominal_capacity = self.design[component]['nominal_capacity_kW']

                self.design[component]['potential_capacity_kW'] = nominal_capacity
                # Set minimal_capacity_kW only for MILP
                if self.config['model_formulation'] == 'MILP':
                    self.design[component]['minimal_capacity_kW'] = nominal_capacity
            elif component_category == 'storage':
                nominal_capacity = self.design[component]['nominal_capacity_kWh']

                self.design[component]['potential_capacity_kWh'] = nominal_capacity
                # Set minimal_capacity_kWh only for MILP
                if self.config['model_formulation'] == 'MILP':
                    self.design[component]['minimal_capacity_kWh'] = nominal_capacity
            else:
                raise ValueError(f"component category {component_category} is unknown.")

    def _component_sanity_check(self, component:str):
        """
        Conducts sanity checks for the parameters of the given component.

        Args:
            component (str): Name of the component (e.g., 'B_large')

        Raises:
            ValueError: If one of the sanity checks fails
        """

        if self.design[component]['category'] == 'production':
            maintenance_cost_percent = self.input_dict['production_maintenance_cost'][component]
            lifetime_a               = self.input_dict['production_lifetime'][component]
            investment_cost          = self.input_dict['production_investment_cost'][component]
            potential_capacity       = self.input_dict['production_potential_capacity'][component]
            operational_cost_per_kWh = self.input_dict['production_operational_cost'][component]

            if self.objective in {'operation', 'max_overcapacity'}:
                nominal_capacity     = self.design[component]['nominal_capacity_kW']
                assert potential_capacity       >= nominal_capacity

            elif (self.objective == 'design') and (self.config['model_formulation'] == 'MILP'):
                minimal_capacity     = self.input_dict['production_minimal_capacity'][component]
                assert potential_capacity       >= minimal_capacity

        elif self.design[component]['category'] == 'storage':

            lifetime_a         = self.input_dict['storage_lifetime'][component]
            maintenance_cost_percent = self.input_dict['storage_maintenance_cost'][component]
            investment_cost          = self.input_dict['storage_investment_cost'][component]
            potential_capacity       = self.input_dict['storage_potential_capacity'][component]

            if self.objective in {'operation', 'max_overcapacity'}:
                nominal_capacity     = self.design[component]['nominal_capacity_kWh']
                assert potential_capacity       >= nominal_capacity

            elif (self.objective == 'design') and (self.config['model_formulation'] == 'MILP'):
                minimal_capacity     = self.input_dict['storage_minimal_capacity'][component]
                assert potential_capacity       >= minimal_capacity

        else:
            raise ValueError(f"component category {self.design[component]['category']} is unknown.")

        assert lifetime_a               >  0
        assert maintenance_cost_percent >= 0
        assert investment_cost          >= 0

        if self.design[component]['category'] == 'production':
            assert operational_cost_per_kWh >= 0

    def _LP_sanity_check(self):
        """
        Checks, if all parameters are set correctly for a LP formulation. 
        If not, a warning is printed.
        """
        if self.config['model_formulation'] == 'MILP':
            return

        # Print warning, if there are multiple support points for the production process effiencies:
        if processes_multi_support := [
            process
            for process in self.processes_production
            if max(
                len(product)
                for product in self.design[process][
                    'efficiency'
                ].values()
            )
            > 1
        ]:
            warnings.warn(
                f"For {processes_multi_support} multiple support points are given." +
                "\n\t In the LP formulation, only the full load efficiency is considered. " +
                "The other support points are ignored.")

    def _add_feasibility_affecting_sets(self):
        """"Adds feasibility-related sets to the input dict"""

        # Initialize empty sets
        production_affect_feas  = []
        storage_affect_feas     = []
        demand_mean_affect_feas = []
        demand_peak_affect_feas = []

        # Check technolgies
        for tech_id in self.design_short:
            tech_data = self.technologies[tech_id]
            if tech_data['affect_feasibility']:

                if tech_data['category'] == 'production':
                    production_affect_feas.append(tech_id)
                elif tech_data['category'] == 'storage':
                    storage_affect_feas.append(tech_id)
                else:
                    raise ValueError(
                        f"Unexpected category: '{tech_data['category']}'."
                    )

        # Check timeseries scaling
        for ts_data in self.ts_scaling.values():
            if ts_data['affect_feasibility']:
                if 'mean' in ts_data['param']:
                    demand_mean_affect_feas.append(ts_data['product'])
                else:
                    raise ValueError(
                        f"Unexpected: no 'mean' value is given for '{ts_data['product']}'."
                    )

                if 'peak' in ts_data['param']:
                    demand_peak_affect_feas.append(ts_data['product'])

        # add to input dict
        self.input_dict['production_affect_feas']  = production_affect_feas
        self.input_dict['storage_affect_feas']     = storage_affect_feas
        self.input_dict['demand_mean_affect_feas'] = demand_mean_affect_feas
        self.input_dict['demand_peak_affect_feas'] = demand_peak_affect_feas

    def _add_feasibility_extreme_values(self):
        """"Adds all feasibility extreme values to the input_dict. These values are used in the 
        first iterations (MinMax) for the feasibile hull exploration. After the MinMax iterations,
        these extreme values will be overwritten."""

        # Production processes
        self.input_dict['min_values_production'] = {
            m: 0 for m in self.input_dict['production_affect_feas']}
        self.input_dict['max_values_production'] = {
            m: 1 for m in self.input_dict['production_affect_feas']}

        # Storage processes
        self.input_dict['min_values_storage'] = {
            m: 0 for m in self.input_dict['storage_affect_feas']}
        self.input_dict['max_values_storage'] = {
            m: 1 for m in self.input_dict['storage_affect_feas']}

        # Demand mean values
        self.input_dict['min_values_demand_mean'] = {
            m: 0 for m in self.input_dict['demand_mean_affect_feas']}
        self.input_dict['max_values_demand_mean'] = {
            m: 1 for m in self.input_dict['demand_mean_affect_feas']}

        # Demand peak values
        self.input_dict['min_values_demand_peak'] = {
            m: 0 for m in self.input_dict['demand_peak_affect_feas']}
        self.input_dict['max_values_demand_peak'] = {
            m: 1 for m in self.input_dict['demand_peak_affect_feas']}

    def _add_default_feasibility_search_directions(self):
        """"Adds default search directions for all feasibility affecting parameters to the
        input_dict"""

        # Production processes
        self.input_dict['default_search_dir_production'] = {
            tech: 0 for tech in self.input_dict['production_affect_feas']
        }

        # Storage processes
        self.input_dict['default_search_dir_storage'] = {
            tech: 0 for tech in self.input_dict['storage_affect_feas']
        }

        # Demand mean values
        self.input_dict['default_search_dir_demand_mean'] = {
            tech: 0 for tech in self.input_dict['demand_mean_affect_feas']
        }

        # Demand peak values
        self.input_dict['default_search_dir_demand_peak'] = {
            tech: 0 for tech in self.input_dict['demand_peak_affect_feas']
        }

    def _add_default_optimal_tac(self):
        """Adds a default value of the TAC of a design optimization."""
        self.input_dict["opt_tac"] = 0.0

    def _add_default_slack(self):
        """Adds a default value for the slack."""
        self.input_dict["slack"] = 0.0

    def _add_processes_extreme_values(self):
        """"Adds all processes extreme values to the input_dict. These values are used in the first
        iterations (MinMax) for the near-optimal space exploration. After the MinMax iterations,
        these extreme values will be overwritten."""

        prod_procs: list[str] = self.input_dict['processes_production']
        stor_procs: list[str] = self.input_dict['processes_storage']

        # Production processes
        self.input_dict['min_values_production'] = {
            tech: 0 for tech in prod_procs}
        self.input_dict['max_values_production'] = {
            tech: self.technologies[tech]['potential_capacity_kW'] for tech in prod_procs}

        # Storage processes
        self.input_dict['min_values_storage'] = {
            tech: 0 for tech in stor_procs}
        self.input_dict['max_values_storage'] = {
            tech: self.technologies[tech]['potential_capacity_kWh'] for tech in stor_procs}

    def _add_default_near_optimal_search_directions(self):
        """
        Set the search direction for every process (production and storage) to zero (default to
        generate the model). 
        """
        all_processes: list[str] = (
            self.input_dict['processes_production'] +
            self.input_dict['processes_storage']
        )
        self.input_dict['default_search_dir_processes'] = {
            tech: 0 for tech in all_processes
        }

class ScaledInputDictGenerator():
    """
    Scales the unscaled input dict to account for scenario uncertainties. 
    Is mostly used for stochastic programming. If no uncertainties are given, a single
    unscaled scenario is used.
    WARNING: Even if no uncertainties are given, this class must be used to scale the
        input_dict for the optimization, as it introduces the set 'scenarios' to the input_dict.

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the
            input information (e.g. project.yaml)
        unscaled_input_dict (dict): dict generated by UnscaledInputDictGenerator
            or AggregatedInputDictGenerator
        sceanrio_df (pd.DataFrame): DataFrame defining the scenario generated by
            scenario_generatio.ScenarioGenerator. If None, one standard scenario is used.
    """
    def __init__(self, data_in_path, project_file: str, unscaled_input_dict:dict,
                 scenario_df:pd.DataFrame=None):
        print("Scaled_InputDict_Generator:")
        self.data_in_path = data_in_path
        assert os.path.exists(self.data_in_path)
        self._load_general_input_yamls(project_file)

        self.unscaled_input_dict = unscaled_input_dict

        # If no scenario is given, generate a single unscaled scenario:
        if scenario_df is None:
            self.scenario_df = pd.DataFrame({'scenarios': ['base'], 'probability': [1]})
            print("\tNo uncertain parameter is given. The input_dict is not scaled.")

        else:
            self.scenario_df = scenario_df
        # Add scenarios:
        self.scenarios = list(self.scenario_df['scenarios'])
        # Add uncertain_parameter_names
        self.uncertain_parameter_names = [
            x for x in list(self.scenario_df.columns)
            if x not in ['scenarios', 'probability', 'cluster']]
        assert all(name in self.uncertainties for name in self.uncertain_parameter_names), (
            f"Uncertain parameter names {self.uncertain_parameter_names} must be in " +
            "uncertain_parameters.yaml")

        # Initialize input dict:
        self.scaled_input_dict = {}

    def _load_general_input_yamls(self, project_file:str):
        """
        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
        """
        # Load project file:
        with open(os.path.join(self.data_in_path, project_file), 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)
        # Load name_uncertainties from config:
        with open(os.path.join(self.data_in_path, self.project['config']),
                  'r', encoding='utf-8') as f:
            name_uncertainties = yaml.load(
                f, Loader=UniqueKeyLoader)['config_uncertainties']['name_uncertainties']
        # Load uncertainties:
        with open(os.path.join(self.data_in_path, self.project['uncertainties']), 'r',
                  encoding='utf-8') as f:
            self.uncertainties = yaml.load(
                f, Loader=UniqueKeyLoader)['uncertainties'][name_uncertainties]
        # Load scenario_independent_parameters:
        with open(os.path.join(self.data_in_path, self.project['uncertainties']), 'r',
                  encoding='utf-8') as f:
            self.scenario_independent_parameters = yaml.load(
                f, Loader=UniqueKeyLoader)['scenario_independent_parameters']

    def generate_scaled_input_dict(self, data_export_path:str=None):
        """ Main function. Scales the unscaled input dict to account for uncertainties.
        Is mostly used for stochastic programming.

        Args:
            data_export_path (str, optional): Path where the scaled dict will be saved.
                None: nothing is exported. Defaults to None.

        Returns:
            dict: scaled input dict
        """
        print("\tGenerating scaled input_dict...")
        # 1. Copy scenario independent parameters
        self._copy_scenario_independent_parameters()

        for scenario in self.scenarios:
            self._add_single_scenario_to_input_dict(scenario)

        # 4. Add scenarios and scenario_probabilities to input_dict:
        self.scaled_input_dict['scenarios'] = self.scenarios
        self._add_probabilities_to_input_dict()

        if data_export_path is not None:
            print(f"\tExporting scaled input_dict to {data_export_path}.pkl")
            with open(f"{data_export_path}.pkl", 'wb') as outfile:
                pickle.dump(self.scaled_input_dict, outfile)

        print("\tDone: Generated scaled input_dict.")
        return self.scaled_input_dict

    def _add_single_scenario_to_input_dict(self, scenario):

        # 2. Copy scenario dependent parameters
        self._copy_scenario_dependent_parameters(scenario)
        # 3. Apply uncertain parameters to scenario dependent parmaeters
        for uncertain_parameter_name in self.uncertain_parameter_names:
            self._apply_uncertain_parameter(scenario, uncertain_parameter_name)

    def _copy_scenario_independent_parameters(self):
        """
        The scenario_independent parameters defined in 'uncertainties.yaml'
        are just copied and not altered.
        """
        for parameter in self.scenario_independent_parameters:
            # Copy the parameter value, if the parameter from scenario_independent parameters
            # is a key in the unscaled_input_dict
            if parameter in self.unscaled_input_dict:
                self.scaled_input_dict[parameter] = deepcopy(self.unscaled_input_dict[parameter])

    def _copy_scenario_dependent_parameters(self, scenario):
        """ Method copies all scenario dependent parameters to the scaled_input_dict."""
        for parameter_name, parameter_dict in self.unscaled_input_dict.items():
            if parameter_name in self.scenario_independent_parameters:
                continue
            # Check, if dict_name already exists:
            if parameter_name not in self.scaled_input_dict:
                self.scaled_input_dict[parameter_name] = {}

            # If the input_value is not a dict, a dict is created
            if not isinstance(parameter_dict, dict):
                self.scaled_input_dict[parameter_name].update({scenario: parameter_dict})
                continue
            # If the dict is empty, continue
            elif len(parameter_dict) == 0:
                continue
            # If the key is a tuple, add scenario name to tuple
            elif isinstance(list(parameter_dict.keys())[0], tuple):
                self.scaled_input_dict[parameter_name].update({
                    key + (scenario,):value for key, value in parameter_dict.items()})
            else:
                self.scaled_input_dict[parameter_name].update({
                    (key, scenario):value for key, value in parameter_dict.items()})

    def _apply_uncertain_parameter(self, scenario:str, uncertain_parameter_name:str):
        """ Method applies the scaling factor to the affected parameters of the uncertain parameter.

        Args:
            scenario (str): name of the scenario (e.g. 's1')
            uncertain_parameter_name (str): name of the uncertain parameter (e.g. 'f_c_el')
        """
        affected_param_dict = self.uncertainties[uncertain_parameter_name]['param']

        for affected_param in affected_param_dict.values():

            # Get_affected_index_tuples
            affected_index_tuples = self.get_affected_index_tuples(scenario, affected_param)
            affected_param_name = affected_param['name']

            # Get scaling factor
            scaling_factor = self.scenario_df[
                self.scenario_df['scenarios'] == scenario][uncertain_parameter_name].values[0]
            assert scaling_factor >= 0, (
                f"Scaling factor for {uncertain_parameter_name} in scenario {scenario} " +
                "must be greater than 0.")
            # Apply scaling factor
            for index_tuple in affected_index_tuples:
                self.scaled_input_dict[affected_param_name][index_tuple] = self.scaled_input_dict[
                    affected_param_name][index_tuple] * scaling_factor

    def get_affected_index_tuples(self, scenario, affected_parameter):
        """ Method filters keys of the affected parameter's dict, to create a list
        of keys on which the uncertain parameter will be applied.

        Args:
            scenario (str): name of the scenario. This is added to the end of all keys
            affected_parameter (dict): contains information about the name and the indices
                of the affected parameter. e.g. self.uncertainties['f_c_el']['param']['p1']

        Returns:
            list of tuples: lists all keys of the affected parameters dict on which the
                uncertain parameter has an effect
        """
        name = affected_parameter['name']
        pre_index = affected_parameter['index']
        all_keys = self.scaled_input_dict[name].keys()

        # If scaled_input_dict is empty, return empty list:
        if len(all_keys) == 0:
            return []

        # Check, that all indices from the affected parameter have the same length:
        length_set = list({len(key) for key in all_keys})
        assert len(length_set) == 1, "Indices from affected parameter must have the same length."
        assert length_set[0]== len(pre_index) + 1, "Length of indices should be one less than keys."

        # Create list containing all allowed indices:
        #       e.g. [['electricity'], [1, 2, 3, 4, 5], ['s1', 's2', 's3]]
        allowed_indices = []
        for i, index_name in enumerate(pre_index):
            if index_name == 'all':
                allowed_indices.append(
                    list({key[i] for key in all_keys})
                )
            else:
                allowed_indices.append([index_name])
        # Add scenarios:
        allowed_indices.append(scenario)
        return [
            key
            for key in all_keys
            if all(key[i] in allowed_indices[i] for i in range(len(key)))
        ]

    def _add_probabilities_to_input_dict(self):
        """
        Adds probabilities to the input dictionary.

        Raises:
            ValueError: If any probability is negative.

        """
        # 1. Check, that no probability is negative:
        if any(self.scenario_df['probability'] < 0):
            raise ValueError("Probabilities must be greater than zero.")

        # 2. Check, if a probability is zero and warn the user
        if any(self.scenario_df['probability'] == 0):
            # Count the number of scenarios with a probability of zero:
            zero_prob_count = len(self.scenario_df[self.scenario_df['probability'] == 0])
            warnings.warn(
                f"{zero_prob_count} scenarios have a probability of zero. "+
                "If its intended, ignore this message.")

        # 3. Check, if probabilities add up to 1. If not, normalize them:
        if abs(self.scenario_df['probability'].sum() - 1) > 1e-10:
            warnings.warn(
                "Probabilities do not add up to 1 (sum = " +
                f"{self.scenario_df['probability'].sum()}). Normalizing them now.")
            self.scenario_df['probability'] = self.scenario_df[
                'probability'] / self.scenario_df['probability'].sum()

        self.scaled_input_dict['probability'] = {}
        # Go through scenario_df row by row and add the probabilities to the scaled_input_dict:
        for _, row in self.scenario_df.iterrows():
            # Add the probability to the input_dict
            self.scaled_input_dict['probability'][row['scenarios']] = row['probability']

class AggregatedInputDictGenerator():
    """ Aggregates the unscaled input dict to reduce the number of time slices.
    The aggregation is done with the tsam package.

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path
            (e.g. project.yaml)
        unaggregated_input_dict (dict): dict generated by UnscaledInputDictGenerator
    """
    def __init__(self, data_in_path, project_file: str, unaggregated_input_dict:dict):
        print("Aggregated_InputDict_Generator:")
        self.data_in_path = data_in_path
        assert os.path.exists(self.data_in_path)
        self.default_tsa_export_path = get_tsa_path(data_in_path, project_file)
        self._load_general_input_yamls(project_file)

        self.unaggregated_input_dict = unaggregated_input_dict.copy()

        # Initialize class attributes:
        self.time_slices           = []
        self.shifted_time_slices   = []
        self.time_slice_weight     = []
        self.time_slice_duration   = []
        self.next_time_slice       = []
        self.aggregation           = None
        self.aggregated_input_dict = {}

    def _tsa(self):
        # Set self.full_time_series_df to the combination of all time dependent dictionaries.
        full_time_series_df = self._get_all_time_series()

        # Create an instance of the TimeSeriesAggregation class
        self.aggregation = tsam.TimeSeriesAggregation(
                timeSeries         = full_time_series_df, # Perform TSA on the combined DataFrame
                noTypicalPeriods   = self.config['time_series_aggregation']['noTypicalPeriods'],
                hoursPerPeriod     = self.config['time_series_aggregation']['hoursPerPeriod'],
                clusterMethod      = self.config['time_series_aggregation']['clusterMethod'],
                resolution         = 1, # Distance threshold below which clusters are merged
                segmentation       = self.config['time_series_aggregation']['segmentation'],
                numericalTolerance = 1e-12,
                extremePeriodMethod= self.config['time_series_aggregation'].get(
                    'extremePeriodMethod', 'None'),
                addPeakMax = self._get_tuples_to_consider_with_extreme_peroid(),
                # tsam currently has a bug for numericalTolerance (version 2.3.6)
                # can be fixed by replacing
                # if abs(diff).max() > self.numericalTolerance:
                # In line 1121 with
                # if abs(diff[warning_list[warning_list>0].index]).max() > self.numericalTolerance:
            )

        # Generate the typical periods based on the specified parameters:
        full_time_series_df = self.aggregation.createTypicalPeriods()

        # Reindex self.full_time_series_df to have unique single-dimensional index starting from 0
        full_time_series_df = full_time_series_df.reset_index()

        # Create an array of integers ranging from 0 to (noTypicalPeriods * hoursPerPeriod) - 1
        # this array will be the new index for self.full_time_series_df:
        hours_per_period = self.config['time_series_aggregation']['hoursPerPeriod']
        n_timesteps = len(full_time_series_df) #n_typical_periods * hours_per_period

        full_time_series_df.index = np.arange(0,n_timesteps)

        self.time_slices = list(full_time_series_df.index)

        # Loop over all typical periods identified in the dataset.
        for typical_period in np.arange(n_timesteps / hours_per_period):
            if self.aggregation.segmentation:
                # Do the stuff for segmentation, not implemented yet
                raise NotImplementedError("Segmentation is not yet implemented.")

            # Identify all time slices (index) that belong to that typical period
            time_slices_within_typical_period = full_time_series_df[
                full_time_series_df.level_0 == typical_period].index

            # Handle wrapping around for the last element
            shifted_time_slices_within_period = np.roll(time_slices_within_typical_period, -1)
            self.shifted_time_slices.extend(shifted_time_slices_within_period)
            # Calculate the weight of each time slice in a typical period.
            self.time_slice_weight.extend(
                self.aggregation.clusterPeriodNoOccur[typical_period] * \
                    np.ones(len(time_slices_within_typical_period))
                    )

            # The duration represents the "real" duration within a cluster:
            self.time_slice_duration.extend(
                np.ones(self.aggregation.hoursPerPeriod) * self.aggregation.resolution)

        n_time_slices = len(self.unaggregated_input_dict['time_slices'])
        total_hours = self.config['time_series_aggregation']['total_hours']
        for value in self.time_slice_weight:
            value *= total_hours / n_time_slices

        # Create a dictionary mapping each time slice to its weight:
        self.time_slice_weight = dict(zip(self.time_slices, self.time_slice_weight))

        # Create a dictionary mapping each time slice to its consecutive time slice:
        self.next_time_slice = dict(zip(self.time_slices, self.shifted_time_slices))

        # Create a dictionary mapping each time slice to its duration:
        self.time_slice_duration = dict(zip(self.time_slices, self.time_slice_duration))

        return full_time_series_df

    def _load_general_input_yamls(self, project_file: str):
        """
        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
        """
        # Load project file:
        with open(os.path.join(self.data_in_path, project_file), 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)
        # Load config:
        with open(
            os.path.join(self.data_in_path, self.project['config']), 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=UniqueKeyLoader)
        # Load timeseries:
        with open(
            os.path.join(self.data_in_path, self.project['timeseries']),'r', encoding='utf-8') as f:
            self.timeseries_original = yaml.load(f, Loader=UniqueKeyLoader)
        # Load ts_scaling:
        with open(os.path.join(self.data_in_path, self.project['timeseries_scaling']),
                  'r', encoding='utf-8') as f:
            self.ts_scaling = yaml.load(
                f, Loader=UniqueKeyLoader
                )['timeseries_scaling'][self.config['timeseries_scaling']]
        # Load products:
        with open(os.path.join(self.data_in_path, self.project['products']),
                  'r', encoding='utf-8') as f:
            self.products = yaml.load(f, Loader=UniqueKeyLoader)

    def _get_all_time_series(self):
        """Combines all time dependent dictionaries into one DataFrame
        used as the input for Time Series Aggregation
        """

        demand_df = convert_to_dataframe(
            self.unaggregated_input_dict['demand'],
            self.unaggregated_input_dict['products'],
            'demand')
        pricebuy_df = convert_to_dataframe(
            self.unaggregated_input_dict['pricebuy'],
            self.unaggregated_input_dict['products'],
            'pricebuy')
        pricesell_df = convert_to_dataframe(
            self.unaggregated_input_dict['pricesell'],
            self.unaggregated_input_dict['products'],
            'pricesell')
        availability_df = convert_to_dataframe(
            self.unaggregated_input_dict['production_availability'],
            self.unaggregated_input_dict['processes_production'],
            'production_availability')
        availability_interpolation_1st_df = convert_to_dataframe(
            self.unaggregated_input_dict['availability_interpolation_1st_ts'],
            self.unaggregated_input_dict['processes_availability_interpolation'],
            'availability_interpolation_1st_ts')
        availability_interpolation_2nd_df = convert_to_dataframe(
            self.unaggregated_input_dict['availability_interpolation_2nd_ts'],
            self.unaggregated_input_dict['processes_availability_interpolation'],
            'availability_interpolation_2nd_ts')

        temperature_df = pd.DataFrame(
            {('temperature', 'temperature'): [
                value for key, value in self.unaggregated_input_dict['temperature'].items()
                if key[0] == 'temperature'
                ]})

        return pd.concat([
            demand_df,
            pricebuy_df,
            pricesell_df,
            availability_df,
            availability_interpolation_1st_df,
            availability_interpolation_2nd_df,
            temperature_df,
            ],
            axis=1)

    def _convert_from_tsa_format_to_tuple(self, input_dict_df: dict):
        """ Converts an aggregated dictionary back to pyomo tuple format to be later
        added into aggregated_input_dict.

        Args: 
            input_dict_df (pd.DataFrame): Aggregated dictionary with the columns 'index',
                'level_0', 'TimeStep', ... .

        Returns:
            dict: Dictionary with keys in tuple format: 'demand':
                {('heat', 0): ..., ('electricity', 0): ... }, 'pricebuy': {...}
        """

        output_dict = {}
        cols = input_dict_df.keys()[2:] # Consider all columns apart from level_0 and TimeStep

        for col in cols:
            # The headers are stored in the format (product, category)
            category = col[1]

            # Initialize create an empty dictionary for each category if needed
            if output_dict.get(category) is None:
                output_dict[category] = {}

            index = 0
            product = col[0]
            for value in input_dict_df[col]:
                output_dict[category][(product, index)] = value
                index += 1

        return output_dict

    def generate_aggregated_input_dict(self, data_export_path:str=None):
        """ main function of AggregatedInputDictGenerator. Does time series aggregation and returns
        aggregated dict

        Args:
            data_export_path (str, optional): Path where the aggregated input dict should be saved.
                If no path is given, nothing is exported. Defaults to None.

        Returns:
            dict: aggregated input dict
        """

        full_time_series_df = self._tsa()

        # Convert self.full_time_series_df back to tuple format
        full_time_series_df = self._convert_from_tsa_format_to_tuple(full_time_series_df)

        # First, we initialize aggregated_input_dict with unaggregated_input_dict
        # Later, we overwrite time-dependent keys in aggregated_input_dict
        aggregated_input_dict = self.unaggregated_input_dict

        # Overwrite the time-dependent keys in aggregated_input_dict
        aggregated_input_dict['time_slices']             = self.time_slices
        aggregated_input_dict['time_slice_weight']       = self.time_slice_weight
        aggregated_input_dict['next_time_slice']         = self.next_time_slice
        aggregated_input_dict['time_slice_duration']     = self.time_slice_duration
        aggregated_input_dict['demand']                  = full_time_series_df['demand']
        aggregated_input_dict['pricebuy']                = full_time_series_df['pricebuy']
        aggregated_input_dict['pricesell']               = full_time_series_df['pricesell']
        aggregated_input_dict['temperature']             = full_time_series_df['temperature']
        aggregated_input_dict['production_availability'] = full_time_series_df[
            'production_availability']

        aggregated_input_dict['availability_interpolation_1st_ts'] = full_time_series_df.get(
            'availability_interpolation_1st_ts', {})
        aggregated_input_dict['availability_interpolation_2nd_ts'] = full_time_series_df.get(
            'availability_interpolation_2nd_ts', {})

        # Update ts_scaling:
        aggregated_input_dict = self._update_timeseries_scaling_demand(aggregated_input_dict)
        aggregated_input_dict = self._update_timeseries_scaling_pricebuy(aggregated_input_dict)

        # Set as class attribute
        self.aggregated_input_dict = aggregated_input_dict

        # Check, that weight * duration == total_hours with tolerance of 1e-10
        self.sanity_check()

        if data_export_path is not None:
            with open(f"{data_export_path}.yaml", 'w', encoding='utf-8') as outfile:
                yaml.dump(
                    self.aggregated_input_dict, outfile, default_flow_style=False, sort_keys=False
                    )

        return self.aggregated_input_dict

    def _update_timeseries_scaling_demand(self, aggregated_input_dict: dict):
        ### 1. Update Sets:
        # Sets do not need to be updated, as the TSA does not change them.

        ### 2. Update Parameters:
        # Update old values for mean and peak
        aggregated_input_dict = add_ts_scaling_old_values(
            'demand', aggregated_input_dict)
        # Bounds do not need to be updated, as the TSA does not change them.

        ### 3. Do Sanity Check:
        # Do general sanity check for the structure of timeseries_scaling.yaml:
        # Not really necessary. This done, just to be sure
        sanity_check_ts(self.ts_scaling)
        # Do specific sanity check for demand:
        sanity_check_ts_scaling_bounds(
            self.ts_scaling, aggregated_input_dict, 'demand')
        sanity_check_ts_scaling_nonnegative(
            self.ts_scaling, aggregated_input_dict, 'demand')

        return aggregated_input_dict

    def _update_timeseries_scaling_pricebuy(self, aggregated_input_dict: dict):
        ### 1. Update Sets:
        # Sets do not need to be updated, as the TSA does not change them.

        ### 2. Update Parameters:
        # Update old values for mean and peak
        aggregated_input_dict = add_ts_scaling_old_values(
            'pricebuy', aggregated_input_dict)
        # Bounds do not need to be updated, as the TSA does not change them.

        ### 3. Do Sanity Check:
        # Do specific sanity check for pricebuy:
        sanity_check_ts_scaling_bounds(
            self.ts_scaling, aggregated_input_dict, 'pricebuy')

        return aggregated_input_dict

    def sanity_check(self):
        """ Conducts sanity check of the time series aggregation
        """
        total_hours = self.config['time_series_aggregation']['total_hours']
        if abs(
            sum(
                self.time_slice_duration[i] * self.time_slice_weight[i]
                for i in  self.time_slice_weight
                ) - total_hours
            ) < 1e-10:
            print("\tSanity check success")

        else:
            print("\tSanity check failed")
            sum_ist = sum(
                self.time_slice_duration[i] * self.time_slice_weight[i]
                for i in  self.time_slice_weight
            )
            sum_soll = total_hours
            raise ValueError(
                f"Sanity check faile, timeslice_duration * time_slice_weight ({sum_ist}) "+
                f"does not add up to total_hours ({sum_soll})"
                )

    def export_tsa_to_yaml(self, custom_export_path=None):
        """ Exports the aggregated input_dict to a yaml files.
        First, the timeseries from the aggregated input_dict are exported to a yaml file.
        Then, the time_slices, time_slice_weight, time_slice_duration and next_time_slice are
        exported to yaml files. The project file is also exported to a yaml file, containing the
        names of the exported files.

        Args:
            custom_export_path (str, optional): path to the directory where the yaml files are
                exported to. Overwrites default location. Defaults to None
        """

        if custom_export_path is None: # If no custom path is given, use the default one
            export_path = self.default_tsa_export_path
        else: # Else, use the custom one:
            # Create folder for tsa data:
            print(f"\tUse custom tsa_export path {custom_export_path}")
            export_path = custom_export_path

        os.makedirs(export_path, exist_ok=True)

        # Create_tsa_export_project:
        tsa_export_project = self._create_tsa_export_project(export_path=export_path)

        # Add all timeseries found in self.timeseries_original
        input_dict_keys = {
            # category in timeseries.yaml : key in input_dict,
            'demand'                      : 'demand',
            'availability'                : 'production_availability',
            'pricebuy'                    : 'pricebuy',
            'pricesell'                   : 'pricesell',
            'temperature'                 : 'temperature',
            'availability_interpolation'  : 'availability_interpolation_2nd_ts'
        }
        for ts_dict in self.timeseries_original.values():
            self._export_single_timeseries(
                ts_info_dict    = ts_dict,
                input_dict_key  = input_dict_keys[ts_dict['categories'][0]],
                export_path     = export_path
            )
        # Export self.timeseries_orginal in export path:
        with open(os.path.join(export_path, tsa_export_project['timeseries']),
                  'w', encoding = 'utf-8') as f:
            yaml.dump(self.timeseries_original, f)

        # Export time_slices:
        with open(os.path.join(export_path, tsa_export_project['time_slices']),
            'w', encoding = 'utf-8') as f:
            yaml.dump(self.aggregated_input_dict['time_slices'], f)

        # Export time_slice_weight:
        with open(os.path.join(export_path, tsa_export_project['time_slice_weight']),
            'w', encoding = 'utf-8') as f:
            # Convert values to float:
            dict_out = {
                key: float(val)
                for key, val in self.aggregated_input_dict['time_slice_weight'].items()
            }
            yaml.dump(dict_out, f)

        # Export time_slice_duration:
        with open(os.path.join(export_path, tsa_export_project['time_slice_duration']),
            'w', encoding = 'utf-8') as f:
            # Convert values to float:
            dict_out = {
                key: float(val)
                for key, val in self.aggregated_input_dict['time_slice_duration'].items()
            }
            yaml.dump(dict_out, f)

        # Export next_time_slice:
        with open(os.path.join(export_path, tsa_export_project['next_time_slice']),
            'w', encoding = 'utf-8') as f:
            # Convert values to int:
            dict_out = {
                key: int(val)
                for key, val in self.aggregated_input_dict['next_time_slice'].items()
            }
            yaml.dump(dict_out, f)

        # Export tsa_config:
        with open(os.path.join(export_path, tsa_export_project['tsa_config']),
            'w', encoding = 'utf-8') as f:
            dict_out = {
                key: val for key, val in self.config['time_series_aggregation'].items()
                if key not in ['tsa_path', 'load_tsa_inputs']
            }
            yaml.dump(dict_out, f)

        print(f"\tExported tsa inputs to {export_path}.")

    def _export_single_timeseries(self, ts_info_dict:dict, input_dict_key:str, export_path:str):
        """ Exports a single timeseries to a csv file.

        Args:
            ts_info_dict (dict): contains the information about the timeseries
                (file, doc, product/process, categories, unit)
            input_dict_key (str): key of in aggregated_input_dict, where the timeseries is stored
            export_path (str): path to the directory where the timeseries should be exported to.

        Raises:
            NotImplementedError: if multiple categories are used for one entry.
        """
        # Raise error, if multiple categories are used for one entry, since its not implemented yet:
        if len(ts_info_dict['categories']) > 1:
            raise NotImplementedError(
                "export_single_timeseries() is currently only working for timeseries entries " +
                "belonging to a single category.")

        product_or_process = ts_info_dict.get('product') or ts_info_dict.get('process')
        # Get Product or Process from ts_info_dict
        if product_or_process:
            # Get aggregated data of the current timeseries:
            values = convert_to_dataframe(
                input_subdict = self.aggregated_input_dict[input_dict_key],
                products_or_processes= [product_or_process],
                category   = ts_info_dict['categories'][0] # Assumes that there is just one category
                ).values

        # Used for temperatures timeseries (where no product or process is referenced)
        else:
            values = pd.DataFrame(
                [[v] for v in self.aggregated_input_dict[input_dict_key].values()]
                ).values

        df_out = pd.DataFrame({'data': values.T[0]})

        # Export DataFrame:
        df_out.to_csv(
            os.path.join(export_path, ts_info_dict['file']),
            index=False
        )

    def _create_tsa_export_project(self, export_path):
        """ Creates a project file for the tsa export.

        Args:
            export_path (str): path to the directory where the yaml files are exported to.

        Returns:
            dict: project dictionary containing the names of the exported files.
        """
        tsa_export_project = {
            'timeseries'          : 'time_series.yaml',
            'time_slices'         : 'time_slices.yaml',
            'time_slice_weight'   : 'time_slice_weight.yaml',
            'time_slice_duration' : 'time_slice_duration.yaml',
            'next_time_slice'     : 'next_time_slice.yaml',
            'tsa_config'          : 'tsa_config.yaml'
        }
        # Save project file:
        with open(os.path.join(export_path, 'project_tsa_export.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(tsa_export_project, f)

        return tsa_export_project

    def _get_tuples_to_consider_with_extreme_peroid(self):
        """ Gets list of demands and prices which should be included for extreme peroids in the tsa.
        if a demand should be inclused is deginded in products.yaml

        Returns:
            list: list of tuples containing the product and the category 'demand' or 'pricebuy'
        """
        # Get products where the extreme periods are relevant for the demand
        products_demand = [
            product for product, value in self.products.items()
            if value.get('include_extreme_demand_peroid', False)
        ]
        product_demand_list = [(product, 'demand') for product in products_demand]

        # Get products where the extreme periods are relevant for pricebuy
        products_pricebuy = [
            product for product, value in self.products.items()
            if value.get('include_extreme_pricebuy_peroid', False)
        ]
        products_pricebuy_list = [(product, 'pricebuy') for product in products_pricebuy]

        # Return list, e.g. [('heat', 'demand'), ('electricity', 'pricebuy')]
        return product_demand_list + products_pricebuy_list

class InputVisualization():
    """ Class to visualize time series from input data"""
    def __init__(self, data_in_path:str, project_file:str, export_path:str = None,
                 aggregated_timeseries_path:str=None):
        """ Plots the input timeseries

        Args:
            data_in_path (str): _description_
            export_path (str, optional): Path to the plot export. Defaults to None.
            aggregated_timeseries_path (str, optional): Path where the aggregated data lies.
                If no path is given, no aggregated data is shown. Defaults to None.
        """

        self.data_in_path = data_in_path # data_in_path is the path to the Input_Data directory
        assert os.path.exists(self.data_in_path)

        if aggregated_timeseries_path is None:
            self.data_path_timeseries = self.data_in_path
        else:
            self.data_path_timeseries = aggregated_timeseries_path

        if export_path is None:
            # give standard export path to avoid crashes on LTT Working stations
            bplt.output_file(os.path.join(
                 get_data_out(self.data_in_path, project_file), 'time_series_visualization.html'))
        self.export_path = export_path

        self._load_all_time_series()

        self.grouped_data = {}

    def _load_all_time_series(self):
        """ Load all the data from time_series.yaml """
        with open(os.path.join(self.data_path_timeseries, 'time_series.yaml'),
                  'r', encoding='utf-8') as f:

            self.all_time_series = yaml.safe_load(f)

    def _load_individual_time_series(self, path_to_csv):
        df = pd.read_csv(path_to_csv)
        return {'values': df.iloc[:, 0].values} # Make the format suitable for plotting with Bokeh

    def _group_data_by_category(self):
        """Groups time series that belong to the same category together
        """

        # Loop over all time_series
        for time_series, metadata in self.all_time_series.items():
            category = metadata['categories'][0] # Assuming each time series only has 1 category

            if category not in self.grouped_data:
                # Create a dictionary for each category inside self.grouped_data:
                self.grouped_data[category] = {'time_series': {}, 'statistics': None}

            path_to_csv = os.path.join(self.data_path_timeseries, metadata['file'])

             # Loading the data for each time series:
            self.grouped_data[
                category
                ]['time_series'][time_series] = self._load_individual_time_series(path_to_csv)

    def _calculate_statistics(self, time_series_data:dict):
        """ Calculates the 'mean', 'max' and 'min' for a given tiemseries.

        Args:
            time_series_data (dict): _description_

        Returns:
            dict: contains 'mean', 'max' and 'min' for the timeseries
        """

        values = time_series_data['values']

        return {
            'mean': np.mean(values),
            'max': np.max(values),
            'min': np.min(values)
        }

    def _generate_statistics_table(self, category:str):
        """ Calculate the statistics of all time series within a category """

        time_series_data = self.grouped_data[category]['time_series']
        statistics = pd.DataFrame(
            columns = ['Time Series', 'Mean', 'Maximum Value', 'Minimum Value']
        )

        for time_series_name, time_series_data in time_series_data.items():
            stats = self._calculate_statistics(time_series_data)
            # Append to the statistics of the current time series to the statistics of the category
            statistics = pd.concat([statistics, pd.DataFrame({
                'Time Series': [time_series_name],
                'Mean': [stats['mean']],
                'Maximum Value': [stats['max']],
                'Minimum Value': [stats['min']]
            })])

        self.grouped_data[category]['statistics'] = statistics

    def visualize_unaggregated_input(self):
        """ Main function to plot unaggregated input timeseries"""
        self._group_data_by_category()

        plots = []
        tables_list = []

        for category, data in self.grouped_data.items():
            # Generate statistics table for the category
            self._generate_statistics_table(category)

            # Get the unit of the first time series
            unit = self.all_time_series[next(iter(data['time_series']))].get('unit')

            plot_width = 1000  # Adjust the width of the plot

            plot = bplt.figure(title = f'Visualization for {category}', x_axis_label = 'TimeStep',
                        y_axis_label = f'{category} / {unit}', width = plot_width)

            colors = itertools.cycle(Category10_10)

            for time_series_name, time_series_data in data['time_series'].items():
                x = np.arange(len(time_series_data['values']))
                y = time_series_data['values']
                color = next(colors)

                plot.line(x = x, y = y, legend_label = time_series_name, color=color)

            plot.legend.location = 'top_left'
            plot.legend.click_policy = 'hide'

            plots.append(plot)  # Append the plot for the current category

            # Create DataTable widget for statistics table
            source = ColumnDataSource(data['statistics'])
            columns = [TableColumn(field = col, title = col) for col in data['statistics'].columns]
            table = DataTable(source = source, columns = columns, width = 600, height = 280)
            tables_list.append(table)

        # Combine plots and tables into a layout
        combined_plots_tables = []
        for plot, table in zip(plots, tables_list):
            combined_plots_tables.extend([plot, table])

        layout = column(combined_plots_tables)

        if self.export_path:  # Check if export path is provided
            bplt.output_file(os.path.join(self.export_path, 'time_series_visualization.html'))
            bplt.save(layout)  # Save the HTML file to the export path

        else:
            bplt.show(layout)  # Otherwise, just show the plot

class AggregatedInputVisualization():
    """ Plots the aggregated timeseries
    """
    def __init__(self, aggregated_input_dict):
        self.aggregated_input_dict = aggregated_input_dict

        # Setting the configurations for plotting
        self.plot_width = 1000
        self.line_width = 2
        self.bar_width = 0.5
        self.colors = itertools.cycle(Category20[20])
        self.table_width = 400
        self.table_height = 280

    def _plot_time_series(self, aggregated_dataframe):
        """generates line plot for an aggregated datarame:
        """
        category = aggregated_dataframe.columns[0][1]

        plot = bplt.figure(
            title        = f'Visualization for aggregated {category}',
            x_axis_label = 'Time Steps',
            y_axis_label = 'Values',
            width        = self.plot_width)

        for col, _ in zip(aggregated_dataframe.columns, self.colors):
            plot.line(
                aggregated_dataframe.index,
                aggregated_dataframe[col],
                legend_label = col[0],
                color        = next(self.colors),
                line_width   = self.line_width)

        plot.legend.click_policy = 'hide'

        return plot

    def visualize_aggregated_input(self, export_path:str=None):
        """ Main function to plot aggregated input timeseries. """

        plots = []
        tables = []

        dataframes = [
            convert_to_dataframe( #aggregated_production_availability_df
                self.aggregated_input_dict['production_availability'],
                self.aggregated_input_dict['processes_production'],
                'production_availability'
                ),
            convert_to_dataframe( # aggregated_demand_df
                self.aggregated_input_dict['demand'],
                self.aggregated_input_dict['products'],
                'demand'
                ),
            convert_to_dataframe( # aggregated_pricebuy_df
                self.aggregated_input_dict['pricebuy'],
                self.aggregated_input_dict['products'],
                'pricebuy'
                ),
            convert_to_dataframe( # aggregated_pricesell_df
                self.aggregated_input_dict['pricesell'],
                self.aggregated_input_dict['products'],
                'pricesell'
                ),
            pd.DataFrame( #aggregated_temperature_df
                {('temperature', 'temperature'): [
                    value for key, value in self.aggregated_input_dict['temperature'].items()
                    if key[0] == 'temperature']
                    })
        ]

        # Visualize time series
        for dataframe in dataframes:

            plot = self._plot_time_series(dataframe)
            plots.append(plot)

            # Calculate mean, maximum, and minimum values for each column
            mean_values = dataframe.mean()
            max_values = dataframe.max()
            min_values = dataframe.min()

            # Create a DataFrame to store the statistics
            statistics_df = pd.DataFrame({
                'Time Series': [col[0] for col in dataframe.columns],
                'Mean': mean_values,
                'Maximum Value': max_values,
                'Minimum Value': min_values
            })

            source = ColumnDataSource(statistics_df)

            # Create a DataTable to display the statistics
            columns = [
                TableColumn(field = column, title = column) for column in statistics_df.columns
                ]
            table = DataTable(
                columns = columns,
                source  = source,
                width   = self.table_width,
                height  = self.table_height)
            tables.append(table)

        # Create statistic table for time_slice_weight
        time_slice_weight_mean = sum(
            self.aggregated_input_dict['time_slice_weight'].values()
            ) / len(self.aggregated_input_dict['time_slice_weight'])
        time_slice_weight_max = max(self.aggregated_input_dict['time_slice_weight'].values())
        time_slice_weight_min = min(self.aggregated_input_dict['time_slice_weight'].values())

        time_slice_weight_stats_df = pd.DataFrame({
            'Statistic'     : 'time_slice_weight',
            'Mean'          : [time_slice_weight_mean],
            'Maximum Value' : [time_slice_weight_max],
            'Minimum Value' : [time_slice_weight_min]
        })

        time_slice_weight_source = ColumnDataSource(time_slice_weight_stats_df)

        time_slice_weight_columns = [
            TableColumn(field = column, title = column)
            for column in time_slice_weight_stats_df.columns]
        time_slice_weight_table = DataTable(
            columns = time_slice_weight_columns,
            source  = time_slice_weight_source,
            width   = self.table_width,
            height  = self.table_height
            )
        tables.append(time_slice_weight_table)

        # Create statistic table for time_slice_duration
        time_slice_duration_mean = sum(
            self.aggregated_input_dict['time_slice_duration'].values()
            ) / len(self.aggregated_input_dict['time_slice_duration'])
        time_slice_duration_max = max(self.aggregated_input_dict['time_slice_duration'].values())
        time_slice_duration_min = min(self.aggregated_input_dict['time_slice_duration'].values())

        time_slice_duration_stats_df = pd.DataFrame({
            'Statistic'     : 'time_slice_duration',
            'Mean'          : [time_slice_duration_mean],
            'Maximum Value' : [time_slice_duration_max],
            'Minimum Value' : [time_slice_duration_min]
        })

        time_slice_duration_source = ColumnDataSource(time_slice_duration_stats_df)

        time_slice_duration_columns = [
            TableColumn(field = column, title = column)
            for column in time_slice_duration_stats_df.columns]
        time_slice_duration_table = DataTable(
            columns = time_slice_duration_columns,
            source  = time_slice_duration_source,
            width   = self.table_width,
            height  = self.table_height
            )
        tables.append(time_slice_duration_table)

        # Visualize time_slice_weight and time_slice_duration
        # Plot time_slice_weight as a line plot
        plot_time_slice_weight = bplt.figure(
            title        = 'Visualization for aggregated time_slice_weight',
            x_axis_label = 'Time Steps',
            y_axis_label = 'Weight',
            width        = self.plot_width)
        plot_time_slice_weight.line(
            list( range( len(self.aggregated_input_dict['time_slice_weight']) ) ),
            list(self.aggregated_input_dict['time_slice_weight'].values()),
            line_width = self.line_width)
        plots.append(plot_time_slice_weight)

        # Plot time_slice_duration as a line plot
        plot_time_slice_duration = bplt.figure(
            title        = 'Visualization for aggregated time_slice_duration',
            x_axis_label = 'Time Steps',
            y_axis_label = 'Duration',
            width        = self.plot_width
            )
        plot_time_slice_duration.line(
            list( range( len(self.aggregated_input_dict['time_slice_duration']) ) ),
            list(self.aggregated_input_dict['time_slice_duration'].values()),
            line_width = self.line_width
            )
        plots.append(plot_time_slice_duration)

        # Combine plots and tables into a layout
        combined_plots_tables = []
        for plot, table in zip(plots, tables):
            combined_plots_tables.extend([plot, table])

        layout = column(combined_plots_tables)

        if export_path is not None:  # Check if export path is provided
            bplt.output_file(os.path.join(export_path, 'aggregated_input_visualization.html'))
            bplt.save(layout)  # Save the HTML file to the export path

        else:
            bplt.show(layout)  # Otherwise, just show the plot

def convert_to_dataframe(input_subdict:dict, products_or_processes:list, category:str):
    """ Helper function for AggregatedInputDictGenerator and AggregatedInputVisualization.
    Generates a dictionary from Pyomo tuple format into a pd.DataFrame.

    Args:
        input_subdict (dict): subdict of input_dict
            generated by UnscaledInputDictGenerator
        products_or_processes (list): list of products or processes
            e.g., ['heat', 'cold'] or ['PV', 'CHP']
        category (str): name of subdict
            e.g, 'demand'

    Returns:
        pd.DataFrame: dict converted into a DataFrame
    """

    product_or_process_data = {
        (product, category): [] for product in products_or_processes
    }

    for key, value in input_subdict.items():
        name = key[0]
        if name in products_or_processes:
            product_or_process_data[(name, category)].append(value)

    return pd.DataFrame(product_or_process_data)

def calculate_solar_collector_availability(
        data_in_path, project_file, solar_irradiation:np.array, temp_ambient:np.array):
    """
    Calculate the specific heat output v_collector(t) for a solar thermal system.

    Parameters (Array):
    - solar_irradiation (np.array): Solar irradiation (G) values at time t (W/m^2).
    - temp_ambient (np.array): Ambient temperatures at time t (°C).

    Parameters (Float), are read from technologies.yaml
    - T_collector: Collector (system) temperature (°C).
    - eta_opt_collector: Optical efficiency of the solar thermal collector
    (dimensionless, e.g., 0.8).
    - k1: Linear heat loss coefficient (W / (K * m^2)).
    - k2: Quadratic heat loss coefficient (W / (K^2 * m^2)).
    - a_collector: Area required per unit installed capacity (m^2 / kW).

    Returns:
    - v_collector: numpy.ndarray
        Specific heat output at time t (dimensionless, kW_heat_output / kW_installed_capacity).
    """

    # Load project file:
    with open(os.path.join(data_in_path, project_file), 'r', encoding='utf-8') as f:
        project = yaml.load(f, Loader=UniqueKeyLoader)
    # Open and read the Technology yaml file
    with open(os.path.join(data_in_path,  project['technologies']), 'r', encoding='utf-8') as f:
        technologies = yaml.load(f, Loader=UniqueKeyLoader)['technologies']

    # Extract the Solar Collector Parameters
    technologies_sc = technologies['Solar_collector']
    k1 = technologies_sc['k1']
    k2 = technologies_sc['k2']
    a_collector = technologies_sc['a_collector']
    eta_opt_collector = technologies_sc['eta_opt_collector']
    temp_collector = technologies_sc['T_collector']

    # Ensure inputs are numpy arrays for element-wise operations
    solar_irradiation = np.array(solar_irradiation)
    temp_ambient = np.array(temp_ambient)

    # Calculate the temperature difference between the collector and ambient
    delta_temp = temp_collector - temp_ambient

    # Calculate the net heat gain per unit area (kW/m^2)
    q = (solar_irradiation * eta_opt_collector - k1 * delta_temp - k2 * delta_temp ** 2) / 1000

    # Compute specific heat output per unit installed capacity (kW/kW_installed)
    v_collector = q / a_collector

    # Ensure that v_collector is not negative, as negative heat output is not physical
    v_collector = np.maximum(v_collector, 0)

    # Export the data to the relevant csv File
    df_v_collector = pd.DataFrame(v_collector, columns=['data'])
    df_v_collector.to_csv(
        os.path.join(data_in_path,  'ts_availability_solar_collector.csv'),
        index=False
        )

    return v_collector
