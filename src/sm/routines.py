from pathlib import Path
import os
import time
import pickle
import pandas as pd
import numpy as np
from core.postprocessing import (get_variable_sum, get_variable_sum_for_product,
                                 get_full_load_hours, PostProcessor)
from core.preprocessing import UnscaledInputDictGenerator, AggregatedInputDictGenerator
from core.helpers import (get_data_out, get_process_names, get_config_file, get_input_file,
                          get_data_computed_in, get_value_of_lost_load,
                          sanity_check_value_of_lost_load)
from core.ts_scaling import get_ts_scaling_names
from sm.surrogate_model import SurrogateModelNN
from sm.sado import (SADOMachineLearning, SADOConexHull, CostPredictorNN, FeasibilityPredictorNN,
                     FeasibilityPredictorNoFP)
from sm.surrogates_feasible_space import Feasibility_LP
from sm.model_manipulation import (generate_initial_model, EnergySystemDesignConverter,
                                 mutate_nominal_capacities, TimeSeriesScaler)
from sm.feasible_space_expl import Convex_Hull_Explorer

def calculate_OPEX_for_batch(df_in:pd.DataFrame, idx_start:int, batch_size:int,
                             data_in_path:str, project_file:str, output_path:str) -> None:
    """
    Calculate OPEX and lost load for a batch of designs and save results to a CSV file.

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        idx_start (int): Starting index of the batch.
        batch_size (int): Size of the batch to process. (e.g. 2000)
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        output_path (str): Path where the output file will be saved.
    
    Raises:
        ValueError: If `idx_start` is not a non-negative integer or `batch_size` is not a positive integer.
    """

    if not isinstance(idx_start, int) or idx_start < 0:
        raise ValueError("idx_start must be a non-negative integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Copy df_in from idx_start to idx_start+batch_size
    design_df_batch = df_in.iloc[idx_start:idx_start+batch_size].copy()

    # Generate blank model:
    optimization, blank_model = generate_initial_model(data_in_path, project_file)

    # Initialize an esd_converter
    esd_converter = EnergySystemDesignConverter(data_in_path, project_file)

    # Pre-allocate arrays for OPEX, lost_load, and solver_time
    OPEX        = np.zeros(len(design_df_batch))
    lost_load   = np.zeros(len(design_df_batch))
    solver_time = np.zeros(len(design_df_batch))
    # Pre-allocate array for product-wise lost_load
    lost_load_products = np.zeros([len(design_df_batch), len(blank_model.products)])

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx, (_, design_series) in enumerate(design_df_batch.iterrows()):
        print(f"idx {idx_start + idx}")
        # Convert the design_series to a design_dict:
        custom_design = esd_converter.get_design_dict_from_series(design_series)

        # Mutate the nominal capacities:
        model = mutate_nominal_capacities(blank_model, custom_design)

        # Solve the model:
        result = optimization.solve_model(model, tee=False)

        # Store OPEX, lost_load, and solver_time values
        OPEX[idx]        = get_variable_sum(model, model.OPEX)
        lost_load[idx]   = get_variable_sum(model, model.Products_lost_load)
        solver_time[idx] = result['Solver'][0]['wall_time']

        # Store product-wise lost_load
        for i, product in enumerate(list(blank_model.products)):
            lost_load_products[idx, i] = get_variable_sum_for_product(
                model,
                model.Products_lost_load,
                product)

    # Add value to the DataFrame:
    design_df_batch['OPEX']      = OPEX
    design_df_batch['lost_load'] = lost_load
    for i, product in enumerate(list(blank_model.products)):
        design_df_batch[f'lost_load_{product}'] = lost_load_products[:, i]

    design_df_batch['solver_time'] = solver_time

    # Make output folder if necessary
    Path(output_path).with_suffix('').mkdir(parents=True, exist_ok=True)

    # Save design_df:
    output_file = Path(output_path).with_suffix('') / f"idx{idx_start}-{idx_start + len(design_df_batch) - 1}.csv"
    print(f"Saved batch to {output_file} .")
    design_df_batch.to_csv(output_file, index=True)

def generate_aggregated_input_data(data_in_path:str, project_file:str) -> dict:
    """ aggregates the input data for the specified data path and project file

    Args:
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.

    Returns:
        dict: aggregated input dict
    """
    print(44*'#'+"\n##### Generating aggregated input data #####\n"+44*'#')
    # generate aggregated inputs
    unscaled_input_dict_old = UnscaledInputDictGenerator(
        data_in_path = data_in_path,
        project_file = project_file,
        objective    = 'design',
        ignore_tsa   = True
    ).generate_unscaled_input_dict()

    # 2. aggregate the input dict and export the timeseries data
    aggregator = AggregatedInputDictGenerator(
        data_in_path = data_in_path,
        project_file = project_file,
        unaggregated_input_dict= unscaled_input_dict_old
    )
    aggregated_input_dict = aggregator.generate_aggregated_input_dict()
    aggregator.export_tsa_to_yaml()
    print(50*'#'+"\n##### Done: Generating aggregated input data #####\n"+50*'#')

    return aggregated_input_dict

def design_optimization_var_ts(
        df_in:pd.DataFrame, data_in_path, project_file:str, timeseries_ignored:list = None,
        debug:bool = False) -> pd.DataFrame:
    """
    Calculate optimal nominal capacities, TAC, CAPEX, OPEX, lost_load, and full_load_hours
    for a DataFrame of conditions (timeseries) and return result DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        timeseries_ignored (list): List of timeseries, which will be ignored.
            Can be taken from SampleGenerator.timeseries_ignored
        debug (bool): If True: gurobi log is printed, output dicts and result plots are saved
    """

    if timeseries_ignored is None:
        timeseries_ignored = []

    # Generate blank model:
    print("Generating blank model...")
    optimization, blank_model = generate_initial_model(
        data_in_path,
        project_file,
        objective='design'
    )
    print("\tDone.")

    # Initialize a timeseries scaler
    ts_scaler = TimeSeriesScaler(
        data_in_path,
        project_file,
        blank_model=blank_model,
        timeseries_ignored=timeseries_ignored
    )

    res = {
        # Pre-allocate arrays for TAC, CAPEX, OPEX, lost_load, and solver_time
        'OPEX'        : np.zeros(len(df_in)),
        'CAPEX'       : np.zeros(len(df_in)),
        'TAC'         : np.zeros(len(df_in)),
        'lost_load'   : np.zeros(len(df_in)),
        'solver_time' : np.zeros(len(df_in)),

        # Pre-allocate array for product-wise lost_load
        'lost_load_products'        : np.zeros([len(df_in), len(blank_model.products)]),
        # Pre-allocate array for production process-wise full_load hours
        'full_load_hour_production' : np.zeros([len(df_in), len(blank_model.processes_production)]),
        # Pre-allocate array for production capacity
        'production_capacity'       : np.zeros([len(df_in), len(blank_model.processes_production)]),
        # Pre-allocate array for storage capacity
        'storage_capacity'          : np.zeros([len(df_in), len(blank_model.processes_storage)]),
        # Pre-allocate empty list for termination condition:
        'termination_condition'     : []
    }

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx, (_, design_and_ts_series) in enumerate(df_in.iterrows()):

        # 1. Mutate time series:
        model = ts_scaler.scale_ts(blank_model, design_and_ts_series)
        # 2. Solve model:
        result = optimization.solve_model(model, tee=debug, symbolic_solver_labels=debug)

        # In debugging mode: Save outputdict and results plots
        if debug:
            filename = f"design_var_ts_id{str(int(design_and_ts_series['id'])).zfill(7)}"
            save_result_plot_and_outputdict(
                data_in_path, project_file, model, result, filename)

        # if model is infeasible, print warning and continue:
        if result['Solver'][0]['Termination condition'].value == 'infeasible':
            print("Model infeasible")
            continue

        # Store OPEX, lost_load, and solver_time values
        res['OPEX'][idx]       = get_variable_sum(model, model.OPEX)
        res['CAPEX'][idx]       = get_variable_sum(model, model.CAPEX)
        res['TAC'][idx]         = model.TAC[None].value
        res['lost_load'][idx]   = get_variable_sum(model, model.Products_lost_load)
        res['solver_time'][idx] = result['Solver'][0]['wall_time']

        # Store product-wise lost_load
        for i, product in enumerate(list(blank_model.products)):
            res['lost_load_products'][idx, i] = get_variable_sum_for_product(
                model,
                model.Products_lost_load,
                product)

        # Store production process-wise full_load_hours and nominal_capacities:
        for i, process in enumerate(list(blank_model.processes_production)):
            res['full_load_hour_production'][idx, i] = get_full_load_hours(model, process)
            res['production_capacity'][idx, i] = model.Production_nominal_capacity[process].value

        # Store storage process-wise nominal_capacities:
        for i, process in enumerate(list(blank_model.processes_storage)):
            res['storage_capacity'][idx, i] =model.Storage_nominal_capacity[process].value

        # Store termination condition:
        res['termination_condition'].append(result['Solver'][0]['Termination condition'].value)

    # Add value to the DataFrame:
    df_in['OPEX']      = res['OPEX']
    df_in['CAPEX']     = res['CAPEX']
    df_in['TAC']       = res['TAC']
    df_in['lost_load'] = res['lost_load']
    # Add Lost Load
    for i, product in enumerate(list(blank_model.products)):
        df_in[f'lost_load_{product}'] = res['lost_load_products'][:, i]
    # Add Production nominal capacity
    for i, process in enumerate(list(blank_model.processes_production)):
        df_in[process] = res['production_capacity'][:, i]
    # Add Storage: nominal capacity
    for i, process in enumerate(list(blank_model.processes_storage)):
        df_in[process] = res['storage_capacity'][:, i]
    # Add Production full_load_hours
    for i, process in enumerate(list(blank_model.processes_production)):
        df_in[f'full_load_hours_{process}'] = res['full_load_hour_production'][:, i]

    df_in['solver_time'] = res['solver_time']
    df_in['termination_condition'] = res['termination_condition']

    return df_in

def operational_optimization_const_ts(
        df_in:pd.DataFrame, data_in_path:str, project_file:str, optimization_type:str = 'operation',
        debug:bool = False) -> pd.DataFrame:
    """
    Calculate TAC, CAPEX, OPEX, lost_load, and full_load_hours for a DataFrame of
    designs and return result DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        optimization_type (str): 'operation' or 'max_overcapacity'
        debug (bool): If True: gurobi log is printed, output dicts and result plots are saved.
    """

    assert optimization_type in {'operation', 'max_overcapacity'}

    # Generate blank model:
    print("Generating blank model...")
    optimization, blank_model = generate_initial_model(
        data_in_path,
        project_file,
        objective=optimization_type
    )
    print("\tDone.")

    # Initialize an esd_converter
    esd_converter = EnergySystemDesignConverter(data_in_path, project_file)

    res = {
        # Pre-allocate arrays for OPEX, TAC, CAPEX, lost_load, and solver_time
        'opex'                      : np.zeros(len(df_in)),
        'tac'                       : np.zeros(len(df_in)),
        'capex'                     : np.zeros(len(df_in)),
        'lost_load'                 : np.zeros(len(df_in)),
        'solver_time'               : np.zeros(len(df_in)),
        'overcapacity'              : np.zeros(len(df_in)),

        # Pre-allocate array for product-wise lost_load
        'lost_load_products'        : np.zeros([len(df_in), len(blank_model.products)]),
        # Pre-allocate array for production process-wise full_load hours
        'full_load_hour_production' : np.zeros([len(df_in), len(blank_model.processes_production)]),
        # Pre-allocate empty list for termination condition:
        'termination_condition'     : []
    }

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx, (_, design_series) in enumerate(df_in.iterrows()):

        # 1. Mutate nominal capacities:
        custom_design = esd_converter.get_design_dict_from_series(design_series)
        model = mutate_nominal_capacities(blank_model, custom_design)

        # 2. Solve the model:
        result = optimization.solve_model(model, tee=debug, symbolic_solver_labels=debug)

        # In debugging mode: Save outputdict and results plots
        if debug:
            filename = f"operation_const_ts_id{str(int(design_series['id'])).zfill(7)}"
            save_result_plot_and_outputdict(
                data_in_path, project_file, model, result, filename)

        # if model is infeasible, print warning and continue:
        if result['Solver'][0]['Termination condition'].value == 'infeasible':
            print("Model infeasible")
            continue

        # Store OPEX, lost_load, and solver_time values
        res['opex'][idx]         = get_variable_sum(model, model.OPEX)
        res['capex'][idx]        = get_variable_sum(model, model.CAPEX)
        res['tac'][idx]          = model.TAC[None].value
        res['lost_load'][idx]    = get_variable_sum(model, model.Products_lost_load)
        res['solver_time'][idx]  = result['Solver'][0]['wall_time']
        res['overcapacity'][idx] = model.Overcapacity[None].value

        # Store product-wise lost_load
        for i, product in enumerate(list(blank_model.products)):
            res['lost_load_products[idx, i]'] = get_variable_sum_for_product(
                model,
                model.Products_lost_load,
                product)

        # Store production process-wise full load hours:
        for i, process in enumerate(list(blank_model.processes_production)):
            res['full_load_hour_production'][idx, i] = get_full_load_hours(model, process)

        # Store termination condition:
        res['termination_condition'].append(result['Solver'][0]['Termination condition'].value)

    # Add value to the DataFrame:
    df_in['OPEX']         = res['opex']
    df_in['CAPEX']        = res['capex']
    df_in['TAC']          = res['tac']
    df_in['lost_load']    = res['lost_load']
    df_in['overcapacity'] = res['overcapacity']

    for i, product in enumerate(list(blank_model.products)):
        df_in[f'lost_load_{product}'] = res['lost_load_products'][:, i]
    for i, process in enumerate(list(blank_model.processes_production)):
        df_in[f'full_load_hours_{process}'] = res['full_load_hour_production'][:, i]

    df_in['solver_time'] = res['solver_time']
    df_in['termination_condition'] = res['termination_condition']

    return df_in

def operational_optimization_var_ts(
        df_in:pd.DataFrame, data_in_path, project_file:str, optimization_type:str = 'operation',
        timeseries_ignored:list = None, debug:bool = False) -> pd.DataFrame:
    """
    Calculate TAC, CAPEX, OPEX, lost_load, and full_load_hours for a DataFrame of
    designs and variable conditions (timeseries) and return result DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        data_in_path (str): Path to the data files.
        project_file (str): Project file name.
        optimization_type (str): 'operation' or 'max_overcapacity'
        timeseries_ignored (list): List of timeseries, which will be ignored.
            Can be taken from SampleGenerator.timeseries_ignored
        debug (bool): If True: gurobi log is printed, output dicts and result plots are saved.
    """

    assert optimization_type in {'operation', 'max_overcapacity'}

    if timeseries_ignored is None:
        timeseries_ignored = []

    # Generate blank model:
    print("Generating blank model...")
    optimization, blank_model = generate_initial_model(
        data_in_path,
        project_file,
        objective= optimization_type
    )
    print("\tDone.")

    # Initialize an esd_converter
    esd_converter = EnergySystemDesignConverter(data_in_path, project_file)

    # Initialize a timeseries scaler
    ts_scaler = TimeSeriesScaler(
        data_in_path,
        project_file,
        blank_model=blank_model,
        timeseries_ignored=timeseries_ignored
    )

    # Pre-allocate arrays for OPEX, TAC, CAPEX, lost_load, and solver_time
    res = {
        'opex'                      : np.zeros(len(df_in)),
        'tac'                       : np.zeros(len(df_in)),
        'capex'                     : np.zeros(len(df_in)),
        'lost_load'                 : np.zeros(len(df_in)),
        'solver_time'               : np.zeros(len(df_in)),
        'overcapacity'              : np.zeros(len(df_in)),

        # Pre-allocate array for product-wise lost_load
        'lost_load_products'        : np.zeros([len(df_in), len(blank_model.products)]),
        # Pre-allocate array for production process-wise full_load hours
        'full_load_hour_production' : np.zeros([len(df_in), len(blank_model.processes_production)]),
        # Pre-allocate empty list for termination condition:
        'termination_condition'     : []
    }

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx, (_, design_and_ts_series) in enumerate(df_in.iterrows()):
        print(f"Sample {idx}")
        # 1. Mutate nominal capacities:
        design_series = design_and_ts_series[
            [index for index in design_and_ts_series.index if index in esd_converter.technologies ]
            ]
        custom_design = esd_converter.get_design_dict_from_series(design_series)
        model = mutate_nominal_capacities(blank_model, custom_design)

        # 2. Mutate time series:
        model = ts_scaler.scale_ts(model, design_and_ts_series)
        # 3. Solve model:
        result = optimization.solve_model(model, tee=debug, symbolic_solver_labels=debug)

        # In debugging mode: Save outputdict and results plots
        if debug:
            filename = f"operation_var_ts_id{str(int(design_and_ts_series['id'])).zfill(7)}"
            save_result_plot_and_outputdict(
                data_in_path, project_file, model, result, filename)

        # if model is infeasible, print warning and continue:
        if result['Solver'][0]['Termination condition'].value == 'infeasible':
            print("Model infeasible")
            continue

        # Store OPEX, lost_load, and solver_time values
        res['opex'][idx]         = get_variable_sum(model, model.OPEX)
        res['capex'][idx]        = get_variable_sum(model, model.CAPEX)
        res['tac'][idx]          = model.TAC[None].value
        res['lost_load'][idx]    = get_variable_sum(model, model.Products_lost_load)
        res['solver_time'][idx]  = result['Solver'][0]['wall_time']
        res['overcapacity'][idx] = model.Overcapacity[None].value

        # Store product-wise lost_load
        for i, product in enumerate(list(blank_model.products)):
            res['lost_load_products'][idx, i] = get_variable_sum_for_product(
                model,
                model.Products_lost_load,
                product)

        # Store production process-wise full load hours:
        for i, process in enumerate(list(blank_model.processes_production)):
            res['full_load_hour_production'][idx, i] = get_full_load_hours(model, process)

        # Store termination condition:
        res['termination_condition'].append(result['Solver'][0]['Termination condition'].value)

    # Add value to the DataFrame:
    df_in['OPEX']         = res['opex']
    df_in['CAPEX']        = res['capex']
    df_in['TAC']          = res['tac']
    df_in['lost_load']    = res['lost_load']
    df_in['overcapacity'] = res['overcapacity']

    for i, product in enumerate(list(blank_model.products)):
        df_in[f'lost_load_{product}'] = res['lost_load_products'][:, i]
    for i, process in enumerate(list(blank_model.processes_production)):
        df_in[f'full_load_hours_{process}'] = res['full_load_hour_production'][:, i]

    df_in['solver_time'] = res['solver_time']
    df_in['termination_condition'] = res['termination_condition']

    return df_in

def save_result_plot_and_outputdict(data_in_path, project_file:str, model, result, filename:str):
    """ Saves the result plot and output dict for a given model and result.

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)
        model: solved Pyomo model
        result: pyomo results object
        filename (str): filename for the output files without extension
    """
    postprocessor = PostProcessor(project_file, data_in_path, model, result)

    data_out = get_data_out(data_in_path, project_file)

    # Export output dict
    postprocessor.export_output_dict( Path(data_out) / f"outputdict_{filename}")

    # Export result plot:
    postprocessor.plot_results(
        export_path=Path(data_out) / f"result_plot_{filename}"
        )

def design_optimization_var_ts_hybrid_sm(
        df_in:pd.DataFrame, data_in_path, project_file:str, opex_sm_name_config: str,
        feas_sm_name_config: str, opex_sm_name:str, feas_sm_name:str, feas_sm_type:str,
        oc_min:float=0, warmstart_df:pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate optimal nominal capacities, TAC, CAPEX, OPEX, lost_load, and full_load_hours
    for a DataFrame of conditions (timeseries) and return result DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        opex_sm_name_config (str): key of the hyperparamters in config.yaml (e.g., ANN_8L)
        feas_sm_name_config (str): key of the hyperparamters in config.yaml (e.g., ANN_8L_LL)
        opex_sm_name (str): name, under which the opex predictor is saved (e.g, OPEX_NN)
        feas_sm_name (str): name, under which the feasibility predictor is saved (e.g, OC_NN)
        feas_sm_type (str): 'lost_load', 'overcapacity', or 'convex_hull'
        oc_min (float): minimal allowed overcapacity. Must be non-negative. A value != 0 is only
            allowed for self.feasibility_predictor_type = 'overcapacity' 
    """

    data_out = get_data_out(data_in_path, project_file)

    # --- Define process and timeseries sets ---
    # Get process sets
    processes = get_process_names(
        data_in_path, project_file,
        only_feasibility_affecting = False,
        storage_or_production      = 'both')
    processes_af = get_process_names( # af = affecting feasibility
        data_in_path, project_file,
        only_feasibility_affecting = True,
        storage_or_production      = 'both')

    # get timeseries scaling sets:
    ts_scalings = get_ts_scaling_names(
        data_in_path, project_file,
        only_feasibility_affecting = False)
    ts_scalings_af = get_ts_scaling_names(
        data_in_path, project_file,
        only_feasibility_affecting = True)

    # --- Pre-allocate arrays and lists for results
    res = {
        'TAC'              : np.zeros(len(df_in)),
        'OPEX_pred'        : np.zeros(len(df_in)),
        'CAPEX'            : np.zeros(len(df_in)),
        'feasibility_pred' : np.zeros(len(df_in)),
        'capacities'       : np.zeros([len(df_in), len(processes)]),
        'n_iter'           : np.zeros(len(df_in)),
        'message'          : [],
        'success'          : [],
        'constr_violation' : np.zeros(len(df_in)),
        'execution_time'   : np.zeros(len(df_in)),
    }

    # --- Load surrogate models ---
    # Load surrogate model for opex:
    sm_opex = SurrogateModelNN(
        data_in_path,
        project_file,
        input_names = ts_scalings + processes, # All ts_scalings and processes are used her
        output_name = 'OPEX_woLL',
        sm_name     = opex_sm_name_config)

    sm_opex.setup_sm()
    sm_opex.load_model(
        model_path = os.path.join(data_out, opex_sm_name), filename = opex_sm_name, load_history=False)

    # Load surrogate model for feasibility:
    sm_feas = SurrogateModelNN(
        data_in_path,
        project_file,
         # Only ts_scalings and processes are used which affect feasibility:
        input_names = ts_scalings_af + processes_af,
        output_name = feas_sm_type,
        sm_name     = feas_sm_name_config)

    sm_feas.setup_sm()
    sm_feas.load_model(
        model_path = os.path.join(data_out, feas_sm_name), filename = feas_sm_name, load_history=False)

    cost_predictor = CostPredictorNN(data_in_path, project_file, sm_opex)
    feas_predictor = FeasibilityPredictorNN(data_in_path, project_file, sm_feas, feas_sm_type,
                                            cost_predictor.processes, cost_predictor.timeseries)
    sado = SADOMachineLearning(cost_predictor, feas_predictor)

    config = get_config_file(data_in_path, project_file)

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx in range(len(df_in)):
        row_df = df_in.iloc[[idx]].copy()  # Use double brackets to keep it as a DataFrame

        initial_guess = None if warmstart_df is None else warmstart_df.loc[idx]

        # --- Solve model ---
        result = sado.predict_optimal_design(
            row_df,
            options = config['hybrid_surrogate_model_options'],
            oc_min  = oc_min,
            custom_guess=initial_guess)

        # --- Store the results of this iteration ---
        res['TAC'][idx]              = result['fun']
        # OPEX is predicted by sm_opex:
        row_df[processes] = result['x'] # Add processes to row_df, to allow prediction by sm
        res['OPEX_pred'][idx]        = sm_opex.predict(row_df).squeeze()
        res['CAPEX'][idx]            = cost_predictor.calc_capex(result['x'])
        res['feasibility_pred'][idx] = sm_feas.predict(row_df).squeeze()

        # Store production process nominal_capacities:
        for i, process in enumerate(processes):
            res['capacities'][idx, i] = result['x'][i]

        res['n_iter'][idx]           = result['niter']
        res['message'].append(         result['message'])
        res['success'].append(         result['success'])
        res['constr_violation'][idx] = result['constr_violation']
        res['execution_time'][idx]   = result['execution_time']

    # Add value to the DataFrame:
    df_in['TAC_pred']         = res['TAC']
    df_in['OPEX_pred']        = res['OPEX_pred']
    df_in['CAPEX_pred']       = res['CAPEX']
    df_in['feasibility_pred'] = res['feasibility_pred']

    # Add Storage: nominal capacity
    for i, process in enumerate(processes):
        df_in.loc[:, process]        = res['capacities'][:, i]

    df_in['n_iter']           = res['n_iter']
    df_in['message']          = res['message']
    df_in['success']          = res['success']
    df_in['constr_violation'] = res['constr_violation']
    df_in['execution_time']   = res['execution_time']

    return df_in

def design_optimization_var_ts_hybrid_sm_convex_hull(
        df_in:pd.DataFrame, data_in_path, project_file:str, opex_sm_name_config:str,
        opex_sm_name:str, vertex_df:pd.DataFrame, export_path:str=None) -> pd.DataFrame:
    """
    Calculate optimal nominal capacities, TAC, CAPEX, OPEX, lost_load, and full_load_hours
    for a DataFrame of conditions (timeseries) and return result DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        opex_sm_name_config (str): key of the hyperparamters in config.yaml (e.g., ANN_8L)
        opex_sm_name (str): name, under which the opex predictor is saved (e.g, OPEX_NN)
        vertex_df (pd.DataFrame): name, under which the feasibility predictor is saved (e.g, OC_NN)
        export_path (str): path from DATA_OUT to the export folder. If None, no results are
            exported. Defaults to None.
    """

    data_out = get_data_out(data_in_path, project_file)
    # make export folder, if export_path is given:
    if export_path is not None:
        export_path = os.path.join(data_out, export_path)
        os.makedirs(export_path, exist_ok=True)

    # --- Define process and timeseries sets ---
    # Get process sets
    processes = get_process_names(
        data_in_path, project_file,
        only_feasibility_affecting = False,
        storage_or_production      = 'both')

    # get timeseries scaling sets:
    ts_scalings = get_ts_scaling_names(
        data_in_path, project_file,
        only_feasibility_affecting = False)

    # --- Pre-allocate arrays and lists for results
    res = {
        'TAC'              : np.zeros(len(df_in)),
        'OPEX_pred'        : np.zeros(len(df_in)),
        'CAPEX'            : np.zeros(len(df_in)),
        'capacities'       : np.zeros([len(df_in), len(processes)]),
        'n_iter'           : np.zeros(len(df_in)),
        'message'          : [],
        'success'          : [],
        'execution_time'   : np.zeros(len(df_in)),
    }
    # --- Load surrogate models ---
    # Load surrogate model for opex:
    sm_opex = SurrogateModelNN(
        data_in_path,
        project_file,
        input_names = ts_scalings + processes, # All ts_scalings and processes are used her
        output_name = 'OPEX_woLL',
        sm_name     = opex_sm_name_config)

    sm_opex.setup_sm()
    sm_opex.load_model(model_path = os.path.join(data_out, opex_sm_name), filename = opex_sm_name)

    cost_predictor = CostPredictorNN(data_in_path, project_file, sm_opex)
    # Load surrogate model for feasibility:
    feasiblity_lp = Feasibility_LP(data_in_path, project_file, vertex_df)
    # --- Setup hybrid surrogate model ---
    sado = SADOConexHull(cost_predictor, feasiblity_lp)

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx in range(len(df_in)):
        print(idx)
        row_df = df_in.iloc[[idx]].copy()

        # Begin timer:
        t0 = time.time()

        # --- Solve model ---
        result = sado.predict_optimal_design(
            row_df,
            max_iter=100,
            alpha_max=1e-4)

        # End timer:
        t1 = time.time()

        # --- Store the results of this iteration ---
        res['TAC'][idx]              = result['fun']
        # OPEX is predicted by sm_opex:
        row_df[processes] = result['x'] # Add processes to row_df, to allow prediction by sm
        res['OPEX_pred'][idx]        = sm_opex.predict(row_df).squeeze()
        res['CAPEX'][idx]            = cost_predictor.calc_capex(result['x'])

        # Store production process nominal_capacities:
        for i, process in enumerate(processes):
            res['capacities'][idx, i] = result['x'][i]

        res['n_iter'][idx]           = result['niter']
        res['execution_time'][idx]   = t1 - t0

        # Export_path results:
        if export_path is not None:
            filename = f"export_{str(int(row_df['id'].values[0])).zfill(6)}.pkl"
            with open(os.path.join(export_path, filename), 'wb') as f:
                pickle.dump(result, f)

    # Add value to the DataFrame:
    df_in['TAC_pred']         = res['TAC']
    df_in['OPEX_pred']        = res['OPEX_pred']
    df_in['CAPEX_pred']       = res['CAPEX']
    # Add Storage: nominal capacity
    for i, process in enumerate(processes):
        df_in.loc[:, process]        = res['capacities'][:, i]

    df_in['n_iter']           = res['n_iter']
    df_in['execution_time']   = res['execution_time']

    return df_in

def design_optimization_var_ts_hybrid_sm_no_feasibility_predictor(
        df_in:pd.DataFrame, data_in_path, project_file:str, opex_sm_name_config: str,
        opex_sm_name:str) -> pd.DataFrame:
    """
    Calculate optimal nominal capacities, TAC, CAPEX, OPEX, lost_load, and full_load_hours
    for a DataFrame of conditions (timeseries) and return result DataFrame. Does not use any
    feasibiltiy predictor

    Args:
        df_in (pd.DataFrame): Input DataFrame with design data.
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        opex_sm_name_config (str): key of the hyperparamters in config.yaml (e.g., ANN_8L)
        opex_sm_name (str): name, under which the opex predictor is saved (e.g, OPEX_NN)
    """

    data_out = get_data_out(data_in_path, project_file)

    # --- Define process and timeseries sets ---
    # Get process sets
    processes = get_process_names(
        data_in_path, project_file,
        only_feasibility_affecting = False,
        storage_or_production      = 'both')

    # get timeseries scaling sets:
    ts_scalings = get_ts_scaling_names(
        data_in_path, project_file,
        only_feasibility_affecting = False)

    # --- Pre-allocate arrays and lists for results
    res = {
        'TAC'              : np.zeros(len(df_in)),
        'OPEX_pred'        : np.zeros(len(df_in)),
        'CAPEX'            : np.zeros(len(df_in)),
        'capacities'       : np.zeros([len(df_in), len(processes)]),
        'n_iter'           : np.zeros(len(df_in)),
        'message'          : [],
        'success'          : [],
        'constr_violation' : np.zeros(len(df_in)),
        'execution_time'   : np.zeros(len(df_in)),
    }

    # --- Load surrogate models ---
    # Load surrogate model for opex:
    sm_opex = SurrogateModelNN(
        data_in_path,
        project_file,
        input_names = ts_scalings + processes, # All ts_scalings and processes are used her
        output_name = 'OPEX_woLL',
        sm_name     = opex_sm_name_config)

    sm_opex.setup_sm()
    sm_opex.load_model(model_path = os.path.join(data_out, opex_sm_name), filename = opex_sm_name)

    cost_predictor = CostPredictorNN(data_in_path, project_file, sm_opex)
    feas_pred_no_fp = FeasibilityPredictorNoFP(data_in_path, project_file, None, "noFP",
                                            cost_predictor.processes, cost_predictor.timeseries)
    sado = SADOMachineLearning(cost_predictor, feas_pred_no_fp)

    config = get_config_file(data_in_path, project_file)

    # Loop over all designs in the batch and calculate OPEX and lost load
    for idx in range(len(df_in)):
        row_df = df_in.iloc[[idx]].copy()  # Use double brackets to keep it as a DataFrame

        # --- Solve model ---
        result = sado.predict_optimal_design(
            row_df,
            options = config['hybrid_surrogate_model_options'])

        # --- Store the results of this iteration ---
        res['TAC'][idx]              = result['fun']
        # OPEX is predicted by sm_opex:
        row_df[processes] = result['x'] # Add processes to row_df, to allow prediction by sm
        res['OPEX_pred'][idx]        = sm_opex.predict(row_df).squeeze()
        res['CAPEX'][idx]            = cost_predictor.calc_capex(result['x'])

        # Store production process nominal_capacities:
        for i, process in enumerate(processes):
            res['capacities'][idx, i] = result['x'][i]

        res['n_iter'][idx]           = result['niter']
        res['message'].append(         result['message'])
        res['success'].append(         result['success'])
        res['constr_violation'][idx] = result['constr_violation']
        res['execution_time'][idx]   = result['execution_time']

    # Add value to the DataFrame:
    df_in['TAC_pred']         = res['TAC']
    df_in['OPEX_pred']        = res['OPEX_pred']
    df_in['CAPEX_pred']       = res['CAPEX']

    # Add Storage: nominal capacity
    for i, process in enumerate(processes):
        df_in.loc[:, process]        = res['capacities'][:, i]

    df_in['n_iter']           = res['n_iter']
    df_in['message']          = res['message']
    df_in['success']          = res['success']
    df_in['constr_violation'] = res['constr_violation']
    df_in['execution_time']   = res['execution_time']

    return df_in

def export_hsm_predictions_for_operational_optimization(
        df_hsm_prediction:pd.DataFrame, data_in_path, project_file:str, filepath_out:str,
        remove_negative_values:bool = False) -> None:
    """ Routine cleans results generate by design_optimization_var_ts_hybrid_sm, so that the .csv
    can be used for operational_optimization_var_ts right away.
    First, all negative capacity values are set to zero.
    Then, the row 'task_index' is removed. 

    Args:
        df_hsm_prediction (pd.DataFrame): result file generated by
            design_optimization_var_ts_hybrid_sm
        data_in_path (str): Path to the data files.
        project_file (str): Project file path.
        filepath_out (str): filepath for the export (incl. .csv)
    """
    # Filter negative values for processes
    df_filtered = df_hsm_prediction.copy()

    if remove_negative_values:
        processes = get_process_names(data_in_path, project_file, False, 'both')

        # Set all processes with negative values to 0
        for process in processes:
            # Count number of negative values:
            n_neg = (df_filtered[process] < 0).sum()
            if n_neg > 0:
                print(f"Warning: '{n_neg}' negative values found for process '{process}'. "
                      "Setting them to zero.")
            df_filtered.loc[df_filtered[process] < 0, process] = 0

    # Remove column 'task_index': (avoids error when hpc tries to insert this column again:)
    if 'task_index' in df_filtered.columns:
        df_filtered = df_filtered.drop(['task_index'], axis=1)

    # Save dataframe:
    df_filtered.to_csv(filepath_out, index=False)

def explore_convex_hull(df_in:pd.DataFrame, data_in_path, project_file:str,
                        debug:bool = False) -> pd.DataFrame:
    """ Does feasible space exploration for each search direction (=row of df_in) and returns df_out
    with the results of the exploration, which are the found vertex points of the convex hull.

    Args:
        df_in (pd.DataFrame): Input DataFrame containing the search directions
        data_in_path (str): Path to the data files.
        project_file (str): Project file name.
        debug (bool, optional):  If True: gurobi log is printed. Defaults to False.

    Raises:
        ValueError: If search_mode 'min_max' is used.

    Returns:
        pd.DataFrame: Contains the results for each search direction.
    """
    # Copy df_in:
    df_out = df_in.copy()

    # Initlialize convex hull explorer object:
    ch_explorer = Convex_Hull_Explorer(data_in_path, project_file)

    # Loop over all rows of df_out
    for idx in df_out.index:
        if idx % 20 == 0:
            print(idx)
        df_out = ch_explorer.run_optimization(df_out, idx, debug)

    return df_out

def prepare_design_results_for_max_overcapacity_optimization(
        df_out:pd.DataFrame, data_in_path, project_file:str,
        file_out_name:str='design_results_for_maxoc.csv') -> None:
    """ This function is intended to be used AFTER a design optimitation to prepare the result df
    for an overcapacity maximization.

    Args:
        df_out (pd.DataFrame): Containing the optimal designs for several scenarios
        data_in_path (str): path to the input files
        project_file (str): project file (e.g., project.yaml)
        file_out_name (str): name of the exported .csv. Defaults to 'design_results_for_maxoc.csv'
    """

    # Copy the dataframe:
    df_export = df_out.copy()

    # Removing columns which would case an error:
    products = list(get_input_file(data_in_path, project_file, 'products').keys())
    processes = get_process_names(data_in_path, project_file, False, 'production')
    column_names = \
        ['task_index', 'OPEX', 'TAC', 'CAPEX', 'lost_load'] + \
        ['solver_time', 'termination_condition']            + \
        [f"lost_load_{product}" for product in products]    + \
        [f"full_load_hours_{process}" for process in processes]

    df_export.drop(columns=column_names, errors='ignore', inplace=True)

    data_computed_in = get_data_computed_in(data_in_path, project_file)
    df_export.to_csv(os.path.join(data_computed_in, file_out_name), index=False)

def combine_operational_and_max_overcapacity_results(data_in: str, project_file:str,
        filename_operation:str, filename_maxoc:str, idx_min:int=0) -> pd.DataFrame:
    """ Combines results of two methods testing the SADO predictions: operational optimizations and
        max overcapacity optimizations

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
        filename_operation (str): path to the .csv file, which contains the results of operational
            optimizations that test SADO prediction 
        filename_maxoc (str):  path to the .csv file, which contains the results of max overcapacity
            optimizations that test SADO prediction 
        idx_min (int, optional): First index used to combine DataFrames. Defaults to 0.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: combined DataFrame, with the 
    """
    voll_dict = get_value_of_lost_load(data_in, project_file)
    # Sanity check:
    sanity_check_value_of_lost_load(voll_dict)
    # Extract last value of dict (sanity check above makes sure, all are identical)
    value_of_lostload = list(voll_dict.values())[0]

    # Read results of operational optimization
    df_operation = pd.read_csv(filename_operation)[idx_min:]
    df_operation = df_operation.sort_values(by='id', ascending=True)

    # Read results of max overcapacity optimization
    df_max_oc = pd.read_csv(filename_maxoc)[idx_min:]
    df_max_oc = df_max_oc.sort_values(by='id', ascending=True)

    # Combine both DataFrames based on the operational opt
    df_combined = df_operation.copy()
    df_combined['overcapacity'] = df_max_oc['overcapacity']
    # Calc TAC that does not consider CENS (cost of energy not served):
    df_combined['TAC_woLL'] = df_combined['TAC'] - df_combined['lost_load'] * value_of_lostload

    # Check that both dataframes have the same length:
    if len(df_operation) != len(df_max_oc):
        print( "Warning: Incomplete resoluts" +
            f"Dataframes in \n{filename_operation}\nand\n{filename_maxoc}\nhave different lengths:"
            f" {len(df_operation)} != {len(df_max_oc)}"
            )

    # Check that both dataframes have the same ids:
    elif not all(df_operation['id'].values == df_max_oc['id'].values):
        raise ValueError(f"IDs in {filename_operation} and {filename_maxoc} do not match!")

    return df_combined

def add_regret_to_sado_results(data_in: str, project_file:str, df_sado:pd.DataFrame,
        df_benchmark:pd.DataFrame) -> pd.DataFrame:
    """ Calculates the regret of a design determined by SADO (or other surrogate-based design
    prediction methods). Before this function is used, the SADO designs must be tested with
    operational optimizations.
    Adds to df_sado:
        'TAC_opt'           # Optimal TAC from design background model
        'regret_wLL'        # Absolute regret (considering CENS), therefore with LL (wLL)
        'rel_regret_wLL'    # Relative regret (considering CENS), therefore with LL (wLL)
        'regret_woLL'       # Absolute regret (not considering CENS), therefore without LL (woLL)
        'rel_regret_woLL'   # Relative regret (not considering CENS), therefore without LL (woLL)

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
        df_sado (pd.DataFrame): contains the sado predictions (design by sado AND the opex
            and tac as calculated by operational optimizations. Can already contain TAC_woLL.
            If not, TAC_woLL is calulcated in this function as well.
        df_benchmark (pd.DataFrame): contains the design and tac as determined by
            background model design optimizations

    Returns:
        pd.DataFrame: copy of df_results_sado but with TAC_opt, rel_regret_woLL, regret_woLL,
            rel_regret, regret
    """

    # Check that TAC_woLL is in df_results_sado:
    if not 'TAC_woLL' in df_sado.columns:
        # Calculate TAC_woLL
        print("'TAC_woLL' not found in df_results_sado. Calculating 'TAC_woLL' now.")
        voll_dict = get_value_of_lost_load(data_in, project_file)
        # Sanity check:
        sanity_check_value_of_lost_load(voll_dict)
        # Extract last value of dict (sanity check above makes sure, all are identical)
        df_sado['TAC_woLL'] = df_sado['TAC'] - df_sado['lost_load'] * list(voll_dict.values())[0]

    # Assert both DataFrames have identical number of indices:
    if len(df_benchmark) != len(df_sado):
        raise ValueError(
            "Length of benchmark and hsm prediction do not match:"
            f" ({len(df_benchmark)} != {len(df_sado)})"
        )

    # Add optimal TAC from design background model:
    df_sado['TAC_opt'] = df_benchmark['TAC']

    ## (considering CENS), therefore with LL (wLL)
    # Calculate absolute regret
    df_sado['regret_wLL']       = df_sado['TAC'].values - df_sado['TAC_opt'].values
    # Calculate relative regret
    df_sado['rel_regret_wLL']   = df_sado['regret_wLL']  / df_sado['TAC_opt'].values * 100

    ## (not considering CENS), therefore without LL (woLL)
    # Calculate absolute regret
    df_sado['regret_woLL']      = df_sado['TAC_woLL'].values - df_sado['TAC_opt'].values
    # Calculate relative regret
    df_sado['rel_regret_woLL']  = df_sado['regret_woLL'] / df_sado['TAC_opt'].values * 100

    return df_sado
