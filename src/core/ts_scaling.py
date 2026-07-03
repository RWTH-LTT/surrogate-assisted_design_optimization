""" Module for handling timeseries scaling"""
import warnings
import numpy as np
from core.helpers import calculate_weighted_mean, get_config_file, get_input_file

def sanity_check_ts(ts_scaling_dict:dict):
    """ Contucts sanity checks for the timeseries scaling dict.

    Args:
        ts_scaling_dict (dict): Loaded from timeseries_scaling.yaml

    """
    # Loop over all uncertainties:
    for ts_name, ts_dict in ts_scaling_dict.items():

        # Skip availability, as the keys differ:
        if ts_dict['ts_type'] == "availability":
            sanity_check_ts_availability(ts_name, ts_dict)

            continue

    #### General check for all scaling_modes:
        # Check, that 'min' and 'max' for 'mean' are non-negative:
        if any(ts_dict['param']['mean'][i] < 0 for i in ['min', 'max']):
            raise ValueError(
                f"Found negative values for uncertainties.{ts_name}.mean ."
            )

        # Check, that 'max' >= 'min' for 'mean':
        if ts_dict['param']['mean']['max'] <= ts_dict['param']['mean']['min']:
            raise ValueError(
                f"('max' <= 'min') for uncertainties.{ts_name}.mean ."
            )

        # Check, that 'scale_mode' is in ['mean_and_peak', 'mean']:
        scale_mode = get_scale_mode_ts(ts_dict)
        if scale_mode not in ['mean_and_peak', 'mean', 'mean_and_relpeak']:
            raise ValueError(
                f"Invalid 'scale_mode' {scale_mode}"
            )

    #### Check for 'scale_mode' == 'mean_and_peak':
        if scale_mode == 'mean_and_peak':

            # Check, that 'min' and 'max' for 'peak' are non-negative:
            if any(ts_dict['param']['peak'][i] < 0 for i in ['min', 'max']):
                raise ValueError(
                    f"Found negative values for uncertainties.{ts_name}.peak ."
                )

            # Check, that 'max' > 'min' for 'peak':
            if ts_dict['param']['peak']['max'] <= ts_dict['param']['peak']['min']:
                raise ValueError(
                    f"('max' <= 'min') for uncertainties.{ts_name}.peak ."
                )

            # Check that 'max(peak)' >= 'max(mean)':
            if ts_dict['param']['peak']['max'] < ts_dict['param']['mean']['max']:
                raise ValueError(
                    f"( max(peak) < max(mean) ) for uncertainties.{ts_name}"
                )

            # Check that 'min(peak)' >= 'min(mean)':
            if ts_dict['param']['peak']['min'] < ts_dict['param']['mean']['min']:
                raise ValueError(
                    f"( min(peak) < min(mean) ) for uncertainties.{ts_name}"
                )

    #### Check for 'scale_mode' == 'mean_and_relpeak':
        if scale_mode == 'mean_and_relpeak':

            # Check, that 'min' and 'max' for 'relpeak' are non-negative:
            if any(ts_dict['param']['relpeak'][i] < 1 for i in ['min', 'max']):
                raise ValueError(
                    f"Found values <1 for uncertainties.{ts_name}.relpeak ."
                )

            # Check, that 'max' > 'min' for 'relpeak':
            if ts_dict['param']['relpeak']['max'] <= ts_dict['param']['relpeak']['min']:
                raise ValueError(
                    f"('max' <= 'min') for uncertainties.{ts_name}.relpeak ."
                )

    ### Check for timeseries depending on product:
        if ts_dict['ts_type'] in ['demand', 'pricebuy', 'pricesell']:

            # Check that 'product' is a key:
            if 'product' not in ts_dict:
                raise KeyError(
                    f"'product' not in uncertainties.{ts_name}"
                )

            # Check that 'process' is not a key:
            if 'process' in ts_dict:
                raise KeyError(
                    f"'process' must not be a key for uncertainties.{ts_name}"
                )

        else:
            raise ValueError(
                f"Invalid 'ts_type' for uncertainties.{ts_name}: '{ts_dict['ts_type']}'"
            )

def sanity_check_ts_availability(ts_name:str, ts_dict:dict):
    """ Conducts sanity check on ts_scaling for availability interpolation

    Args:
        ts_name (str): name of the scaling
        ts_dict (dict): subdict from timeseries_scaling.yaml
    """

    scaler = ts_dict['param']['interpolate']

    # Sanity check for scaler min < default < max:
    if not scaler['min'] <= scaler['default'] <= scaler['max']:
        raise ValueError(
            f"{ts_name}: min <= default <= max is voilated."
        )
    # Sanity check: min => 0:
    if scaler['min'] < 0:
        raise ValueError(
            f"{ts_name}: min must be nonnegative, but is {scaler['min']}."
        )
    # Sanity check: max <= 1:
    if scaler['max'] > 1:
        raise ValueError(
            f"{ts_name}: max must not be larger 1, but is {scaler['max']}."
        )
    # Sanity check: max >= min:
    if scaler['max'] < scaler['min']:
        raise ValueError(
            f"{ts_name}: max ({scaler['max']}) must not be smaller than "+
            f"min ({scaler['max']})."
        )

    # Sanity check: param must not contain 'mean' or 'peak':
    if any(key in scaler for key in ['mean', 'peak']):
        raise ValueError(
            f"'availability' for '{ts_name}' must not contain peak' or 'mean'"
        )

def get_scale_mode_ts(ts_dict:dict):
    """ Gets the scale mode ('mean', 'mean_and_peak', or 'interpolate') from a subdict in 
    timeseries_scaling.yaml
    'mean': for demand and pricebuy.
        Only the mean value of the timeseries is scaled up and down.
    'mean_and_peak': for demand and pricebuy.
        Both mean and peak value can be scaled up and down
    'mean_and_relpeak': for pricebuy.
        Both mean and peak value can be scaled up and down. The peak value is defined relative to
        the mean value. relpeak=1 -> no variation. relpeak=5 -> high variation
    'interpolate': for 'availability'
        The resulting timeseries is generated by interpolating between two timeseries.

    Args:
        ts_dict (dict): subdict from  timeseries_scaling.yaml

    Returns:
        str: 'mean', 'mean_and_peak', 'mean_and_relpeak' or 'availability'
    """

    # Check that no more than two arguments are given:
    if len(ts_dict['param']) > 2:
        raise ValueError(
            "Something is wrong with the ts_scaling dict, as there are more than 2 entries: " +
            f"{ts_dict['param'].keys()}"
        )

    if ts_dict['ts_type'] in {'demand', 'pricebuy'}:
        # Check, that only valid keywords are present:
        for param in ts_dict['param']:
            if param not in {'peak', 'mean', 'relpeak', 'interpolate'}:
                raise ValueError(f"Invalid keyword for ts_scaling: '{param}'")
        # Case 1: (valid) 'peak' and 'mean' are in param:
        if ('peak' in ts_dict['param']) and ('mean' in ts_dict['param']):
            return 'mean_and_peak'

        # Case 2: (valid) 'peak' and 'mean' are in param:
        if ('relpeak' in ts_dict['param']) and ('mean' in ts_dict['param']):
            return 'mean_and_relpeak'

        # Case 3: (valid) only 'mean' is in param:
        if 'mean' in ts_dict['param']:
            assert len(ts_dict['param']) == 1
            return 'mean'

        # Case 4: (invalid) only 'peak' is in param:
        if 'peak' in ts_dict['param'] or 'relpeak' in ts_dict['param'] :
            raise ValueError(
                "Only 'peak' is given under 'param'. This is invalid. Please add 'mean'."
            )

        # Case 5: (invalid) nothing is given in param:
        raise ValueError(
            "No 'peak' or 'mean' in ts_scaling.param. "
        )

    if ts_dict['ts_type'] == 'availability':
        return 'interpolate'

    raise ValueError(f"Unknown ts_type '{ts_dict['ts_type']}'")

def get_products_or_processes_scaling_ts(ts_scaling:dict, ts_type: str, product_or_process:str):
    """ Returns a list of products/ processes which are used to index a timeseries

    Args:
        ts_dict (dict): content of timeseries_scaling.yaml
        ts_type (str): name of the timeseries in input_dict, e.g., 'demand'
        product_or_process (str): 'product' or 'process

    Raises:
        SyntaxError: If an invalid product_or_process in given

    Returns:
        list: list of products
        list: list of keys in time_series_scaling.yaml for the respective ts_type
    """

    if product_or_process not in {'product', 'process'}:
        raise SyntaxError(f"Expected 'product' or 'process'. Got '{product_or_process}' instead.")

    product_or_process_list = []
    ts_scaling_name_list = []
    for ts_name, ts_dict in ts_scaling.items():
        if ts_type == ts_dict['ts_type']:
            assert ts_dict[product_or_process] not in product_or_process_list
            product_or_process_list.append(ts_dict[product_or_process])
            ts_scaling_name_list.append(ts_name)

    # # Print warning, if no timeseries has been found:
    # if not ts_scaling_name_list:
    #     warnings.warn(
    #         f"No timeseries of type '{ts_type}' is used. " +
    #         "Please doublecheck."
    #     )
    return product_or_process_list, ts_scaling_name_list

def sanity_check_ts_scaling_bounds(ts_scaling:dict, input_dict:dict, ts_type:str):
    """ Conducts sanity check: The mean and peak in the original timeseries
    must be inside the bounds defined timeseries_scaling.yaml

    Args:
        ts_scaling (dict): content of timeseries_scaling.yaml
        input_dict (dict): input dict generated by UnscaledInputDictGenerator
            or AggregatedInputDictGenerator
        ts_type (str): name of the timeseries in input_dict, e.g., 'demand'
    """
    # Get product list for scaled timeseries of ts_type
    products, key_list = get_products_or_processes_scaling_ts(
        ts_scaling, ts_type=ts_type, product_or_process='product'
    )

    # Loop over all timeseries, which can get scaled:
    for product, ts_key in dict(zip(products, key_list)).items():
        ### For 'mean' and 'mean_and_peak':
        # Read min and max mean values:
        mean_min = ts_scaling[ts_key]['param']['mean']['min']
        mean_max = ts_scaling[ts_key]['param']['mean']['max']
        # Get mean_old:
        mean_old = input_dict[f"ts_{ts_type}_old_mean"][product]
        if not mean_min <= mean_old <= mean_max:
            warnings.warn(
                f"\nScaling factors for '{product}' 'mean' {ts_type} do not match the " +
                "original timeseries:"+
                f"\n\tts_scaling: min = {mean_min}, max = {mean_max}."+
                f"\n\tOriginal mean = {mean_old} is not inside the bounds."
            )
        ### Only for 'mean_and_peak':
        if get_scale_mode_ts(ts_scaling[ts_key]) == 'mean_and_peak':
            # Read min and max mean values:
            peak_min = ts_scaling[ts_key]['param']['peak']['min']
            peak_max = ts_scaling[ts_key]['param']['peak']['max']
            # Get mean_old:
            peak_old = input_dict[f"ts_{ts_type}_old_peak"][product]
            if not peak_min <= peak_old <= peak_max:
                warnings.warn(
                    f"\nScaling factors for '{product}' 'peak' {ts_type} do not match the " +
                    "original timeseries:"+
                    f"\n\tts_scaling: min = {peak_min}, max = {peak_max}."+
                    f"\n\tOriginal peak = {peak_old} is not inside the bounds."
                )
        ### Only for 'mean_and_relpeak':
        if get_scale_mode_ts(ts_scaling[ts_key]) == 'mean_and_relpeak':
            # Read min and max mean values:
            relpeak_min = ts_scaling[ts_key]['param']['relpeak']['min']
            relpeak_max = ts_scaling[ts_key]['param']['relpeak']['max']
            peak_min = mean_min * relpeak_min
            peak_max = mean_max * relpeak_max
            # Get mean_old:
            peak_old = input_dict[f"ts_{ts_type}_old_peak"][product]
            if not peak_min <= peak_old <= peak_max:
                warnings.warn(
                    f"\nScaling factors for '{product}' 'relpeak' {ts_type} do not match the " +
                    "original timeseries:"+
                    f"\n\tts_scaling relpeak: min = {relpeak_min}, max = {relpeak_max}."+
                    f"\n\tCan only lead to peak values in: min = {peak_min}, max = {peak_max}."+
                    f"\n\tOriginal peak = {peak_old} is not inside the bounds."
                )

def sanity_check_ts_scaling_nonnegative(ts_scaling:dict, input_dict:dict, ts_type:str):
    """ Checks, if timeseries scaling could lead to negative values.

    Args:
        ts_scaling (dict): content of timeseries_scaling.yaml
        input_dict (dict): input dict generated by UnscaledInputDictGenerator
            or AggregatedInputDictGenerator
        ts_type (str): name of the timeseries in input_dict, e.g., 'demand'

    Raises:
        ValueError: If timeseries scaling lead to negative values
    """
    # Get product list for scaled timeseries of ts_type
    products, key_list = get_products_or_processes_scaling_ts(
        ts_scaling, ts_type=ts_type, product_or_process='product'
    )

    # Loop over all timeseries, which can get scaled:
    for product, ts_key in dict(zip(products, key_list)).items():
        # If the orginal timeseries in nonnegative, negative values are only possible if
        # peak and mean are both scaled. Therefore, the only "mean" cases are not checked
        if get_scale_mode_ts(ts_scaling[ts_key]) == 'mean':
            continue

        # The critical combination, which can lead to negative values is
        # maximum value of peak with minimum value of mean. If the lowest point
        # of the original timeseries is still nonnegative for that scaling, all
        # other scalings are feasible aswell.

        # Read max peak value:
        peak_max = ts_scaling[ts_key]['param']['peak']['max']
        # Read min mean value:
        mean_min = ts_scaling[ts_key]['param']['mean']['min']

        # Get peak_old:
        peak_old = input_dict[f"ts_{ts_type}_old_peak"][product]
        # Get mean_old:
        mean_old = input_dict[f"ts_{ts_type}_old_mean"][product]

        # calculate a = (peak_new - mean_new) / (peak_old - mean_old)
        a = (peak_max - mean_min) / (peak_old - mean_old)

        # b = mean_new - a * mean_old
        b = mean_min - a * mean_old

        # Get minimal value of the original timeseries:
        min_old = min(
            input_dict[ts_type][key]
            for key in input_dict[ts_type].keys()
            if key[0] == product
        )
        # Calculate critical scalign:
        min_crit = a * min_old + b
        # if min_crit is negative, print a warning:
        if min_crit < 0:

            crit_term = 1 - ((peak_old - mean_old)/(min_old - mean_old))
            warnings.warn(
                f"\nFor the '{product}' {ts_type} timeseries, " +
                f"scaling leads to negative values: {min_crit}. " +
                "Current values:"+
                f"\npeak_max = {peak_max}"
                f"\nmean_min = {mean_min}"
                f"\npeak_old = {peak_old}"
                f"\nmean_old = {mean_old}"
                "\nThere are multiple ways to prevent that:" +
                "\n\t- In timeseries_scaling.yaml: Adjust the min value of 'mean' " +
                f"to {peak_max / crit_term} or higher."+
                "\n\t- In timeseries_scaling.yaml: Adjust the max value of 'peak' " +
                f"to {mean_min * crit_term} or lower." +
                "\n\t- Change the timeseries data."
            )

def add_ts_scaling_old_values(ts_type:str, input_dict: dict):
    """ Adds the old values (from the input .csv files) for a product timeseries
    to the input_dict

    Args:
        ts_type (str): name of the timeseries
            e.g., 'demand' or 'pricebuy'
        input_dict (dict): input dict generated by UnscaledInputDictGenerator
            or AggregatedInputDictGenerator
    
    Return:
        (dict): Input dict with two new entries:
            1. 
    """
    ts_old_mean = {}
    ts_old_peak = {}

    for product in input_dict['products']:
        # Fill in mean:
        ts_old_mean[product] = calculate_weighted_mean(
            timeseries_dict = input_dict[ts_type],
            product_or_process= product,
            weights_dict=input_dict['time_slice_weight']
        )
        # Fill in peak:
        ts_old_peak[product] = max(
            input_dict[ts_type][k]
            for k in input_dict[ts_type].keys()
            if k[0] == product
        )

    # Add to input_dict:
    input_dict[f"ts_{ts_type}_old_mean"] = ts_old_mean
    input_dict[f"ts_{ts_type}_old_peak"] = ts_old_peak

    return input_dict

def get_ts_scaling_names(data_in_path:str, project_file:str, only_feasibility_affecting:bool,
                         which_ts_scale_mode:str='all'):
    """ Returns a list of names of all timeseries scalings, which are used in the project.

    Args:
        data_in_path (str): path to the data input directory
        project_file (str): name of the project file
        only_feasibility_affecting (bool): If True, only return timeseries scalings
            which affect feasibility. If False, return all timeseries scalings.
        which_ts_scale_mode (str): Which scale mode can be returned. Must be in 
            {'mean', 'peak', 'interpolate', 'all'}
    Returns:
        list: List of names of all timeseries scalings, which are used in the project.
    """

    assert which_ts_scale_mode in {'mean', 'peak', 'realpeak', 'interpolate', 'all'}

    # Get ts_scaling dict
    config = get_config_file(data_in_path, project_file)
    ts_scaling_dicts = get_input_file(
        data_in_path, project_file, 'timeseries_scaling'
        )['timeseries_scaling'][config['timeseries_scaling']]

    # Initialize empty list
    ts_scaling_names = []

    # Loop over all entries:
    for ts_dict in ts_scaling_dicts.values():
        # Skip, if no effect on feasiblity and 'only_feasibility_affecting' is True
        if only_feasibility_affecting and not ts_dict['affect_feasibility']:
            continue

        ts_scale_mode = get_scale_mode_ts(ts_dict)

        if ts_scale_mode == 'mean':
            if which_ts_scale_mode in {'mean', 'all'}:
                ts_scaling_names.append(f"{ts_dict['ts_type']}_{ts_dict['product']}_mean")

        elif ts_scale_mode == 'interpolate':
            if which_ts_scale_mode in {'interpolate', 'all'}:
                ts_scaling_names.append(f"{ts_dict['ts_type']}_{ts_dict['process']}_interpolate")

        elif ts_scale_mode == 'mean_and_peak':
            if which_ts_scale_mode in {'mean', 'all'}:
                ts_scaling_names.append(f"{ts_dict['ts_type']}_{ts_dict['product']}_mean")
            if which_ts_scale_mode in {'peak', 'all'}:
                ts_scaling_names.append(f"{ts_dict['ts_type']}_{ts_dict['product']}_peak")

        elif ts_scale_mode == 'mean_and_relpeak':
            if which_ts_scale_mode in {'mean', 'all'}:
                ts_scaling_names.append(f"{ts_dict['ts_type']}_{ts_dict['product']}_mean")
            if which_ts_scale_mode in {'relpeak', 'all'}:
                ts_scaling_names.append(f"{ts_dict['ts_type']}_{ts_dict['product']}_relpeak")

    return ts_scaling_names

def get_ts_scaling_key(ts_dict:dict, ts_scaling:dict) -> str:
    """Gets the key that the ts_scaling entry in ts_scaling.yaml has. (e.g., 'ts_demand_heat')"""
    ts_scaling_key = [
        key for key, value in ts_scaling.items()
        if (
            ts_dict['ts_type'] == value['ts_type']
            and
            ts_dict['product_or_process'] == value.get('process', value.get('product'))
        )]

    if len(ts_scaling_key) != 1:
        raise KeyError(
            f"Something went wrong with {ts_dict}, as '{len(ts_scaling_key)}' " +
            "possible ts_scalings exist.")

    return ts_scaling_key[0]

def get_ts_scaling_dict(ts_scaling_name:dict, ts_scaling:dict) -> dict:
    """Gets the dict that the ts_scaling_name entry in ts_scaling.yaml has. """
    ts_type, product_or_process, _ = ts_scaling_name.split('_')

    ts_scaling_key = [
        key for key, value in ts_scaling.items()
        if (
            ts_type == value['ts_type']
            and
            product_or_process == value.get('process', value.get('product'))
        )]

    if len(ts_scaling_key) != 1:
        raise KeyError(
            f"Something went wrong with {ts_scaling_name}, as '{len(ts_scaling_key)}' " +
            "possible ts_scalings exist.")

    return ts_scaling[ts_scaling_key[0]]


def scale_timeseries(timeseries: np.ndarray, scale_mode: str, old_mean: float,
                     old_peak: float = None, new_mean: float = None, new_peak: float = None,
                     new_relpeak: float = None) -> np.ndarray:
    """
    Scales a timeseries using a linear transformation (y_new = a * y_old + b).

    The scaling factors 'a' and 'b' are determined by the scale_mode.

    Args:
        timeseries: The unscaled timeseries data numpy array.
        scale_mode: The scaling method. Must be one of:
                    'mean', 'mean_and_peak', 'mean_and_relpeak'.
        old_mean: The mean of the original timeseries.
        old_peak: The peak of the original timeseries (required for
                  'mean_and_peak' and 'mean_and_relpeak').
        new_mean: The target mean value.
        new_peak: The target peak value (required for 'mean_and_peak').
        new_relpeak: The target relative peak (new_peak = new_mean * new_relpeak)
                     (required for 'mean_and_relpeak').

    Returns:
        A new numpy array with the scaled timeseries.
    """

    a = 0.0
    b = 0.0

    if scale_mode == 'mean':
        a = new_mean / old_mean
        b = 0.0

    elif scale_mode == 'mean_and_peak':
        if new_peak < new_mean:
            warnings.warn(f"new_peak ({new_peak}) < new_mean ({new_mean}). Proceeding anyway.")

        a = (new_peak - new_mean) / (old_peak - old_mean)
        b = new_mean - a * old_mean

    elif scale_mode == 'mean_and_relpeak':
        internal_new_peak = new_mean * new_relpeak

        if internal_new_peak < new_mean:
            warnings.warn(
                f"Calculated new_peak ({internal_new_peak}) < new_mean ({new_mean}). " +
                "Proceeding anyway."
            )

        a = (internal_new_peak - new_mean) / (old_peak - old_mean)
        b = new_mean - a * old_mean

    else:
        raise ValueError(
            f"Unknown scale_mode: '{scale_mode}'. " +
            "Supported modes are 'mean', 'mean_and_peak', 'mean_and_relpeak'.")

    return a * timeseries + b

def interpolate_timeseries(
        timeseries_1: np.ndarray, timeseries_2: np.ndarray, scaler: float) -> np.ndarray:
    """
    Linearly interpolates between two timeseries.

    y_new = timeseries_1 * (1 - scaler) + timeseries_2 * scaler

    Args:
        timeseries_1: The first timeseries (y_1st).
        timeseries_2: The second timeseries (y_2nd).
        scaler: The interpolation factor. 0.0 returns timeseries_1,
                1.0 returns timeseries_2.

    Returns:
        A new numpy array with the interpolated timeseries.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    ts1 = np.asarray(timeseries_1)
    ts2 = np.asarray(timeseries_2)

    if ts1.shape != ts2.shape:
        raise ValueError(f"Timeseries shapes do not match: {ts1.shape} vs {ts2.shape}")

    if not (0.0 <= scaler <= 1.0):
        warnings.warn(
            f"Scaler ({scaler}) is outside the typical [0, 1] range. Proceeding with extrapolation."
            )

    return ts1 * (1 - scaler) + ts2 * scaler
