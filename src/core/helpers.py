""" Module with helper functions. """
import os
from pathlib import Path
import yaml

class UniqueKeyLoader(yaml.SafeLoader):
    """ A YAML loader that raises an error when duplicate keys are found. 
    This fixes the default behavior of PyYAML, which silently overwrites duplicate keys."""
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)

def add_file_extension(filename:str, extension:str):
    """ adds file extension to a filename, if it not already exists

    Args:
        filename (str): filename with or without extension
        extension (str): file extension (e.g. 'csv')

    Returns:
        str: filename with the extension
    """
    # remove prefix, if existing (e.g. '.csv' -> 'csv')
    extension = extension.removeprefix('.')

    # if the filename does not end with the extension, add extension and return
    if not filename.endswith(f'.{extension}'):
        return f'{filename}.{extension}'

    return filename

def get_experiment_id(data_in_path, project_file):
    """ read experiment_id from project_file (e.g. project.yaml) and returns it

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)

    Returns:
        str: experiment_id
    """
    with open(os.path.join(data_in_path, project_file), 'r', encoding='utf-8') as file:
        project = yaml.load(file, Loader=UniqueKeyLoader)
    experiment_id = project['experiment_ID']
    assert isinstance(experiment_id, str)

    return experiment_id

def get_config_file(data_in_path, project_file):
    """ read the config file and returns it

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)

    Returns:
        dict: config file
    """
    with open(os.path.join(data_in_path, project_file), 'r', encoding='utf-8') as file:
        project = yaml.load(file, Loader=UniqueKeyLoader)

    with open(os.path.join(data_in_path, project['config']), 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=UniqueKeyLoader)

    return config

def get_data_computed_in(data_in_path: str, project_file:str):
    """ Get the path to the DATA_COMPUTED_IN directory

    Args:
        data_in_path  (str): path to the DATA_IN directory
        project_file (str): path to the project file

    Returns:
        (Path): path to the DATA_COMPUTED_IN directory
    """
    experiment_id = get_experiment_id(data_in_path, project_file)
    # Get the parent directory of the data_in_path
    data_in_path = Path(data_in_path)

    # Create the path to the DATA_COMPUTED_IN directory
    data_computed_in_path =  data_in_path.parent / 'ComputedInput_Data' / experiment_id

    # If the directory does not exist, create it
    os.makedirs(data_computed_in_path, exist_ok=True)

    return data_computed_in_path

def get_data_out(data_in_path: str, project_file:str):
    """ Get the path to the DATA_OUT directory

    Args:
        data_in_path  (str): path to the DATA_IN directory
        project_file (str): path to the project file

    Returns:
        (Path): path to the DATA_OUT directory
    """
    experiment_id = get_experiment_id(data_in_path, project_file)
    # Get the parent directory of the data_in_path
    data_in_path = Path(data_in_path)

    # Create the path to the DATA_OUT directory
    data_out_path =  data_in_path.parent / 'Output_Data' / experiment_id

    # If the directory does not exist, create it
    os.makedirs(data_out_path, exist_ok=True)

    return data_out_path

def get_tsa_path(data_in_path, project_file):
    """ Returns the default path fo the tsa_export files

    Args:
        data_in_path (str): ath to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)

    Returns:
        str:  path fo the tsa_export files
    """

    config = get_config_file(data_in_path, project_file)
    experiment_id = get_experiment_id(data_in_path, project_file)

    tsa_path = os.path.join(
        data_in_path, config['time_series_aggregation']['tsa_path'],experiment_id)
    os.makedirs(tsa_path, exist_ok=True)
    return tsa_path

def get_minimal_capacity(technology_dict: dict):
    """ Returns the minimal capacity of a technology.

    Args:
        technology_dict (dict): dict of a specific technology generated from technologies.yaml.
            e.g. self.technologies['B'] for the technology_dict of 'B'

    Raises:
        ValueError: Raised if the category of the technology is unknown.

    Returns:
        float: Minimal capacity of the technology.            
    """

    # Get the category of the technology
    technology_type = technology_dict['category']

    # Get and return the minimal capacity depending on the category
    if technology_type == 'production':
        return technology_dict.get('minimal_capacity_kW', 0)
    if technology_type == 'storage':
        return technology_dict.get('minimal_capacity_kWh', 0)
    raise ValueError(f"Unknown category '{technology_type}'.")

def get_potential_capacity(technology_dict: dict):
    """ Returns the potential capacity of a technology.

    Args:
        technology_dict (dict): dict of a specific technology generated from technologies.yaml.
            e.g. self.technologies['B'] for the technology_dict of 'B'

    Raises:
        ValueError: Raised if the category of the technology is unknown.

    Returns:
        float: Potential capacity of the technology.            
    """

    # Get the category of the technology
    technology_type = technology_dict['category']

    # Get and return the minimal capacity depending on the category
    if technology_type == 'production':
        return technology_dict['potential_capacity_kW']
    if technology_type == 'storage':
        return technology_dict['potential_capacity_kWh']

    raise ValueError(f"Unknown category '{technology_type}' for technology.")

def get_value_of_lost_load(data_in_path:str, project_file:str):
    """ Get the value of lost load for each product that allows lost load

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)

    Returns:
        dict: dict with product as key and value_of_lost_load as value 
    """
    # Load project file:
    with open(os.path.join(data_in_path, project_file), 'r', encoding='utf-8') as file:
        project = yaml.load(file, Loader=UniqueKeyLoader)
    # Load Products:
    with open(os.path.join(data_in_path,  project['products']), 'r', encoding='utf-8') as file:
        products = yaml.load(file, Loader=UniqueKeyLoader)

    # Return only value_of_lost_load if 'allow_lost_load' == True for that product
    return {
        product: params['value_of_lost_load']
        for product, params in products.items()
        if params['allow_lost_load']
    }

def get_input_file(data_in_path:str, project_file:str, input_str:str):
    """ Reads the input file from project and returns it:

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
            (e.g. project.yaml)
        input_str (str): name of the input in the project file
            (e.g. 'uncertainties', 'config', technologies', 'designs', 'superstructures',
            'timeseries', 'general_param', 'products', 'technology_categories')
    
    Returns
        dict: content of the input yaml. file 
    """

    # Load project file:
    with open(os.path.join(data_in_path, project_file), 'r', encoding='utf-8') as file:
        project = yaml.load(file, Loader=UniqueKeyLoader)
    # Load Products:
    with open(os.path.join(data_in_path,  project[input_str]), 'r', encoding='utf-8') as file:
        input_dict = yaml.load(file, Loader=UniqueKeyLoader)

    return input_dict

def calculate_weighted_mean(timeseries_dict:dict, product_or_process, weights_dict,
                       product_or_process_position:int=0, t_position:int = 1):
    """ get the weightes mean of a timeseries in dict format.
    Is used for timeseries scaling.

    Args:
        timeseries_dict (dict): subdict from unscaled_input_dict,
            e.g., unscaled_input_dict['demand']
        product_or_process (str): name of the product or process
            e.g., 'cold' or 'PV'
        weights_dict (dict): time_slice_weight generated by AggregatedInputDictGenerator,
            e.g., {1: 2000, 2: 3760, 3: 3000} for 3 timesteps
        product_or_process_position (int, optional): position of the product/process index
            in the timeseries_dicts indices. Defaults to 0.
        t_position (int, optional): _position of the timestep index
            in the timeseries_dicts indices. Defaults to 1.

    Returns:
        float: Weighted mean
    """
    my_sum = sum(
        timeseries_dict[index] * weights_dict[index[t_position]]
        for index in timeseries_dict
        if index[product_or_process_position] == product_or_process
    )

    n_values = sum(weights_dict.values())

    return my_sum / n_values

def get_process_names(data_in_path:str, project_file:str, only_feasibility_affecting:bool,
                      storage_or_production:str='both'):
    """ Get all process names from the superstructure file.

    Args:
        data_in_path (str): path to the input files, where the project file lies
        project_file (str): name of the .yaml file in data_in_path containing the input information
        only_feasibility_affecting (bool): If True, only processes that affect feasibility are
            returned.
        storage_or_production (str): 'storage', 'production', or 'both'

    Returns:
        list: List of process names that are part of the superstructure.
    """
    assert storage_or_production in {'storage', 'production', 'both'}

    # Get superstructure and design
    config = get_config_file(data_in_path, project_file)

    superstructure = get_input_file(
        data_in_path, project_file, 'superstructures'
        )['superstructures'][config['superstructure_name']]

    technologies = get_input_file(data_in_path, project_file, 'technologies')['technologies']
    # Initialize empty list
    processes = []

    # Loop over all entries:
    for process, process_dict in superstructure.items():

        technology_dict = technologies[process_dict['type']]
        # Skip, if no effect on feasiblity and 'only_feasibility_affecting' is True
        if only_feasibility_affecting and not technology_dict['affect_feasibility']:
            continue

        # Skip production processes, if only storage processes are wanted
        if storage_or_production == 'storage' and technology_dict['category'] == 'production':
            continue

        # Skip production processes, if only storage processes are wanted
        if storage_or_production == 'production' and technology_dict['category'] == 'storage':
            continue

        processes.append(process)

    return processes

def set_rwth_color(name, brightness=100):
    """ Return color according to RWTH Aachen University corporate design.

    Args:
        name (string): String indicating name of color
        brightness (float, optional): Brightness of color, i.e. 10, 25, 50, 75, 100.
    
    Returns:
        str: Hex-code of color
    """
    if name == 'blue':
        if brightness == 100:
            return '#00549F'
        elif brightness == 75:
            return '#407FB7'
        elif brightness == 50:
            return '#8EBAE5'
        elif brightness == 25:
            return '#C7DDF2'
        else:
            return '#E8F1FA'

    elif name == 'black':
        if brightness == 100:
            return '#000000'
        elif brightness == 75:
            return '#646567'
        elif brightness == 50:
            return '#9C9E9F'
        elif brightness == 25:
            return '#CFD1D2'
        else:
            return '#ECEDED'

    elif name == 'magenta':
        if brightness == 100:
            return '#E30066'
        elif brightness == 75:
            return '#E96088'
        elif brightness == 50:
            return '#F19EB1'
        elif brightness == 25:
            return '#F9D2DA'
        else:
            return '#FDEEF0'

    elif name == 'yellow':
        if brightness == 100:
            return '#FFED00'
        elif brightness == 75:
            return '#FFF055'
        elif brightness == 50:
            return '#FFF59B'
        elif brightness == 25:
            return '#FFFAD1'
        else:
            return '#FFFDEE'

    elif name == 'petrol':
        if brightness == 100:
            return '#006165'
        elif brightness == 75:
            return '#2D7F83'
        elif brightness == 50:
            return '#7DA4A7'
        elif brightness == 25:
            return '#BFD0D1'
        else:
            return '#E6ECEC'

    elif name == 'cyan':
        if brightness == 100:
            return '#0098A1'
        elif brightness == 75:
            return '#00B1B7'
        elif brightness == 50:
            return '#89CCCF'
        elif brightness == 25:
            return '#CAE7E7'
        else:
            return '#EBF6F6'

    elif name == 'green':
        if brightness == 100:
            return '#57AB27'
        elif brightness == 75:
            return '#8DC060'
        elif brightness == 50:
            return '#B8D698'
        elif brightness == 25:
            return '#DDEBCE'
        else:
            return '#F2F7EC'

    elif name == 'maygreen':
        if brightness == 100:
            return '#BDCD00'
        elif brightness == 75:
            return '#D0D95C'
        elif brightness == 50:
            return '#E0E69A'
        elif brightness == 25:
            return '#F0F3D0'
        else:
            return '#F9FAED'

    elif name == 'orange':
        if brightness == 100:
            return '#F6A800'
        elif brightness == 75:
            return '#FABE50'
        elif brightness == 50:
            return '#FDD48F'
        elif brightness == 25:
            return '#FEEAC9'
        else:
            return '#FFF7EA'

    elif name == 'red':
        if brightness == 100:
            return '#CC071E'
        elif brightness == 75:
            return '#D85C41'
        elif brightness == 50:
            return '#E69679'
        elif brightness == 25:
            return '#F3CDBB'
        else:
            return '#FAEBE3'

    elif name == 'bordeaux':
        if brightness == 100:
            return '#A11035'
        elif brightness == 75:
            return '#B65256'
        elif brightness == 50:
            return '#CD8B87'
        elif brightness == 25:
            return '#E5C5C0'
        else:
            return '#F5E8E5'

    elif name == 'purple':
        if brightness == 100:
            return '#612158'
        elif brightness == 75:
            return '#834E75'
        elif brightness == 50:
            return '#A8859E'
        elif brightness == 25:
            return '#D2C0CD'
        else:
            return '#EDE5EA'

    elif name == 'lilac':
        if brightness == 100:
            return '#7A6FAC'
        elif brightness == 75:
            return '#9B91C1'
        elif brightness == 50:
            return '#BCB5D7'
        elif brightness == 25:
            return '#DEDAEB'
        else:
            return '#F2F0F7'

    else:
        print('Color not known. Use black instead.')
        return '#000000'

def sanity_check_value_of_lost_load(voll_dict:dict) -> None:
    """ Checks, if all feasibility-affecting products have identical values of lost load.
    This assumption is necessary for the current implementation TAC_woLL (= TAC - lostload * voll)

    Args:
        voll_dict (dict): contains value_of_lost_load, generated by get_value_of_lost_load()

    Raises:
        ValueError: if not all values are identical

    """

    if len(set(voll_dict.values())) != 1:
        raise ValueError(
            "function is currently only working properly if all feasibility-affecting products " +
            f"have identical value_of_lost_load.\nThis is not the case: {voll_dict}."
        )
    return
