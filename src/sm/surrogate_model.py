""" Module containing the SurrogateModel class for training and using surrogate models"""
import os
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
import joblib
import yaml
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skorch.callbacks import Callback, Checkpoint
from skorch import NeuralNetRegressor
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from core.helpers import get_potential_capacity, UniqueKeyLoader, get_data_out
from core.ts_scaling import get_ts_scaling_key
from core.plotting_helpers import binarize_overcapacity

class CustomLossFormatting(Callback):
    """ Custom callback to format the loss output at the end of each epoch."""
    def on_epoch_end(self, sm, **kwargs):
        """ Called at the end of each epoch to format and print the loss values."""
        # Get the train and valid loss from the history
        train_loss = sm.history[-1, 'train_loss']
        valid_loss = sm.history[-1, 'valid_loss']

        # Format them to display more digits or exponential notation
        formatted_train_loss = f"{train_loss:.6e}"  # Exponential notation
        formatted_valid_loss = f"{valid_loss:.6e}"  # Exponential notation

        print(
            f"Epoch {len(sm.history)}: train_loss={formatted_train_loss}, "+
            f"valid_loss={formatted_valid_loss}")

class SurrogateModel(ABC):
    """ Abstract class for surrogate models, which is used their for training and usage."""
    def __init__(self, data_in_path:str, project_file:str,input_names:list, output_name:str,
                 sm_name:str, hp_df_row:pd.DataFrame=None):
        """
        Args:
            data_in_path (str): path to the input files, where the project file lies
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
            input_names (list): names of the columns that should be used as input
            output_name (str): name of the column that should be used as output
            sm_name (str): name of the surrogate model hyperparameters in config.yaml
                (e.g. 'lost_load')
            hp_df_row (pd.DataFrame): Contains new hyperparameters which overwrite hyper-
                parameters from config.yaml. Defaults to None.
        """
        # Check that the output_name is not in the input_names
        if output_name in input_names:
            raise ValueError("output_name is in input_names")

        self.data_in_path = data_in_path
        self.project_file= project_file

        self._load_general_input_yamls(project_file)

        self.input_names_process, self.input_names_timeseries = self._categorize_input_names(
            input_names)
        helper_dict_process = build_helper_dict_process(self.technologies, self.input_names_process)
        helper_dict_ts_scaling = build_helper_dict_ts(self.ts_scaling, self.input_names_timeseries)
        self.helper_dict = helper_dict_process | helper_dict_ts_scaling

        self.output_name = output_name

        self.output_max = None # will be set when loading data or when generating training set
        self.input_process_max = np.array([
            self.helper_dict[key]['max_value']
            for key in self.input_names_process
        ])
        self.input_timeseries_max = np.array([
            self.helper_dict[key]['max_value']
            for key in self.input_names_timeseries
        ])

        self._load_general_input_yamls(project_file)

        # Set hyperparameters:
        self.hp = self._set_hyperparameters(sm_name, hp_df_row)
        print(self.hp, flush=True)
        self._hp_sanity_check()

        self.sm = None

        # Initiate place holder for train, validation and test sets:
        self.x_train = None
        self.y_train = None
        self.x_val = self._update_dtype([], unsqueeze=True)
        self.y_val = self._update_dtype([], unsqueeze=True)
        self.x_test = None
        self.y_test = None

    @abstractmethod
    def setup_sm(self, checkpoint_path:str=None) -> None:
        """ Generates the model. For NN its the neural network. For GP its the Gaussian Process.

        Args:
            checkpoint_path (str): Only relevant for SurrogateModelNN. Path where the checkpoints
                are saved. Defaults to None.
        """
        raise NotImplementedError("Subclasses must implement setup_sm()")

    @abstractmethod
    def save_model(self, output_path:str=None, filename:str=None) -> None:
        """ Saves the surrogate model's paramters.

        Args:
            output_path (str, optional): Path, where the model is saved. If None, DATA_OUT is used.
                Defaults to None.
            filename (str, optional): filename of the surrogate model. If None, a default name is
                used. Defaults to None.
        """
        raise NotImplementedError("Subclasses must implement save_model()")

    @abstractmethod
    def load_model(self, model_path:str=None, filename:str=None, load_history=False):
        """ Loads the model's parameters from the specified path.

        Args:
            model_path (str, optional): absolute path where the model data lies. Defaults to None.
            filename (str, optional): name of the surrogate data. Defaults to None.
            load_history (bool, optional): Only relevant for SurrogateModelNN. True, if the history
                should be loaded (can take some time for long training time). Defaults to False.
        """
        raise NotImplementedError("Subclasses must implement load_model()")

    @abstractmethod
    def _update_dtype(self, array:np.array, unsqueeze:bool=False):
        """ Returns the array either as np.array (for SurrogateModelGP) or as torch.tensor (for
        SurrogateModelNN)

        Args:
            array (np.array): input array to be converted
            unsqueeze (bool, optional): If the array should be squeezed. Defaults to False.

        (np.array or torch.tensor): Converted array
        """
        raise NotImplementedError("Subclasses must implement _update_dtype()")

    @abstractmethod
    def _get_default_filename(self) -> str:
        """ Generates a default filename for saving/loading the model.

        Returns:
            str: default filename
        """
        raise NotImplementedError("Subclasses must implement _get_default_filename()")

    @abstractmethod
    def _hp_sanity_check(self) -> None:
        """ Check if the hyperparameters are valid.

        Raises:
            ValueError: If the hyperparameters are not valid.
        """
        raise NotImplementedError("Subclasses must implement _hp_sanity_check()")

    def _load_general_input_yamls(self, project_file:str):
        """Loads the general input yamls from the data_in_path and stores them
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

        # Load ts_scaling:
        filename = os.path.join(self.data_in_path,  self.project['timeseries_scaling'])
        with open(filename, 'r', encoding='utf-8') as file:
            self.ts_scaling = yaml.load(
                file, Loader=UniqueKeyLoader
                )['timeseries_scaling'][self.config['timeseries_scaling']]

    def generate_train_val_test_sets(self, df_unscaled:pd.DataFrame, split_test_data:bool=True):
        """ Generates the test and training set from the unscaled data.

        Args:
            df_unscaled (pd.DataFrame): DataFrame containing the unscaled data with the input and
                output columns. Will be split into training and test set.
            split_test_data (bool, optional): If true, test data will be sampled from df_unscaled.
                if False, test data should be sampled outside of SurrogateModel. Defaults to True.
        """

        # Get outputmax from df_unscaled:
        self.output_max = df_unscaled[self.output_name].max()

        input_names = self.input_names_process + self.input_names_timeseries

        # Get the unscaled data:
        x_unscaled = df_unscaled[input_names].values
        y_unscaled = df_unscaled[self.output_name].values

        # Scale the data:
        x_scaled = x_unscaled / [self.helper_dict[key]['max_value'] for key in input_names]
        y_scaled = y_unscaled  / self.output_max

        if split_test_data: # get test_size from config
            test_size = self.hp['test_size']

            # Split into train and test
            x_train_val, x_test, y_train_val, y_test = train_test_split(
                x_scaled, y_scaled,
                test_size=test_size,
                random_state=self.config['seed']
        )
        else:
            test_size = 0
            x_train_val = x_scaled
            y_train_val = y_scaled
            x_test = []
            y_test = []

        # Calculate proportion for val size relative to temp
        val_proportion = self.hp['val_size'] / (1 - test_size)

        # Split train+val into train and val
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val,
            test_size=val_proportion,
            random_state=self.config['seed']
        )

        # Convert to float32 for compatibility with PyTorch
        self.x_train = self._update_dtype(x_train, unsqueeze=False)
        self.x_val   = self._update_dtype(x_val,   unsqueeze=False)
        self.x_test  = self._update_dtype(x_test,  unsqueeze=False)
        self.y_train = self._update_dtype(y_train, unsqueeze=True)
        self.y_val   = self._update_dtype(y_val,   unsqueeze=True)
        self.y_test  = self._update_dtype(y_test,  unsqueeze=True)

        print("Generated training, validation, and test sets.")

    def train_model(self, checkpoint_path:str= None):
        """ Trains the model.

        Args:
            callpack_path (str, optional): path, where the checkpoints are saved.
                If None: checkpoints are saved to DATA_OUT. Defaults to None.
        """
        # If no sm was initialized, initialize the sm before:
        if self.sm is None:
            print("Initalizing self.sm...")
            self.setup_sm(checkpoint_path=checkpoint_path)
        self.sm.fit(X=self.x_train, y=self.y_train)

    def predict(self, df_unscaled_predict:pd.DataFrame):
        """ Predicts the output for the given unscaled input data."""
        # Get scaled pytorch tensor:
        x_in_scaled = self.get_x_in_scaled(df_unscaled_predict)

        # Predict output with self.sm:
        y_predicted_scaled = self.sm.predict(x_in_scaled)

        # Return rescaled predicted output:
        return y_predicted_scaled * self.output_max

    def get_x_in_scaled(self, df_in:pd.DataFrame):
        """ Scales the input data and returns it as a PyTorch tensor.

        Args:
            df_in (pd.DataFrame): DataFrame containing the input data to be scaled.

        Returns:
            torch.tensor: Scaled input data as a PyTorch tensor.
        """
        input_names = self.input_names_process + self.input_names_timeseries

        # Get numeric values:
        x_in_unscaled = df_in[input_names].values

        # Warn, if the maximum value in the inputs is < 1, as this indicates the use of scaled data:
        if x_in_unscaled.max() < 1:
            warnings.warn(
                f"The input data has a maximum value of {x_in_unscaled.max()} < 1. "+
                "This indicates already scaled data. "+
                "However, the intended use of SurrogateModel.predict() is with unscaled data."
                )

        # Scale the data:
        x_in_scaled = x_in_unscaled / [self.helper_dict[key]['max_value'] for key in input_names]

        # Convert to float32 for compatibility with PyTorch
        return torch.tensor(x_in_scaled, dtype=torch.float32)

    def _set_hyperparameters(self, sm_name:str, hp_df_row:pd.DataFrame=None) -> dict:
        """ Overwrites hyperparameters from config.yaml with hyperparameters
        given in hp_series

        Args:
            sm_name (str): Name of the surrogate model in config.yaml
            hp_df_row (pd.DataFrame, optional): Contains the new hyperparameters. 
                Defaults to None.

        Returns:
            dict: Contains the updated hyperparameters.
        """

        # Get hyperparameters from config file:
        hp_config = deepcopy(self.config['surrogate_models'][sm_name])

        # Return hp_config, if no hp_series is given:
        if hp_df_row is None:
            return hp_config

        # Check, that hp_df_row is a pd.DataFrame:
        if not isinstance(hp_df_row, pd.DataFrame):
            raise TypeError(f"Expected pd.Series or None, got {type(hp_df_row)} instead.")
        # Check, that hp_df_row has only one row:
        if len(hp_df_row) != 1:
            raise ValueError(f"hp_df_row must only have 1 row, got {hp_df_row} instead.")

        # Get all keys from hp_series which also appear in the config hps:
        hp_keys = [
            key for key in hp_df_row.columns
            if key in hp_config]

        # Overwrite the values for the respective keys,
        for hp_key in hp_keys:
            hp_config[hp_key] = hp_df_row[hp_key][0]

        # Report, which keys are overwritten
        print(f"Overwriting {hp_keys}.")
        # Report, which indices from hp_series where not used:
        unused_indexes = [i for i in hp_df_row.columns if i not in hp_keys]
        print(f"Did not use {unused_indexes} from hp_series.")

        return hp_config

    def _categorize_input_names(self, input_names) -> tuple:
        """ Splits the input names into input names for processes and input names for timeseries

        Returns:
            (list, list): (list of the process names, list of the timeseries names)
        """
        # First, filter out capacities:
        input_names_process = [
            name for name in input_names if name in self.technologies
        ]

        # Then, assume that the remaining input names are timeseries inputs:
        input_names_timeseries = [
            name for name in input_names if name not in input_names_process
        ]

        return input_names_process, input_names_timeseries

class SurrogateModelNN(SurrogateModel):
    """ Class for neural network surrogate models, which is used their for training and usage."""

    def save_model(self, output_path:str=None, filename:str=None) -> None:
        if output_path is None:
            output_path = get_data_out(self.data_in_path, self.project_file)

        if filename is None:
            filename = self._get_default_filename()

        os.makedirs(output_path, exist_ok=True)

        self.sm.save_params(
            f_params   =os.path.join(output_path, f"{filename}_param.pkl"),
            f_optimizer=os.path.join(output_path, f"{filename}_optimizer.pkl"),
            f_criterion=os.path.join(output_path, f"{filename}_criterion.pkl"),
            f_history  =os.path.join(output_path, f"{filename}_history.pkl"))

        # Save output_max as yaml file:
        with open(os.path.join(output_path, f"{filename}_input_output.yaml"),
                  'w', encoding = 'utf-8') as f:
            input_output_dict = {
                'input_names_process'    : self.input_names_process,
                'input_names_timeseries' : self.input_names_timeseries,
                'input_process_max'      : self.input_process_max.tolist(),
                'input_timeseries_max'   : self.input_timeseries_max.tolist(),
                'output_name'            : self.output_name,
                'output_max'             : self.output_max.tolist()
            }
            yaml.dump(input_output_dict, f)

        print(f"Saved model to {os.path.join(output_path, filename)}.")

    def load_model(self, model_path:str=None, filename:str=None, load_history=False):
        # If no sm was initialized, initialize the sm before:
        if self.sm is None:
            print("initalizing self.sm")
            self.setup_sm()

        if model_path is None:
            model_path = get_data_out(self.data_in_path, self.project_file)

        if filename is None:
            filename = self._get_default_filename()

        f_history = os.path.join(model_path, f"{filename}_history.pkl") if load_history else None
        self.sm.load_params(
            f_params   =os.path.join(model_path, f"{filename}_param.pkl"),
            f_optimizer=os.path.join(model_path, f"{filename}_optimizer.pkl"),
            f_criterion=os.path.join(model_path, f"{filename}_criterion.pkl"),
            f_history  =f_history)

        # Load output_max from yaml file:
        filename_complete = os.path.join(model_path, f"{filename}_input_output.yaml")
        with open(filename_complete, 'r', encoding='utf-8') as f:
            input_output_dict = yaml.load(f, Loader=UniqueKeyLoader)

        self.output_max = input_output_dict['output_max']

        # Sanity check:
        if self.output_name != input_output_dict['output_name']:
            raise ValueError(
                f"output_name does not match: expected '{self.output_name}', "
                f" but got '{input_output_dict['output_name']}'."
            )

        if self.input_names_process != input_output_dict['input_names_process']:
            raise ValueError(
                f"input_names_process does not match: expected '{self.input_names_process}', " +
                f"but got '{input_output_dict['input_names_process']}'."
            )

        if self.input_names_timeseries != input_output_dict['input_names_timeseries']:
            raise ValueError(
                f"input_names_timeseries does not match: expected '{self.input_names_timeseries}'" +
                f", but got '{input_output_dict['input_names_timeseries']}'."
            )

        if (self.input_process_max != np.array(input_output_dict['input_process_max'])).any():
            raise ValueError(
                f"input_process_max does not match: expected '{self.input_process_max}', " +
                f"but got '{input_output_dict['input_process_max']}'."
            )

        if (self.input_timeseries_max != np.array(input_output_dict['input_timeseries_max'])).any():
            raise ValueError(
                f"input_timeseries_max does not match: expected '{self.input_timeseries_max}', " +
                f"but got '{input_output_dict['input_timeseries_max']}'."
            )

    def _update_dtype(self, array:np.array, unsqueeze:bool=False) -> torch.tensor:
        if unsqueeze:
            return torch.tensor(array, dtype=torch.float32).unsqueeze(1)

        # Else
        return torch.tensor(array, dtype=torch.float32)

    def _get_default_filename(self) -> str:
        return (
            f"model_{self.output_name}_{self.hp['N_layers']}HL_" +
            f"{self.hp['N_nodes']}N_{self.hp['N_epochs']}E.pkl"
            )

    def setup_sm(self, checkpoint_path:str=None) -> None:
        # Setup model:
        nn_model = self._setup_model()

        callbacks =[CustomLossFormatting()]

        # Create a Dataset for validation data
        val_ds = Dataset(self.x_val, self.y_val)

        # Activate checkpointing, if config says so:
        if self.hp.get('checkpoints', False):
            if checkpoint_path is None:
                checkpoint_path = get_data_out(self.data_in_path, self.project_file)
            else:
                os.makedirs(checkpoint_path, exist_ok=True)

            def monitor(sm):
                return sm.history[-1, 'valid_loss_best']

            callbacks.append(Checkpoint(
                monitor=monitor(self.sm),
                dirname=checkpoint_path,
                f_params="params_{last_epoch[epoch]}.pkl",
                f_model="model.pkl",
                f_criterion="criterion.pkl",
                f_history="history.pkl",
                ))

        # Initialize the NeuralNetRegressor with the Sequential model
        self.sm = NeuralNetRegressor(
            nn_model,
            max_epochs              = int(self.hp['N_epochs']), # Number of epochs
            lr                      = self.hp['lr'],        # Learning rate
            batch_size              = int(self.hp['batch_size']),
            criterion               = nn.MSELoss,
            optimizer__weight_decay = self.hp['weight_decay'],
            optimizer               = torch.optim.Adam,
            iterator_train__shuffle = True,  # Shuffle the training data
            callbacks               = callbacks,
            # Use GPU if available, otherwise CPU:
            device                  = "cuda" if torch.cuda.is_available() else "cpu",
            # Tells Skorch to use the predefined valdiation data to calculate the validation loss:
            train_split             = predefined_split(val_ds),
        )

        print(
            "------ Calculations use GPU! ------"
            if torch.cuda.is_available() else "------ Calculations use CPU! ------")

        self.sm.initialize()

    def _setup_model(self) -> nn.Sequential:
        """ Sets up the Sequential model for regression

        Returns:
            nn.Sequential(): generated model
        """

        n_nodes = self.hp['N_nodes']

        # Initialize model:
        nn_model = nn.Sequential()

        # Add input layer:
        nn_model.add_module(
            'linear_in',
            nn.Linear(len(self.input_names_process + self.input_names_timeseries), n_nodes)
            )
        if self.hp['act_fun'] == 'ReLU':
            nn_model.add_module('relu_in', nn.ReLU())

        # Add hidden layers:
        for i in range(self.hp['N_layers']):
            nn_model.add_module(f'linear_{i}', nn.Linear(n_nodes, n_nodes))
            if self.hp['act_fun'] == 'ReLU':
                nn_model.add_module(f'relu_{i}', nn.ReLU())

        # Add output layer:
        nn_model.add_module('linear_out', nn.Linear(n_nodes, 1))

        # Add ReLU to last layer
        if self.hp['act_out'] == 'ReLU':
            nn_model.add_module('relu_out', nn.ReLU())
            print("output ReLU added")

        return nn_model

    def _hp_sanity_check(self) -> None:
        # Check, that valid activation function is used:
        allowed_activation_functions = ['ReLU', 'Linear']
        if self.hp['act_fun'] not in allowed_activation_functions:
            raise ValueError(
                f"Activation function '{self.hp['act_fun']}' not in {allowed_activation_functions}"
                )

        # Check, that the last layer activation function is either 'ReLU' or None
        if self.hp['act_out'] not in ['ReLU', 'None']:
            raise ValueError(
                f"Output activation function '{self.hp['act_out']}' not in ['ReLU', None]"
                )

    def plot_training_loss(self, ax=None) -> plt.Axes:
        """ Plots the training and validation loss over epochs."""
        # plot training and validation loss
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.sm.history[:, 'train_loss'], label='train loss')
        ax.plot(self.sm.history[:, 'valid_loss'], label='valid loss')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        return ax

class SurrogateModelNN_Binary(SurrogateModelNN):
    """
    Neural-network surrogate model for binary feasibility prediction.

    Uses BCEWithLogitsLoss:
        - model output = raw logits
        - sigmoid is applied only during prediction
        - target values must be binary: 0/1, bool, or feasible/infeasible strings
    """

    def generate_train_val_test_sets(
        self,
        df_unscaled: pd.DataFrame,
        split_test_data: bool = True
    ):
        """
        Generates train, validation, and test sets for binary classification.

        Unlike regression, the binary output is not scaled by output_max.
        """

        # Keep output_max only for compatibility with inherited save/load logic.
        # It is not used for prediction.
        self.output_max = np.array(1.0)

        input_names = self.input_names_process + self.input_names_timeseries

        # Get unscaled inputs
        x_unscaled = df_unscaled[input_names].values

        # Convert labels to binary float values
        if self.output_name == 'overcapacity':
            y_binary = binarize_overcapacity(df_unscaled[self.output_name])
        else:
            raise NotImplementedError("Currently, only 'overcapacity' is supported as output_name for SurrogateModelNN_Binary.")

        # Scale inputs only
        x_scaled = x_unscaled / [
            self.helper_dict[key]["max_value"]
            for key in input_names
        ]

        if split_test_data:
            test_size = self.hp["test_size"]

            x_train_val, x_test, y_train_val, y_test = train_test_split(
                x_scaled,
                y_binary,
                test_size=test_size,
                random_state=self.config["seed"],
                stratify=y_binary if len(np.unique(y_binary)) == 2 else None,
            )
        else:
            test_size = 0
            x_train_val = x_scaled
            y_train_val = y_binary
            x_test = []
            y_test = []

        val_proportion = self.hp["val_size"] / (1 - test_size)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=val_proportion,
            random_state=self.config["seed"],
            stratify=y_train_val if len(np.unique(y_train_val)) == 2 else None,
        )

        self.x_train = self._update_dtype(x_train, unsqueeze=False)
        self.x_val   = self._update_dtype(x_val,   unsqueeze=False)
        self.x_test  = self._update_dtype(x_test,  unsqueeze=False)

        self.y_train = self._update_dtype(y_train, unsqueeze=True)
        self.y_val   = self._update_dtype(y_val,   unsqueeze=True)
        self.y_test  = self._update_dtype(y_test,  unsqueeze=True)

        print("Generated binary training, validation, and test sets.")

    def setup_sm(self, checkpoint_path: str = None) -> None:
        """
        Sets up the neural network using BCEWithLogitsLoss.
        """

        nn_model = self._setup_model()

        callbacks = [CustomLossFormatting()]

        val_ds = Dataset(self.x_val, self.y_val)

        if self.hp.get("checkpoints", False):
            if checkpoint_path is None:
                checkpoint_path = get_data_out(self.data_in_path, self.project_file)
            else:
                os.makedirs(checkpoint_path, exist_ok=True)

            callbacks.append(
                Checkpoint(
                    monitor="valid_loss_best",
                    dirname=checkpoint_path,
                    f_params="params_{last_epoch[epoch]}.pkl",
                    f_model="model.pkl",
                    f_criterion="criterion.pkl",
                    f_history="history.pkl",
                )
            )

        self.sm = NeuralNetRegressor(
            nn_model,
            max_epochs=int(self.hp["N_epochs"]),
            lr=self.hp["lr"],
            batch_size=int(self.hp["batch_size"]),
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=self.hp["weight_decay"],
            iterator_train__shuffle=True,
            callbacks=callbacks,
            device="cuda" if torch.cuda.is_available() else "cpu",
            train_split=predefined_split(val_ds),
        )

        print(
            "------ Calculations use GPU! ------"
            if torch.cuda.is_available()
            else "------ Calculations use CPU! ------"
        )

        self.sm.initialize()

    def _setup_model(self) -> nn.Sequential:
        """
        Sets up the Sequential model for binary classification.

        Important:
        No sigmoid is added here because BCEWithLogitsLoss expects raw logits.
        """

        n_nodes = int(self.hp["N_nodes"])
        n_inputs = len(self.input_names_process + self.input_names_timeseries)

        nn_model = nn.Sequential()

        nn_model.add_module(
            "linear_in",
            nn.Linear(n_inputs, n_nodes)
        )

        if self.hp["act_fun"] == "ReLU":
            nn_model.add_module("relu_in", nn.ReLU())

        for i in range(int(self.hp["N_layers"])):
            nn_model.add_module(f"linear_{i}", nn.Linear(n_nodes, n_nodes))

            if self.hp["act_fun"] == "ReLU":
                nn_model.add_module(f"relu_{i}", nn.ReLU())

        # Raw logit output
        nn_model.add_module("linear_out", nn.Linear(n_nodes, 1))

        return nn_model

    def predict_logits(self, df_unscaled_predict: pd.DataFrame) -> np.ndarray:
        x = self.get_x_in_scaled(df_unscaled_predict)

        self.sm.module_.eval()

        with torch.no_grad():
            logits = self.sm.module_(x.to(self.sm.device))

        return logits.detach().cpu().numpy().reshape(-1)

    def predict_proba(self, df_unscaled_predict: pd.DataFrame) -> np.ndarray:
        logits = self.predict_logits(df_unscaled_predict)
        logits = torch.tensor(logits, dtype=torch.float32)

        return torch.sigmoid(logits).numpy()

    def predict(
        self,
        df_unscaled_predict: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predicts binary feasibility labels.

        Returns:
            np.ndarray: 1 = feasible, 0 = infeasible
        """

        probabilities = self.predict_proba(df_unscaled_predict)

        return (probabilities >= threshold).astype(int)

    def _get_default_filename(self) -> str:
        return (
            f"model_binary_{self.output_name}_{self.hp['N_layers']}HL_"
            f"{self.hp['N_nodes']}N_{self.hp['N_epochs']}E.pkl"
        )

    def _hp_sanity_check(self) -> None:
        allowed_activation_functions = ["ReLU", "Linear"]

        if self.hp["act_fun"] not in allowed_activation_functions:
            raise ValueError(
                f"Activation function '{self.hp['act_fun']}' not in "
                f"{allowed_activation_functions}"
            )

        # For BCEWithLogitsLoss, the output must be raw logits.
        # Therefore, no output activation should be used.
        if self.hp.get("act_out", "None") not in ["None", None]:
            raise ValueError(
                "SurrogateModelNN_Binary uses BCEWithLogitsLoss and therefore "
                "requires raw logits. Set act_out to 'None'."
            )

    def plot_training_loss(self, ax=None) -> plt.Axes:
        """ Plots the training and validation loss over epochs."""
        # plot training and validation loss
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.sm.history[:, 'train_loss'], label='train loss')
        ax.plot(self.sm.history[:, 'valid_loss'], label='valid loss')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        return ax

class SurrogateModelGP(SurrogateModel):
    """ Class for gaussian process surrogate models, which is used their for training and usage."""

    def save_model(self, output_path:str=None, filename:str=None):

        if output_path is None:
            output_path = get_data_out(self.data_in_path, self.project_file)

        if filename is None:
            filename = self._get_default_filename()

        os.makedirs(output_path, exist_ok=True)

        gp_file = os.path.join(output_path, f"{filename}_gp.pkl")
        joblib.dump(self.sm, gp_file)

        # Save output_max as yaml file:
        with open(os.path.join(output_path, f"{filename}_input_output.yaml"),
                  'w', encoding = 'utf-8') as f:
            input_output_dict = {
                'input_names_process'    : self.input_names_process,
                'input_names_timeseries' : self.input_names_timeseries,
                'input_process_max'      : self.input_process_max.tolist(),
                'input_timeseries_max'   : self.input_timeseries_max.tolist(),
                'output_name'            : self.output_name,
                'output_max'             : float(self.output_max)
            }
            yaml.dump(input_output_dict, f)

        print(
            f"Saved GP model to {gp_file} and metadata to " +
            f"{os.path.join(output_path, f'{filename}_input_output.yaml')}.")

    def load_model(self, model_path:str=None, filename:str=None, load_history=False):

        if load_history:
            raise ValueError(
                f"'load_history' is '{load_history}', but must be 'False' for SurrogateModelGP.")

        if model_path is None:
            model_path = get_data_out(self.data_in_path, self.project_file)

        if filename is None:
            filename = self._get_default_filename()

        self.sm = joblib.load(os.path.join(model_path, f"{filename}_gp.pkl"))

        # Load output_max from yaml file:
        with open(os.path.join(model_path, f"{filename}_input_output.yaml"),'r',encoding='utf-8') as f:
            input_output_dict = yaml.load(f, Loader=UniqueKeyLoader)

        self.output_max = input_output_dict['output_max']

        # Sanity check:
        if self.output_name != input_output_dict['output_name']:
            raise ValueError(
                f"output_name does not match: expected '{self.output_name}', "
                f" but got '{input_output_dict['output_name']}'."
            )

        if self.input_names_process != input_output_dict['input_names_process']:
            raise ValueError(
                f"input_names_process does not match: expected '{self.input_names_process}', " +
                f"but got '{input_output_dict['input_names_process']}'."
            )

        if self.input_names_timeseries != input_output_dict['input_names_timeseries']:
            raise ValueError(
                f"input_names_timeseries does not match: expected '{self.input_names_timeseries}'" +
                f", but got '{input_output_dict['input_names_timeseries']}'."
            )

        if (self.input_process_max != np.array(input_output_dict['input_process_max'])).any():
            raise ValueError(
                f"input_process_max does not match: expected '{self.input_process_max}', " +
                f"but got '{input_output_dict['input_process_max']}'."
            )

        if (self.input_timeseries_max != np.array(input_output_dict['input_timeseries_max'])).any():
            raise ValueError(
                f"input_timeseries_max does not match: expected '{self.input_timeseries_max}', " +
                f"but got '{input_output_dict['input_timeseries_max']}'."
            )

    def _update_dtype(self, array:np.array, unsqueeze:bool=False) -> np.array:
        if unsqueeze:
            return np.asarray(array, dtype=np.float64).ravel()

        # Else
        return np.asarray(array, dtype=np.float64)

    def _get_default_filename(self) -> str:
        return (
            f"model_{self.output_name}_{self.hp['N_layers']}HL_" +
            f"{self.hp['N_nodes']}N_{self.hp['N_epochs']}E.pkl"
            )

    def setup_sm(self, checkpoint_path:str=None):

        # Make sure that checkpoint_path (which is only required for SurrogateModelNN) is None:
        if checkpoint_path is not None:
            raise ValueError(f"Expected checkpoint_path to be None, but it is '{checkpoint_path}'.")

        # Setup model:
        if self.hp['kernel'] != 'Matern':
            raise ValueError(f"Unknown Kernel: {self.hp['kernel']}")

        length_scale = self.hp['length_scale']
        nu = self.hp['nu']
        kernel = Matern(length_scale=length_scale, nu=nu)
        alpha = self.hp['alpha']

        self.sm = GaussianProcessRegressor(
            kernel=kernel,
            optimizer="fmin_l_bfgs_b",
            normalize_y=False,
            alpha=alpha)

    def _hp_sanity_check(self):
        # Check, that valid kernel is used:
        allowed_kernels = ['Matern']
        if self.hp['kernel'] not in allowed_kernels:
            raise ValueError(
                f"Kernel '{self.hp['kernel']}' not in {allowed_kernels}"
                )

def build_helper_dict_process(technologies:dict, input_names_process:list) -> dict:
    """Build the helper dict for process input data. The helper dict looks like:
    helper_dict_ts = {
        'demand_electricity_mean': {
            'type': 'process_capacity',
            'max_value': 500},
        ...}

    Args:
        technologies (dict): technology dict from technologies.yaml
        input_names_process (list): list of the process names used as input

    Returns:
        dict: helper dict for the surrogate models.
    """

    # Build helper dict for processes:
    return  {
        key: {
            'type'      : 'process_capacity',
            'max_value' : get_potential_capacity(technologies[key])
        }
        for key in input_names_process
    }

def build_helper_dict_ts(ts_scaling:dict, input_names_timeseries: list) -> dict:
    """ Build the helper dict for timeseries input data. The helper dict looks like:
    helper_dict_ts = {
        'demand_electricity_mean': {
            'ts_type': 'demand',
            'product_or_process': 'electricity',
            'mean_or_peak': 'mean',
            'max_value' : 1000},
        ...
        'availability_PV_mean': {
            'ts_type': 'availability',
            'product_or_process': 'PV',
            'mean_or_peak': 'mean',
            'max_value': 1}
    }

    Args:
        ts_scaling (dict): timeseries scaling dict from ts_scaling.yaml
        input_names_timeseries (list): list of the timeseries scaling names used as input

    Returns:
        dict: helper_dict_ts
    """
    helper_dict_ts = {
        key: {'type' : 'timeseries'}
        for key in input_names_timeseries
    }

    # Initiliaze sets needed for Sanity check
    ts_types = {value['ts_type'] for value in ts_scaling.values()}

    for key, ts_dict in helper_dict_ts.items():
        key_split = key.split('_')

        # Save ts_type at position 0 ('demand', 'pricebuy', ...):
        ts_dict['ts_type'] = key_split[0]
        # Save product_or_process at position 1 ('electricity', 'PV', ...):
        ts_dict['product_or_process'] = key_split[1]
        # Save mean_or_peak at position 2:
        ts_dict['mean_or_peak'] = key_split[2]

        # Add timeseries scaling key (key in timeseries_scaling.yaml)
        ts_dict['ts_scaling_key'] = get_ts_scaling_key(ts_dict, ts_scaling)
        # Add max value from timeseries_scaling.yaml:
        ts_dict['max_value'] = ts_scaling[
            ts_dict['ts_scaling_key']
            ]['param'][ts_dict['mean_or_peak']]['max']

        # Sanity checks
        if ts_dict['ts_type'] not in ts_types:
            raise ValueError(f"Unknown ts_type '{ts_dict['ts_type']}'.")
        if ts_dict['mean_or_peak'] not in {'mean', 'peak', 'relpeak', 'interpolate'}:
            raise ValueError(
                "Expected 'mean', 'peak', 'relpeak', or 'interpolate'. "+
                f"Got '{ts_dict['mean_or_peak']}' instead."
                )
        if len(key_split) != 3:
            raise ValueError(f"key name {key} does not confirm the name convention.")

    return helper_dict_ts
