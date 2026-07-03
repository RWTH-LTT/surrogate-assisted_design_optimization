""" Module containing the Hybrid Surrogate Model class for surrogate assisted design optimization"""
import os
from abc import ABC, abstractmethod
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy.stats import norm
from core.helpers import get_potential_capacity, UniqueKeyLoader
from sm.surrogates_feasible_space import Feasibility_LP
from sm.surrogate_model import SurrogateModelNN, SurrogateModelGP, SurrogateModel

####################################################################################################
#                                  COST PREDICTOR                                                  #
####################################################################################################

class CostPredictor(ABC):
    """ Abstract base class for cost predictors.
    The cost predictor calculates the total annual cost (TAC) and its gradient.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model:SurrogateModel):
        """
        Args:
            data_in_path (str): path to the input files, where the project file lies
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
            surrogate_model (SurrogateModel): surrogate model to predict OPEX
        """
        self.data_in_path    = data_in_path
        self.project_file    = project_file
        self.surrogate_model = surrogate_model

        self._load_general_input_yamls(project_file)

        # --- Add process sets: ---
        self.processes         = self.surrogate_model.input_names_process
        # --- Add timeseries sets: ---
        self.timeseries        = self.surrogate_model.input_names_timeseries

        # Add numpy arrays for invest cost, maintenance cost, and apvf:
        self._add_apvf_to_technologies()
        self.invest_cost, self.maintenance_cost, self.apvf = self._add_economic_arrays()

    def calc_capex(self, x_cap:np.array) -> float:
        """ Calculates the capital expenditure (capex) based on the given capacities.

        Args:
            x_cap (np.array): Array of capacities for each process.

        Returns:
            float: The total capital expenditure.
        """
        # Calculate capex:
        return sum(x_cap * self.invest_cost * (
                1 / self.apvf
                + self.maintenance_cost / 100
            )
        )

    def calc_capex_grad(self) -> np.array:
        """ Calculates the gradient of the capital expenditure (capex) w.r.t. the capacities.

        Returns:
            np.array: The gradient of the total capital expenditure w.r.t. the capacities.
        """
        # Just return the constant coefficient vector (independent of x_cap)
        return self.invest_cost * (
            1 / self.apvf + self.maintenance_cost / 100
        )

    def _load_general_input_yamls(self, project_file:str) -> None:
        """ Loads the general input yamls from the data_in_path and stores them
        in the class attributes.

        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
        """

        # Load project file:
        filename = os.path.join(self.data_in_path, project_file)
        with open(filename, 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)

        # Load config:
        filename = os.path.join(self.data_in_path,  self.project['config'])
        with open(filename, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=UniqueKeyLoader)

        # Load general parameters:
        filename = os.path.join(self.data_in_path,  self.project['general_param'])
        with open(filename, 'r', encoding='utf-8') as f:
            self.general_params = yaml.load(f, Loader=UniqueKeyLoader)

        # Load technology data:
        filename = os.path.join(self.data_in_path,  self.project['technologies']) 
        with open(filename, 'r', encoding='utf-8') as f:
            self.technologies = yaml.load(f, Loader=UniqueKeyLoader)['technologies']

    def _add_economic_arrays(self) -> tuple:
        """ Adds numpy arrays for invest cost, maintenance cost,
        and annualized present value factor (apvf)

        Returns:
            tuple: (invest_cost, maintenance_cost, apvf)
        """
        invest_cost = np.array([
            self.technologies[process].get(
                'investment_cost_EUR_kW', 
                self.technologies[process].get('investment_cost_EUR_kWh'))
            for process in self.processes
        ])

        maintenance_cost = np.array([
            self.technologies[process]['maintenance_cost_percent']
            for process in self.processes
        ])

        apvf = np.array([
            self.technologies[process]['apvf']
            for process in self.processes
        ])

        return invest_cost, maintenance_cost, apvf

    def _add_apvf_to_technologies(self) -> None:
        """
        Adds the annualized present value factor (apvf) to each technology in self.technologies.
        """
        discount_rate = self.general_params['discount_rate_percent']

        for process in self.processes:
            self.technologies[process]['apvf'] = (
                ( (discount_rate + 1) ** self.technologies[process]['lifetime_a'] - 1)
                /
                (discount_rate * (discount_rate + 1)**self.technologies[process]['lifetime_a'])
            )

class CostPredictorNN(CostPredictor):
    """ Cost predictor using a neural network surrogate model to predict OPEX.
    Inherits from CostPredictor.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model:SurrogateModelNN):

        super().__init__(data_in_path, project_file, surrogate_model)

        # Sanity check: feasibility_predictor must be a Surrogate Model
        if not isinstance(surrogate_model, SurrogateModelNN):
            raise TypeError(
                "Expected type(surrogate_model) to be SurrogateModelNN. Instead it is " +
                f"'{type(surrogate_model)}'."
            )

    def calc_tac_and_grad(self, x_cap_scaled:np.array, x_ts_scaled:np.array) -> tuple:
        """ Calculates the total annual cost (tac) and its gradient with respect to the capacities.

        Args:
            x_cap_scaled (np.array): process capacities (controlled by the optimization).
                Scaled to O(1) by deviding by process_max
            x_ts (np.array): timeseries values (fixed parameters for the optimization).
                x_ts is unscaled.

        Returns:
            tuple: A tuple containing the total annual cost (tac) and its gradient with respect
                to the capacities.
        """
        # Get scaling info
        x_cap_max = self.surrogate_model.input_process_max
        y_max     = self.surrogate_model.output_max

        # Concatenate and wrap in autograd tensor
        x_cap_scaled_torch = torch.tensor(
            x_cap_scaled, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        x_ts_scaled_torch = torch.tensor(x_ts_scaled, dtype=torch.float32).unsqueeze(0)
        input_torch = torch.cat([x_cap_scaled_torch, x_ts_scaled_torch], dim=1)

        # Get PyTorch model
        model = self.surrogate_model.sm.module_

        # Forward pass through ANN to get normalized OPEX
        opex_scaled = model(input_torch).squeeze()  # scalar

        # Backward pass to get gradient of OPEX wrt scaled x
        grad_total, = torch.autograd.grad(
            outputs=opex_scaled,
            inputs=x_cap_scaled_torch,
            retain_graph=False,
            create_graph=False
        )
        grad_opex_scaled = grad_total.detach().numpy().squeeze()

        # CAPEX (true function)
        x_cap = x_cap_scaled * x_cap_max
        capex = self.calc_capex(x_cap)      # scalar
        grad_capex = self.calc_capex_grad() # shape (x_cap,)

        capex_scaled = capex / y_max
        grad_capex_scaled = grad_capex * x_cap_max / y_max

        # Combine
        tac_scaled = capex_scaled + opex_scaled.item()
        grad_tac_scaled = grad_capex_scaled + grad_opex_scaled

        # Return tac, grad_tac
        return tac_scaled, grad_tac_scaled

class CostPredictorGP(CostPredictor):
    """ Cost predictor using a Gaussian Process surrogate model to predict OPEX.
    Inherits from CostPredictor.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model:SurrogateModelGP):

        super().__init__(data_in_path, project_file, surrogate_model)

        # Sanity check: feasibility_predictor must be a Surrogate Model
        if not isinstance(surrogate_model, SurrogateModelGP):
            raise TypeError(
                "Expected type(surrogate_model) to be SurrogateModelGP. Instead it is " +
                f"'{type(surrogate_model)}'."
            )

    def calc_tac(self, x_cap_scaled:np.array, x_ts_scaled:np.array) -> tuple:
        """ Calculates the total annual cost (tac) based on the given capacities.
        Not implemented yet.
        """
        raise NotImplementedError("calc_tac not yet implemented")

####################################################################################################
#                                  FEASIBILITY PREDICTOR                                           #
####################################################################################################

class FeasibilityPredictor(ABC):
    """ Abstract base class for feasibility predictors.
    The feasibility predictor provides feasibility constraints for the optimization.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model:SurrogateModel,
                 feasibility_predictor_type:str, processes_full:list, timeseries_full:list):
        """
        Args:
            data_in_path (str): path to the input files, where the project file lies
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
            surrogate_model (SurrogateModel): surrogate model to predict feasibility
            feasibility_predictor_type (str): type of feasibility predictor
            processes_full (list): list of all processes
            timeseries_full (list): list of all timeseries
        """
        self.data_in_path               = data_in_path
        self.project_file               = project_file
        self.surrogate_model            = surrogate_model
        self.feasibility_predictor_type = feasibility_predictor_type

        self._load_general_input_yamls(project_file)

        self.processes = processes_full
        self.timeseries = timeseries_full

        if self.feasibility_predictor_type == "noFP":
            self.processes_af  = []
            self.timeseries_af = []
        else:
            self.processes_af  = self.surrogate_model.input_names_process
            self.timeseries_af = self.surrogate_model.input_names_timeseries

        self.processes_af_idx  = [
            i for i, value in enumerate(self.processes)
            if value in self.processes_af]
        self.timeseries_af_idx = [
            i for i, value in enumerate(self.timeseries)
            if value in self.timeseries_af]

    @abstractmethod
    def get_feasibility_constraint(self, x_ts_scaled:np.array, oc_min:float=0, gp_conf_ub:float=None
                                   )-> list[NonlinearConstraint]:
        """ Get the feasibility constraint based on the type of feasibility predictor.

        Args:
            x_ts_scaled (np.array): timeseries values (fixed parameters for the optimization)
            oc_min (float): minimal allowed overcapacity.
                Only relevant for feasibility_type = 'overcapacity'. Defaults to 0.
            gp_conf_ub (float): minimal required confidence (upper bound) in feasibility prediction.
                gp_conf_ub must be between 0 and 1.
                gp_conf_ub = 0.5: mean prediction is sufficient -> offset = 0
                gp_conf_ub = 0.95: offset = 1.6448 (determined by scipy.stats.norm.ppf
                Only relevant for SurrogateModelGP. Defaults to None.
        Raises:
            ValueError: If the feasibility_predictor_type is not recognized.

        Returns:
            NonlinearConstraint: A constraint that ensures the feasibility of the design.
        """
        raise NotImplementedError("Subclasses must implement get_feasibility_constraint()")

    def _load_general_input_yamls(self, project_file:str) -> None:
        """ Loads the general input yamls from the data_in_path and stores them
        in the class attributes.

        Args:
            project_file (str): name of the .yaml file in data_in_path containing
                the input information (e.g. project.yaml)
        """

        # Load project file:
        filename = os.path.join(self.data_in_path, project_file)
        with open(filename, 'r', encoding='utf-8') as f:
            self.project = yaml.load(f, Loader=UniqueKeyLoader)

        # Load config:
        filename = os.path.join(self.data_in_path,  self.project['config'])
        with open(filename, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=UniqueKeyLoader)

        # Load general parameters:
        filename = os.path.join(self.data_in_path,  self.project['general_param'])
        with open(filename, 'r', encoding='utf-8') as f:
            self.general_params = yaml.load(f, Loader=UniqueKeyLoader)

        # Load technology data:
        filename = os.path.join(self.data_in_path,  self.project['technologies']) 
        with open(filename, 'r', encoding='utf-8') as f:
            self.technologies = yaml.load(f, Loader=UniqueKeyLoader)['technologies']

    def _oc_min_sanity_check(self, oc_min:float):
        """ Sanity check for oc_min values. oc_min must be non-negative. A value != 0 is only
                allowed for self.feasibility_predictor_type = 'overcapacity' 


        Args:
            oc_min (float): minimal allowed overcapacity
        """

        if oc_min < 0:
            raise ValueError(
                f"oc_min must be non-negative but is oc_min = {oc_min}."
            )

        if oc_min > 0 and self.feasibility_predictor_type != 'overcapacity':
            raise ValueError(
                "oc_min must be positive if self.feasibility_predictor_type == 'overcapacity', " +
                f"but is oc_min = {oc_min}."
            )

class FeasibilityPredictorNN(FeasibilityPredictor):
    """ Feasibility predictor using a neural network surrogate model to predict feasibility.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model,
                 feasibility_predictor_type:str, processes_full:list, timeseries_full:list):

        super().__init__(data_in_path, project_file, surrogate_model, feasibility_predictor_type,
                         processes_full, timeseries_full)

        # Sanity check: feasibility_predictor must be a Surrogate Model
        if not isinstance(surrogate_model, SurrogateModelNN):
            raise TypeError(
                "Expected type(surrogate_model) to be SurrogateModelNN. Instead it is " +
                f"'{type(surrogate_model)}'."
            )

    def get_feasibility_constraint(self, x_ts_scaled:np.array, oc_min:float=0,
                                   gp_conf_ub:float=None)-> list[NonlinearConstraint]:

        # Check that no gp_conf_ub is given:
        if gp_conf_ub is not None:
            raise ValueError(
                f"gp_conf_ub must be None for FeasibilityPredictorNN, but is '{gp_conf_ub}'."
            )
        # oc min sanity check:
        self._oc_min_sanity_check(oc_min)

        if self.feasibility_predictor_type == 'lost_load':
            return [NonlinearConstraint(
                fun=lambda x_cap: self._feasibility_and_grad(x_cap, x_ts_scaled)[0],
                jac=lambda x_cap: self._feasibility_and_grad(x_cap, x_ts_scaled)[1],
                lb=-np.inf,
                ub=0)]

        if self.feasibility_predictor_type == 'overcapacity':
            return [NonlinearConstraint(
                fun=lambda x_cap: self._feasibility_and_grad(x_cap, x_ts_scaled)[0],
                jac=lambda x_cap: self._feasibility_and_grad(x_cap, x_ts_scaled)[1],
                lb= oc_min / self.surrogate_model.output_max,
                ub= np.inf)
            ]

        raise ValueError(
            f"Invalid feasibility_predictor_type: {self.feasibility_predictor_type}. " +
            "Expected: 'lost_load', 'overcapacity'."
        )

    def _feasibility_and_grad(self, x_cap_scaled:np.array, x_ts_scaled:np.array) -> tuple:
        """ Calculates the feasibility and its gradient with respect to the capacities.

        Args:
            x_cap_scaled (np.array): process capacities (controlled by the optimization).
                Scaled to O(1) by deviding by process_max
            x_ts_scaled (np.array): timeseries values (fixed parameters for the optimization).
                x_ts_scaled is to O(1).

        Returns:
            tuple:
                np.array: feasibility value (scaled to O(1))
                np.array: gradient of the feasibility w.r.t. the capacities (scaled to O(1))
        """
        # Extract relevant subsets for feasibility
        x_cap_af_scaled = x_cap_scaled[[self.processes.index(p) for p in self.processes_af]]
        x_ts_af_scaled  = x_ts_scaled[[self.timeseries.index(t) for t in self.timeseries_af]]

        x_cap_af_scaled = np.array(list(x_cap_af_scaled))
        x_ts_af_scaled  = np.array(list(x_ts_af_scaled))

        # Torch input
        x_cap_af_tensor = torch.tensor(
            x_cap_af_scaled, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        x_ts_af_tensor  = torch.tensor(x_ts_af_scaled, dtype=torch.float32).unsqueeze(0)

        input_tensor = torch.cat([x_cap_af_tensor, x_ts_af_tensor], dim=1)

        # Forward pass
        model = self.surrogate_model.sm.module_
        feasibility_scaled = model(input_tensor).squeeze()

        # Backward pass (w.r.t. optimizer-controlled capacities only)
        grad_af_scaled, = torch.autograd.grad(
            outputs      = feasibility_scaled,
            inputs       = x_cap_af_tensor,
            retain_graph = False
        )
        grad_af_scaled = grad_af_scaled.detach().numpy().squeeze()

        # Expand gradient to full x_cap shape
        grad_full_scaled = np.zeros_like(x_cap_scaled, dtype=float)

        for i_af, process in enumerate(self.processes_af):
            idx_full = self.processes.index(process)
            grad_full_scaled[idx_full] = grad_af_scaled[i_af]

        return feasibility_scaled.item(), grad_full_scaled

class FeasibilityPredictorGP(FeasibilityPredictor):
    """ Feasibility predictor using a Gaussian Process surrogate model to predict feasibility.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model,
                 feasibility_predictor_type:str, processes_full:list, timeseries_full:list):

        super().__init__(data_in_path, project_file, surrogate_model, feasibility_predictor_type,
                         processes_full, timeseries_full)

        # Sanity check: feasibility_predictor must be a GP Surrogate Model
        if not isinstance(surrogate_model, SurrogateModelGP):
            raise TypeError(
                "Expected type(surrogate_model) to be SurrogateModelGP. Instead it is " +
                f"'{type(surrogate_model)}'."
            )

    def get_feasibility_constraint(self, x_ts_scaled:np.array, oc_min:float=0,
                                   gp_conf_ub:float=None)-> list[NonlinearConstraint]:
        # oc min sanity check:
        self._oc_min_sanity_check(oc_min)

        # gp_conf_ub sanity_check: (must be 0 < gp_conf_ub < 1)
        if gp_conf_ub is not None:
            # gp_conf must be scalar:
            if np.size(gp_conf_ub) != 1:
                raise ValueError(f"gp_conf_ub must be scalar, but is '{gp_conf_ub}'.")

            if gp_conf_ub <= 0 or 1 >= gp_conf_ub:
                raise ValueError(f"gp_conf_ub must be in (0, 1), but is '{gp_conf_ub}'.")
        # if gp_conf_ub is None, set it to 0.5 -> results in 0 offset
        else:
            gp_conf_ub : float = 0.5

        if self.feasibility_predictor_type == 'lost_load':
            raise NotImplementedError(
                "lost_load feasibility prediction is not implemented with GPs")

        if self.feasibility_predictor_type == 'overcapacity':
            def fun(x_cap):
                # Get true mean and std from GP:
                y_mean, y_std = self._feasibility_scaled_gp(x_cap, x_ts_scaled, return_std=True)
                print(f"xcap = {x_cap}, ymean={y_mean}")

                # Calculate offset from confidence upper bound:
                offset : np.float64 = norm.ppf(gp_conf_ub)

                # Subtract offset from mean:
                return y_mean - offset * y_std

            return [NonlinearConstraint(
                fun=fun,
                jac='2-point',
                lb= 0, #oc_min / self.surrogate_model.output_max,
                ub= np.inf)
            ]

        raise ValueError(
            f"Invalid feasibility_predictor_type: {self.feasibility_predictor_type}. " +
            "Expected: 'lost_load', 'overcapacity'."
        )

    def _feasibility_scaled_gp(self, x_cap_scaled: np.ndarray, x_ts_scaled: np.ndarray,
                               return_std: bool = False):
        """ Compute the scaled feasibility value using the GP model.

        Args:
            x_cap_scaled (np.array): process capacities (controlled by the optimization).
                Scaled to O(1) by deviding by process_max
            x_ts_scaled (np.array): timeseries values (fixed parameters for the optimization).
                x_ts_scaled is to O(1).

        Returns:
            float: scaled feasibility value (same scale as GP training target, i.e., y_scaled)
        """
        # --- Extract relevant subsets for feasibility ---
        x_cap_af_scaled = x_cap_scaled[[self.processes.index(p) for p in self.processes_af]]
        x_ts_af_scaled  = x_ts_scaled[[self.timeseries.index(t) for t in self.timeseries_af]]

        x_cap_af_scaled = np.array(list(x_cap_af_scaled))
        x_ts_af_scaled  = np.array(list(x_ts_af_scaled))

        # Concatenate to match SurrogateModel_GP input order:
        # input_names_process + input_names_timeseries
        x_gp = np.concatenate([x_cap_af_scaled, x_ts_af_scaled], axis=0).reshape(1, -1)

        # Predict mean (and std if requested)
        if return_std:
            y_mean, y_std = self.surrogate_model.sm.predict(x_gp, return_std=True)
            return float(y_mean.ravel()[0]), float(y_std.ravel()[0])

        y_scaled = self.surrogate_model.sm.predict(x_gp)

        return float(y_scaled.ravel()[0])

class FeasibilityPredictorNoFP(FeasibilityPredictor):
    """ Feasibility predictor that does not impose any feasibility constraints.
    """
    def __init__(self, data_in_path:str, project_file:str, surrogate_model,
                 feasibility_predictor_type:str, processes_full:list, timeseries_full:list):

        # Sanity check: feasibility_predictor must be a Surrogate Model
        if surrogate_model is not None:
            raise TypeError(
                "Expected type(surrogate_model) to be SurrogateModelNN. Instead it is " +
                f"'{type(surrogate_model)}'."
            )

        if feasibility_predictor_type != "noFP":
            raise ValueError(
                f"Invalid feasibility_predictor_type: {feasibility_predictor_type}. " +
                "Expected: 'noFP'."
            )

        super().__init__(data_in_path, project_file, surrogate_model, feasibility_predictor_type,
                         processes_full, timeseries_full)

    def get_feasibility_constraint(self, x_ts_scaled:np.array, oc_min:float=None,
                                   gp_conf_ub:float=None)-> list:

        # Check that not gp_conf_ub is given:
        if gp_conf_ub is not None:
            raise ValueError(
                f"gp_conf_ub must be None for FeasibilityPredictorNoFP, but is '{gp_conf_ub}'."
            )
        # oc min sanity check:
        self._oc_min_sanity_check(oc_min)

        return []

####################################################################################################
#                               SURROGATE ASSISTED DESIGN OPTIMIZATION                             #
####################################################################################################

class SADO(ABC):
    """ Abstract base class for Surrogate Assisted Design Optimization (SADO).
    """
    def __init__(self, cost_predictor:CostPredictor, feasibility_predictor:FeasibilityPredictor):
        """
        Args:
            cost_predictor (CostPredictor): Predictor for the total annual cost (TAC)
            feasibility_predictor (FeasibilityPredictor): Predictor for the feasibility constraints
        """

        self.cost_predictor        = cost_predictor
        self.feasibility_predictor = feasibility_predictor

        # Set bounds:
        self.processes = cost_predictor.processes
        self.timeseries = cost_predictor.timeseries

        self.potential_capacities = [
            get_potential_capacity(self.cost_predictor.technologies[p]) for p in self.processes]
        self.bounds = self._get_initial_bounds()

    def _get_initial_bounds(self) -> list:
        """ Calculates the initial bounds for the optimization problem.
        The bounds are calculated based on the potential capacity of each process.

        Returns:
            list: A list of tuples, where each tuple contains the lower and upper bound
                for the potential capacity of each process.
        """

        # Sanity Check, that potential capacities of processes match input_process_max:
        if not (
            self.cost_predictor.surrogate_model.input_process_max
            ==
            np.array( self.potential_capacities )).all():
            raise ValueError(
                "Potential capacities loaded from surrogate model:\n" +
                f"({self.cost_predictor.surrogate_model.input_process_max})\n do not fit potential"+
                " capacities from technologies.yaml:\n" +
                f"({ np.array( self.potential_capacities )})."
            )

        return [(0, 1)] * len(self.processes)

class SADOMachineLearning(SADO):
    """ Surrogate Assisted Design Optimization (SADO) using machine learning
    for both cost and feasibility prediction.
    """
    def __init__(self, cost_predictor:CostPredictor, feasibility_predictor:FeasibilityPredictor):

        super().__init__(cost_predictor, feasibility_predictor)

        # Sanity check: feasibility_predictor must be a FeasibilityPredictor
        if not isinstance(feasibility_predictor, FeasibilityPredictor):
            raise TypeError(
                "Expected type(feasibility_predictor) to be FeasibilityPredictor. Instead it is " +
                f"'{type(feasibility_predictor)}'."
            )

    def predict_optimal_design(self, df_ts_in:pd.DataFrame, options:dict, oc_min:float=0,
                               gp_conf_ub:float=None, custom_guess : pd.Series = None) -> dict:
        """ Predicts the optimal design based on the given timeseries input data.

        Args:
            df_ts_in (pd.DataFrame): DataFrame containing the timeseries input data.
            options (dict): Options for the optimization algorithm, e.g. {'maxiter': 1000}.
            oc_min (float): minimal allowed overcapacity. Must be non-negative. A value != 0 is only
                allowed for self.feasibility_predictor.feasibility_predictor_type = 'overcapacity'
            gp_conf_ub (float): minimal required confidence (upper bound) in feasibility prediction.
                gp_conf_ub must be between 0 and 1.
                gp_conf_ub = 0.5: mean prediction is sufficient -> offset = 0
                gp_conf_ub = 0.95: offset = 1.6448 (determined by scipy.stats.norm.ppf
                Only relevant for SurrogateModelGP. Defaults to None.
            custom_guess (pd.Series): Custom initial guess for the design capacities.

        Returns:
            dict: result of the optimization
        """
        assert len(df_ts_in) == 1

        # Extract (unscaled) input scaling factors for the timeseries:
        x_ts = df_ts_in[self.timeseries].values.squeeze()
        x_ts_max  = self.cost_predictor.surrogate_model.input_timeseries_max
        x_ts_scaled = x_ts / x_ts_max

        if custom_guess is None: # Use pot. capacity as bound if no custom initial guess is given:
            initial_guess = np.array([
                self.bounds[i][1] for i in range(len(self.processes))
            ])
        else: # Otherwise, use custom guess:
            initial_guess = self._get_custom_initial_guess(custom_guess)

        # Get the feasibility constraint function:
        feasibility_constraint : list = self.feasibility_predictor.get_feasibility_constraint(
            x_ts_scaled, oc_min, gp_conf_ub)

        # Case distinction: is cost_predictor CostPredictorNN or CostPredictorGP?
        if isinstance(self.cost_predictor, CostPredictorNN):
            result = minimize(
                fun=lambda x_cap_scaled: self.cost_predictor.calc_tac_and_grad(
                    x_cap_scaled, x_ts_scaled)[0],
                jac=lambda x_cap_scaled: self.cost_predictor.calc_tac_and_grad(
                    x_cap_scaled, x_ts_scaled)[1],
                x0= initial_guess,
                method='trust-constr',
                bounds=self.bounds,
                constraints=feasibility_constraint,
                options = options
            )
        elif isinstance(self.cost_predictor, CostPredictorGP):
            raise NotImplementedError("cost min is not implemented for GPs yet.")
        else:
            raise TypeError(f"Invalid cost_predictor type: {type(self.cost_predictor)}")

        # Add unscaled results
        result = dict(result)
        result['fun_scaled'] = result['fun']
        result['fun'] = result['fun'] * self.cost_predictor.surrogate_model.output_max

        result['x_scaled'] = result['x']
        result['x'] = result['x'] * self.cost_predictor.surrogate_model.input_process_max

        return result

    def _get_custom_initial_guess(self, guess:pd.Series) -> np.ndarray:
        """ Extracts the values of process capacities, scales them with their respective potential
        capacities, and returns them.

        Args:
            guess (pd.Series): pandas.Series containing all processes custom capacities

        Returns:
            np.ndarray: custom initial guess for design capacities scaled to O(1)
        """

        # typecheck:
        if not isinstance(guess, pd.Series):
            raise TypeError(f"Exptected guess to be a pd.Series, but got '{type(guess)}'.")

        inital_guess_uncaled : np.ndarray = guess[self.processes].values
        inital_guess_scaled : np.ndarray = inital_guess_uncaled / self.potential_capacities

        tol = 1e-10
        if (inital_guess_scaled > 1 + tol).any():
            raise ValueError(
                f"custom intial guess exeeds potential capacity: '{inital_guess_scaled}'.")
        if (inital_guess_scaled < 0 - tol).any():
            raise ValueError(
                f"custom intial guess contains negative values: '{inital_guess_scaled}'.")

        return inital_guess_scaled

class SADOConexHull(SADO):
    """ Surrogate Assisted Design Optimization (SADO) using machine learning
    for cost prediction and convex hull for feasibility prediction.
    """
    def __init__(self, cost_predictor:CostPredictor, feasibility_predictor:Feasibility_LP):

        # Sanity check: feasibility_predictor must be a Feasibility_LP
        if not isinstance(feasibility_predictor, Feasibility_LP):
            raise TypeError(
                "Expected type(feasibility_predictor) to be Feasibility_LP. Instead it is " +
                f"'{type(feasibility_predictor)}'."
            )
        # CH currently only implemented for CostPredictorNN:
        if not isinstance(cost_predictor, CostPredictorNN):
            raise NotImplementedError(
                "SADOConexHull is currently only implemented for CostPredictorNN as cost_predictor."
            )
        super().__init__(cost_predictor, feasibility_predictor)

        # Add feasiblity affecting processes:
        self.processes_af      = self.feasibility_predictor.input_names_process
        self.processes_af_idx  = [
            i for i, value in enumerate(self.processes)
            if value in self.processes_af
            ]
        # Sanity check:
        assert len(self.processes_af_idx) == len(self.processes_af)

        # Add feasiblity affecting timeseries:
        self.timeseries_af     = self.feasibility_predictor.input_names_timeseries
        self.timeseries_af_idx = [
            i for i, value in enumerate(self.timeseries)
            if value in self.timeseries_af
            ]
        # Sanity check:
        assert len(self.timeseries_af_idx) == len(self.timeseries_af)

        # ------------ Define indizes of input variables: ---------------
        # Indices of capacities in inputs which affect feasibility
        idx_cap_in_af = [
            i for i, k in enumerate(self.processes_af + self.timeseries_af) if k in self.processes]
        # Indices of timeseries in inputs which affect feasibility
        idx_ts_in_af = [
            i for i, k in enumerate(self.processes_af + self.timeseries_af) if k in self.timeseries]
        # Indices of capacities affecting feasibility in capacities
        self.idx_af_in_cap = [
            i for i, k in enumerate(self.processes) if k in self.processes_af]
        # Indices of capacities NOT affecting feasibility in capacities
        self.idx_naf_in_cap = [
            i for i, k in enumerate(self.processes) if k not in self.processes_af]
        # Indices of timeseries affecting feasibility in timeseries
        self.idx_af_in_ts = [i for i, k in enumerate(self.timeseries) if k in self.timeseries_af]

        # box bounds for the capacities NOT affecting feasibiltiy (already scaled to O(1))
        self.y_min = np.array([0] * len(self.idx_naf_in_cap))
        self.y_max = np.array([1] * len(self.idx_naf_in_cap))

        # ------------ Define vertices of the convex hull: ---------------
        # All vertices from the convex hull
        self.V = self.feasibility_predictor.vertices.T
        # Split vertices into V_cap (free) and V_Ts (fixed)
        self.V_cap = self.V[idx_cap_in_af, :] # (d_afcap, nV)
        self.V_ts = self.V[idx_ts_in_af, :] # (d_afts, nV)
        self.nV = self.V.shape[1] # number of vertices

    def predict_optimal_design(self, df_ts_in:pd.DataFrame, max_iter:int=200,
                               alpha_max:float=5e-4) -> dict:
        """ Main method to predict the optimal design based on the given timeseries input data.

        Args:
            df_ts_in (pd.DataFrame): DataFrame containing the timeseries input data.
            max_iter (int, optional): Maximum number of iterations for the Frank-Wolfe algorithm.
                Defaults to 200.
            alpha_max (float, optional): Maximum step size for the Amijo line search.
                Defaults to 5e-4.

        Returns:
            dict: result of the optimization
        """
        assert len(df_ts_in) == 1

        # ------------ Get x_ts_af from df_ts_in: ------------
        x_ts = df_ts_in[self.timeseries].values.squeeze()
        x_ts_scaled = x_ts / self.cost_predictor.surrogate_model.input_timeseries_max
        x_ts_af =x_ts_scaled[self.idx_af_in_ts]

        # ------------ Initialize A and b matrix for lmo_hull: ------------
        E = np.vstack([np.ones((1, self.nV)), self.V_ts])
        e = np.concatenate([np.array([1.0]), x_ts_af])

        # ------------ Get inital points: ------------
        # Get initial feasible point (solve LP with zero objective) ----
        lam0 = linprog(np.zeros(self.nV), A_eq=E, b_eq=e, bounds=(0, None), method="highs").x
        x_cap_af = self.V_cap @ lam0
        # x_cap_naf = 0.5*(np.asarray(self.y_min) + np.asarray(self.y_max)) # Start inside the box
        x_cap_naf = np.random.rand(len(self.idx_naf_in_cap))
        # ------------ Initialize empty lists as trackers for bookkeeping ------------
        history_tac = []
        history_x_cap = []
        history_grad = []

        # ------------ Iteratively optimize tac via Frank Wolfe ------------
        for k_iter in range(max_iter):
            x_cap = self._assemble_xcap(x_cap_af, x_cap_naf)
            # Get tac (f) and the gradient of tac w.r.t x_cap
            tac, grad_cap = self.cost_predictor.calc_tac_and_grad(x_cap, x_ts_scaled)

            # Bookkeeping: add tac and x_cap
            history_tac.append(tac)
            history_x_cap.append(x_cap)
            history_grad.append(grad_cap)

            # Split gradient
            grad_cap_af = grad_cap[np.asarray(self.idx_af_in_cap)]
            grad_cap_naf  = grad_cap[np.asarray(self.idx_naf_in_cap)]

            # Call LMO (linear minimization oracle) on hull section and box
            # s_* denoted the point of intersection with the convex hull
            s_cap_af, _ = self._lmo_hull(grad_cap_af, E, e)
            s_cap_naf = np.where(grad_cap_naf > 0.0, self.y_min, self.y_max)

            # Frank Wolfe direction
            d_cap_af  = s_cap_af  - x_cap_af
            d_cap_naf = s_cap_naf - x_cap_naf

            # Do Amijo line search.
            # Build direction of x_cap
            d_cap = np.zeros_like(x_cap)
            d_cap[np.asarray(self.idx_af_in_cap )] = d_cap_af
            d_cap[np.asarray(self.idx_naf_in_cap)] = d_cap_naf
            alpha = self._amijo_line_search(
                x_cap, tac, d_cap, grad_cap, x_ts_scaled, alpha_max=alpha_max)

            # Update x values:
            x_cap_af  = x_cap_af  + alpha * d_cap_af
            x_cap_naf = x_cap_naf + alpha * d_cap_naf

        # After last iteration: Add final x_cap and tac values:
        history_tac.append(tac)
        history_x_cap.append(x_cap)
        history_grad.append(grad_cap)

        history_tac = np.array(history_tac)
        history_x_cap = np.array(history_x_cap)
        history_grad = np.array(history_grad)

        # Build result dict
        result = {
            'x_scaled': history_x_cap[np.argmin(history_tac) - 1],
            'fun_scaled': np.min(history_tac),
            'niter': int(k_iter),
            'history_x_cap': history_x_cap,
            'history_tac': history_tac,
            'history_grad': history_grad,
        }
        result['fun'] = result['fun_scaled'] * self.cost_predictor.surrogate_model.output_max
        result['x'] = result['x_scaled'] * self.cost_predictor.surrogate_model.input_process_max

        return result

    def _lmo_hull(self, grad_afcap:np.array, E:np.array, e:np.array, obj_random:float=0) -> tuple:
        """ Linear Minimization Oracle (LMO) for the convex hull feasibility predictor.

        Args:
            grad_afcap (np.array): gradient w.r.t. the capacities affecting feasibility
            E (np.array): equality constraint matrix
            e (np.array): equality constraint vector
            obj_random (float, optional): magnitude of random noise added to the objective
                coefficients. Default is 0.

        Returns:
            tuple:
                s_afcap (np.array): point on the convex hull minimizing the linearized objective
                lam (np.array): coefficients of the convex combination of vertices

        """
        # Minimize grad_afcap^T (V_free lambda) = (V_free^T grad_afcap)^T lambda
        c = self.V_cap.T @ grad_afcap  # (nV,)
        # Add some randomness:
        c = c + obj_random*np.random.randn(*c.shape)
        res = linprog(c, A_eq=E, b_eq=e,
                    bounds=(0, None), method="highs", options={"presolve": True})
        if res.status != 0:
            raise RuntimeError(f"LMO LP infeasible or failed: {res.message}")
        lam = res.x
        s_afcap = self.V_cap @ lam
        return s_afcap, lam

    def _assemble_xcap(self, x_cap_af:np.array, x_cap_naf:np.array):
        """ Assembles the full x_cap vector from the feasibility affecting"""
        d_cap = len(self.processes)
        x_cap = np.zeros(d_cap, dtype=float)
        x_cap[np.asarray(self.idx_af_in_cap)] = x_cap_af
        x_cap[np.asarray(self.idx_naf_in_cap)] = x_cap_naf
        return x_cap

    def _amijo_line_search(
            self, x_cap:np.array, f:float, d_cap:np.array, grad_cap:np.ndarray,
            x_ts_scaled:np.array, c1:float=1e-4, alpha_max:float=5e-4) -> float:
        """ Performs an Amijo line search to find the step size alpha.

        Args:
            x_cap (np.array): current capacities
            f (float): current function value
            d_cap (np.array): search direction
            grad_cap (np.ndarray): gradient at current capacities
            x_ts_scaled (np.array): scaled timeseries values
            c1 (float, optional): Armijo parameter. Default is 1e-4.
            alpha_max (float, optional): maximum step size. Default is 5e-4.

        Returns:
            float: _description_
        """
        df0 = grad_cap @ d_cap

        alpha = 1
        while True:
            x_cap_trial = x_cap + alpha * d_cap
            f_trial, _ = self.cost_predictor.calc_tac_and_grad(x_cap_trial, x_ts_scaled)
            if f_trial <= f + c1 * alpha * df0 or alpha < alpha_max:
                break
            alpha *= 0.5

        return alpha
