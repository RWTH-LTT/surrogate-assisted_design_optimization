# WebApp to simulate an energy system surrogate model with Streamlit.

# .\venv\your_venv\Scripts\activate
# streamlit run .\webapp\sm\webapp_full.py


# ==========================================================================================
# region Imports
# ==========================================================================================
from dataclasses import dataclass
from typing import Dict, List
import base64
import os
import yaml
import pandas as pd
import plotly.express as px
import streamlit as st
from sm.surrogates_feasible_space import Feasibility_LP
from sm.surrogate_model import SurrogateModelNN
from sm.sado import SADOMachineLearning, SADOConexHull, CostPredictorNN, FeasibilityPredictorNN
from core.ts_scaling import (get_ts_scaling_names, interpolate_timeseries, scale_timeseries, get_ts_scaling_dict)
from core.helpers import (UniqueKeyLoader, get_data_out, get_experiment_id, get_process_names,
                          get_tsa_path, get_potential_capacity,  get_input_file, get_config_file)


PROJECT_FILE = 'project_ID14_10.yaml'
CASE = 'JP1'
ROOT = os.getcwd()
DATA_IN = os.path.join(ROOT, 'data', CASE, 'Input_Data')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEASIBILITY_ICON = "🔑"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Surrogate Model Demonstrator",
    layout="wide",
    initial_sidebar_state="expanded")


# ==========================================================================================
# region Global constants
# ==========================================================================================
PLOT_CONFIG = {
    'staticPlot': False, 'displayModeBar': False, 'scrollZoom': False,
    'showAxisDragHandles': False, 'showAxisRangeEntryBoxes': False,
}

LEGEND_STYLE = dict(
    yanchor="top", y=0.99, xanchor="left", x=0.01,
    bgcolor='rgba(255,255,255,1)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1
)


def get_design_helper(data_in, project_file, processes:list=None) -> dict:
    """_summary_

    Args:
        processes (list, optional): _description_. Defaults to None.

    Returns:
        dict: containing dicts for name and potential capacity, like
        DESIGN_HELPER = {
            'B':       {name: "Natural Gas Boiler (GB) in kW", pot_cap: 10000},
            'CHP':     {name: "Combined Heat and Power (CHP) in kW", pot_cap:8000},
            'HTHP':    {name: "High Temperature Heat Pump (HTHP) in kW", pot_cap:10000},
            'AC':      {name: "Absorption Chiller (AC) in kW", pot_cap:16000},
            'CC':      {name: "Compression Chiller (CC) in kW", pot_cap:16000},
            'PV':      {name: "Photovoltaic (PV) in kW", pot_cap:5000},
            'TES':     {name: "Thermal Energy Storage (TES) in kWh", pot_cap:100000},
            'battery': {name: "Li-ion Battery (BAT) in kWh", pot_cap:10000}
        }
    """
    if processes is None:
        processes = ['B', 'CHP', 'HTHP', 'AC', 'CC', 'PV', 'TES', 'battery'] 

    technologies = get_input_file(data_in, project_file, 'technologies')['technologies']

    design_helper = {}

    for process in processes:
        unit = 'kW' if technologies[process]['category'] == 'production' else 'kWh'
        potential_capacity = get_potential_capacity(technologies[process])
        name = technologies[process]['name']
        name_capitalized =  " ".join(word.capitalize() for word in name.split())
        display_name = technologies[process]['display_name']

        name_long = f"{name_capitalized} ({display_name}) in {unit}"

        # FOR HTHP: double potential capacity, as electricity is falsly the reference product
        if process == 'HTHP':
            assert technologies[process]['reference'] == 'electricity'
            potential_capacity = 2 * potential_capacity

        design_helper[process] = {
            'name': name_long,
            'potential_capacity': potential_capacity,
            'default_capacity': technologies[process]['default_capacity'],
            'affect_feasibility': technologies[process]['affect_feasibility'],
            'category'          : technologies[process]['category']}

    return design_helper

def get_ts_scaling_helper(data_in, project_file):

    config : dict = get_config_file(data_in, project_file)

    ts_scaling_yaml : dict = get_input_file(
        data_in, project_file, 'timeseries_scaling')['timeseries_scaling'][
            config['timeseries_scaling']]
    ts_scaling_names : list = get_ts_scaling_names(
        data_in, project_file, only_feasibility_affecting= False)

    ts_scaling_helper : dict = {}

    for ts_scaling_name in ts_scaling_names:
        ts_scaling_dict = get_ts_scaling_dict(ts_scaling_name, ts_scaling_yaml)

        unit = ts_scaling_dict['unit']

        ts_scale_type = ts_scaling_name.split('_')[-1]
        subdict_str = {
            'mean': f" (Mean) in {unit}",
            'peak': f" (Peak) in {unit}",
            'relpeak': " (Peak Relative to Mean)",
            'interpolate': "",
        }[ts_scale_type]

        min_ =  float(ts_scaling_dict['param'][ts_scale_type]['min'])
        max_ =  float(ts_scaling_dict['param'][ts_scale_type]['max'])

        ts_scaling_helper[ts_scaling_name] = {
            'name' : ts_scaling_dict['display_name'] + subdict_str,
            'min' : min_,
            'max' : max_,
            'default' : (min_ + max_) / 2,
            'ts_type': ts_scaling_dict['ts_type'],
            'affect_feasibility' : ts_scaling_dict['affect_feasibility']
        }
    return ts_scaling_helper

DESIGN_HELPER = get_design_helper(DATA_IN, PROJECT_FILE)
TS_SCALING_HELPER = get_ts_scaling_helper(DATA_IN, PROJECT_FILE)

STRATEGY_OPTIONS = [
    {   "title": "Overcapacity (OC) Predictor",
        "features": [
            {"type": "good", "text": "Custom Safety Margin"},
            {"type": "bad",  "text": "No Feasibility Guarantee"},]},
    {   "title": "Energy-Not-Supplied (ENS) Predictor",
        "features": [
            {"type": "bad", "text": "No Feasibility Guarantee"},
            {"type": "bad", "text": "Inconsistent"},]},
    {   "title": "Convex Hull (CH) Predictor",
        "features": [
            {"type": "good", "text": "Guaranteed Feasibility"},
            {"type": "bad",  "text": "Slow"},]}
]


# ==========================================================================================
# region Types / dataclasses
# ==========================================================================================
@dataclass
class ResourceState:
    ''' Holds all loaded resources to avoid global variables. '''
    project_file: str
    data_in: str
    data_out: str
    processes: List[str]
    unscaled_ts_data: Dict[str, pd.DataFrame]
    sm_opex: SurrogateModelNN
    sm_oc: SurrogateModelNN
    sm_ll: SurrogateModelNN
    feasibility_lp: Feasibility_LP
    sado_oc: SADOMachineLearning
    sado_ll: SADOMachineLearning
    sado_ch: SADOConexHull


# ==========================================================================================
# region Custom CSS + image
# ==========================================================================================
def setup_styling() -> None:
    """
    Injects custom CSS to style the app.

    It overrides the default Streamlit look to create custom font sizes,
    specific button colors, and layout spacing.
    """
    st.markdown(
        """
        <style>
        /* ==========================================================================
           1. GLOBAL TEXT & TYPOGRAPHY
           Controls the default font sizes for the whole app.
           ========================================================================== */

        /* Set the base text size for the entire page */
        html, body, [class*="css"] {
            font-size: 16px !important;
        }

        /* Ensure text inside Forms and markdown paragraphs is readable */
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] label p,
        div[data-testid="stMarkdownContainer"] p {
            font-size: 16px !important;   /* adjust to 15–18px */
            line-height: 1.25 !important;
        }

        /* Custom sizes for Titles (H1) and Subtitles (H2, H3) */
        h1 { font-size: 2.0em !important; }
        h2 { font-size: 1.4em !important; }
        h3 { font-size: 1.1em !important; }

        [data-testid="stHeader"] {
        background: transparent !important;
        height: 3rem !important;
        }

        /* ==========================================================================
           2. SEGMENTED CONTROL (TOGGLE BUTTONS)
           Customizes the look of selection button (Operational Surrogate Model/SADO toggle at the top).
           ========================================================================== */

        /* Make the text inside the toggle button very large (50px). */
        button[data-testid*="stBaseButton-segmented_control"] div[data-testid="stMarkdownContainer"] p {
            font-size: 1.5em !important;
            line-height: 1.2 !important;
        }

        /* Increase the physical height of the button so the large text fits inside. */
        button[data-testid*="stBaseButton-segmented_control"] {
            min-height: 30px !important;
            padding-top: 3px !important;
            padding-bottom: 3px !important;
        }

        /* ==========================================================================
           3. THE "OPTIMIZE" BUTTON
           Styles the specific main action button (Green).
           ========================================================================== */

        /* Find the button located next to our hidden "button-marker"
           and style it distinctively (Green background, white text). */
        div:has(> div > p > span#button-marker) + div button,
        div:has(span#button-marker) + div button {
            background-color: #38761D;          /* Dark Green */
            color: white !important;
            border: 2px solid #38761D;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            transition: all 0.3s ease;          /* Smooth animation */
        }
        div:has(span#button-marker) + div button {
        min-height: 44px !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        }

        div:has(span#button-marker) + div button p {
        font-size: 18px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        }

        /* Hover Effect: Make the button slightly lighter and grow when mouse is over it. */
        div:has(span#button-marker) + div button:hover {
            background-color: #579841;
            border-color: #579841;
            transform: scale(1.02);
        }

        /* Hide the marker itself (it is only used to locate the button). */
        span#button-marker { display: none; }

        /* ==========================================================================
           4. PAGE LAYOUT
           Adjusts margins and width of the main page area.
           ========================================================================== */

        /* Adjust whitespace around the edges of the page. */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 5rem;    /* More space on left */
            padding-right: 5rem;   /* More space on right */
        }

        /* Force the app to use the full width of the screen (good for dashboards). */
        section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
            max-width: 3600px;
        }

        /* ==========================================================================
           5. CONTENT BOXES (Cards)
           Creates visual styling for grouped content (White or Gray backgrounds).
           ========================================================================== */

        /* Helper classes are hidden, we style the container that holds them. */
        .white-box, .gray-box, .metric-box { display: none; }

        /* Common style for both White and Gray boxes:
           Add a black border, rounded corners, and a shadow. */
        [data-testid="stColumn"]:has(.white-box) > [data-testid="stVerticalBlock"],
        [data-testid="stColumn"]:has(.gray-box) > [data-testid="stVerticalBlock"] {
            border: 3px solid black !important;
            border-radius: 12px !important;
            padding: 4px 6px !important;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin: 6px !important;;
        }

        /* Specific background color for White boxes */
        [data-testid="stColumn"]:has(.white-box) > [data-testid="stVerticalBlock"] {
            background-color: #FFFFFF !important;
        }

        /* Specific background color for Gray boxes */
        [data-testid="stColumn"]:has(.gray-box) > [data-testid="stVerticalBlock"] {
            background-color: #eaeaea !important;
        }

        /* ==========================================================================
           6. FINANCIAL METRICS (KPI Cards)
           Styles the boxes showing numbers like OPEX, CAPEX, etc.
           ========================================================================== */

        /* Container style for the metric card (Gray background, centered). */
        [data-testid="stColumn"]:has(.metric-box) > [data-testid="stVerticalBlock"] {
            border: 3px solid black !important;
            border-radius: 12px !important;
            background-color: #eaeaea !important;
            padding: 6px 12px !important;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin: 6px !important;
        }

        /* Style the Label (e.g., the word "OPEX").
           Makes it bold. */
        [data-testid="stColumn"]:has(.metric-box) [data-testid="stMetricLabel"] * {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: #333 !important;
            text-align: center !important;
            line-height: 1.1 !important;
        }

        /* Style the Value (e.g., the number "$500").
           Makes it very large and centered. */
        [data-testid="stColumn"]:has(.metric-box) [data-testid="stMetricValue"] * {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            text-align: center !important;
        }

        /* Reduce vertical spacing inside columns */
        [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }

        /* --- SLIDERS: keep label normal --- */
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] label p,
        div[data-testid="stSlider"] [data-testid="stWidgetLabel"] * {
            font-size: 1rem !important;     /* or 0.95rem */
            line-height: 1.2 !important;
        }

        /* tick labels under the slider */
        div[data-testid="stSlider"] [data-testid="stTickBar"] * {
            font-size: 0.75rem !important;  /* ~12px */
        }

        /* value readout / bubble */
        div[data-testid="stSlider"] [data-baseweb="slider"] * {
            font-size: 0.85rem !important;  /* ~14px */
        }

        /* Expander header (the clickable title) */
        div[data-testid="stExpander"] summary {
            font-size: 0.9rem !important;
        }

        /* Expander content */
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] li {
            font-size: 0.85rem !important;
            line-height: 1.35 !important;
        }

        /* LEFT-anchor the whole radio group in the sidebar */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"]{
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;   /* <- left */
        }

        /* LEFT-anchor each option row (dot + label) */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label{
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;  /* <- left */
        align-items: center !important;
        gap: 0.4rem !important;
        }

        /* Keep text left too */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label span{
        text-align: left !important;
        }

        
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def get_img_base64(file_path: str) -> str:
    """ Reads an image file and returns its base64 encoded string. """
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# ==========================================================================================
# region TS and SM Loading
# ==========================================================================================
@st.cache_resource
def load_surrogate_models_and_unscaled_ts(data_in: str, project_file: str) -> ResourceState:
    ''' Loads all surrogate models and unscaled timeseries data. '''

    print(f"Experiment ID: '{get_experiment_id(data_in, project_file)}'\n\n")

    data_out = get_data_out(data_in, project_file)

    # Get all Input Names
    processes = get_process_names(
        data_in, project_file, only_feasibility_affecting=False, storage_or_production='both')
    ts_scaling_names = get_ts_scaling_names(data_in, project_file, only_feasibility_affecting=False)

    # Get Input Names affecting Feasibility
    processes_af = get_process_names(
        data_in, project_file, only_feasibility_affecting=True, storage_or_production='both')
    ts_scaling_names_af = get_ts_scaling_names(data_in, project_file, only_feasibility_affecting=True)

    tsa_path = get_tsa_path(data_in, project_file)
    yaml_file_path = os.path.join(tsa_path, 'time_series.yaml')

    # Load time series configuration from YAML
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        timeseries_yaml = yaml.load(f, Loader=UniqueKeyLoader)
    timeseries_df = pd.DataFrame.from_dict(timeseries_yaml, orient='index')

    # Load all unscaled time series data into a dictionary
    unscaled_ts_data = {}
    for ts_name, filename in timeseries_df['file'].items():
        file_path = os.path.join(tsa_path, filename)
        data = pd.read_csv(file_path)
        unscaled_ts_data[ts_name] = data


    #---------------------------------------------------------------------------------------------
    # Load Surrogate Models
    #---------------------------------------------------------------------------------------------
    print("Loading Surrogate Models...")

    # OPEX Surrogate
    print("\n - OPEX Surrogate Model - ")
    sm_opex = SurrogateModelNN(
        data_in,
        project_file,
        input_names =   processes + ts_scaling_names,
        output_name='OPEX_woLL',
        sm_name='ANN_4L')
    sm_opex.setup_sm()
    sm_opex.load_model(os.path.join(data_out, 'NN_opex'), 'NN_opex')

    # SADO with Overcapacity Predictor
    print("\n - Overcapacity Predictor - ")
    sm_oc = SurrogateModelNN(
        data_in,
        project_file,
        input_names = processes_af + ts_scaling_names_af,
        output_name='overcapacity',
        sm_name='ANN_4L')
    sm_oc.setup_sm()
    sm_oc.load_model(model_path = os.path.join(data_out, 'NN_overcapacity'),
                     filename = 'NN_overcapacity')

    # SADO with Energy-Not-Supplied Predictor
    print("\n - Energy-Not-Supplied Predictor - ")
    sm_ll = SurrogateModelNN(
        data_in,
        project_file,
        input_names = processes_af + ts_scaling_names_af,
        output_name='lost_load',
        sm_name='ANN_4L')
    sm_ll.setup_sm()
    sm_ll.load_model(model_path = os.path.join(data_out, 'NN_lostload'), filename = 'NN_lostload')

    # SADO with  Convex Hull Predictor
    print("\n - Convex Hull Predictor - ")
    df_vertex = pd.read_csv(
        os.path.join(data_out, 'convex_hull_vertex_results_5000.csv'))
    feasibility_lp = Feasibility_LP(
        data_in, project_file,
        df_vertices=df_vertex
    )

    #---------------------------------------------------------------------------------------------
    # Build Cost and Feasibility Predictors
    #---------------------------------------------------------------------------------------------
    cost_predictor = CostPredictorNN(data_in, project_file, sm_opex)
    feas_pred_oc = FeasibilityPredictorNN(data_in, project_file, sm_oc, 'overcapacity',
                                          processes_full=cost_predictor.processes,
                                          timeseries_full=cost_predictor.timeseries)
    feas_pred_ll = FeasibilityPredictorNN(data_in, project_file, sm_ll, 'lost_load',
                                          processes_full=cost_predictor.processes,
                                          timeseries_full=cost_predictor.timeseries)

    #---------------------------------------------------------------------------------------------
    # Build Surrogate-Assisted Design Optimizations
    #---------------------------------------------------------------------------------------------
    print("\n\n Building SADO's...")

    sado_oc = SADOMachineLearning(
        cost_predictor          = cost_predictor,
        feasibility_predictor   = feas_pred_oc
    )

    sado_ll = SADOMachineLearning(
        cost_predictor          = cost_predictor,
        feasibility_predictor   = feas_pred_ll
    )

    sado_ch = SADOConexHull(
        cost_predictor          = cost_predictor,
        feasibility_predictor   = feasibility_lp
    )
    print("\n\n\n")

    return ResourceState(
        project_file=project_file,
        data_in=data_in,
        data_out=data_out,
        processes=processes,
        unscaled_ts_data=unscaled_ts_data,
        sm_opex=sm_opex,
        sm_oc=sm_oc,
        sm_ll=sm_ll,
        feasibility_lp=feasibility_lp,
        sado_oc=sado_oc,
        sado_ll=sado_ll,
        sado_ch=sado_ch
    )

# ==========================================================================================
# region Central Logic
# ==========================================================================================
def scale_all_timeseries(unscaled_data: Dict[str, pd.DataFrame], inputs: dict) -> Dict[str, pd.DataFrame]:
    """
    Scales all timeseries based on the user inputs dictionary.
    Returns a dictionary of DataFrames.
    """
    scaled_data = {}

    # Helper to apply the scaling
    def apply_scale(name, scale_mode, new_mean, new_peak=None, new_relpeak=None):

        # Extract raw data
        unscaled_timeseries = unscaled_data[name]

        # Perform the math
        scaled_timeseries = scale_timeseries(
            timeseries = unscaled_timeseries,
            scale_mode = scale_mode,
            old_mean = unscaled_timeseries.values.mean(),
            old_peak = unscaled_timeseries.values.max(),
            new_mean = new_mean,
            new_peak = new_peak,
            new_relpeak = new_relpeak
        )

        # Return a DataFrame to keep column names/dates intact
        return pd.DataFrame(scaled_timeseries, index=unscaled_timeseries.index, columns=unscaled_timeseries.columns)

    # Electricity Demand
    scaled_data['ts_demand_el'] = apply_scale(
        'ts_demand_el', 'mean_and_peak', inputs['demand_electricity_mean'], inputs['demand_electricity_peak']
    )

    # Heat Demand
    scaled_data['ts_demand_heat'] = apply_scale(
        'ts_demand_heat', 'mean_and_peak', inputs['demand_heat_mean'], inputs['demand_heat_peak']
    )

    # Cold Demand
    scaled_data['ts_demand_cold'] = apply_scale(
        'ts_demand_cold', 'mean', inputs['demand_cold_mean']
    )

    # Electricity Price
    scaled_data['ts_price_el_buy'] = apply_scale(
        'ts_price_el_buy', 'mean_and_relpeak', inputs['pricebuy_electricity_mean'], new_relpeak=inputs['pricebuy_electricity_relpeak']
    )

    # Gas Price
    scaled_data['ts_price_gas_buy'] = apply_scale(
        'ts_price_gas_buy', 'mean', inputs['pricebuy_gas_mean']
    )

    # PV Interpolation
    ts1 = unscaled_data['ts_availability_photovoltaics']
    ts2 = unscaled_data['ts_availability_interpolation_photovoltaics']

    pv_scaled = interpolate_timeseries(
        timeseries_1=ts1,
        timeseries_2=ts2,
        scaler=inputs['availability_PV_interpolate']
    )
    scaled_data['ts_availability_photovoltaics'] = pd.DataFrame(
        pv_scaled, index=ts1.index, columns=ts1.columns
    )

    return scaled_data

def calculate_kpis(inputs: dict, state: ResourceState) -> dict:
    """
    Prepares input data and runs all surrogate models.
    Returns a dictionary with the final values.
    """
    # Prepare the Data for the Models
    model_data = inputs.copy()
    model_data['task_index'] = 1
    model_data['id'] = 1
    df = pd.DataFrame([model_data])

    # OPEX (Convert to Million EUR)
    opex_val = state.sm_opex.predict(df)[0,0] / 1_000_000

    # OC Predictor (Convert to %)
    oc_val = state.sm_oc.predict(df)[0,0] * 100

    # ENS Predictor (Convert to MWh)
    ens_val = state.sm_ll.predict(df)[0,0] / 1_000

    # Feasibility (0 or 1)
    feasibility_val = state.feasibility_lp.predict_feasibility(df)[0][0]

    return {
        "opex": opex_val,
        "OC": oc_val,
        "ENS": ens_val,
        "feasibility": feasibility_val
    }

def calculate_design(inputs: dict, strategy: str, threshold: float, iterations: int, state: ResourceState) -> dict:
    """
    Optimizes a single design point and returns design variables plus financial metrics.
    """
    # Prepare Input Data
    # Create a single-row DataFrame with required ID columns
    input_data = inputs.copy()
    input_data.update({'task_index': 1, 'id': 1})
    df = pd.DataFrame([input_data])

    # Select Strategy Model
    if "Overcapacity" in strategy:
        sado = state.sado_oc
        result = sado.predict_optimal_design(
            df,
            options={"maxiter": iterations, "verbose": 1},
            oc_min=threshold / 100.0
        )
    elif "Energy-Not-Supplied" in strategy:
        sado = state.sado_ll
        result = sado.predict_optimal_design(
            df,
            options={"maxiter": iterations, "verbose": 1},
            oc_min=0.0
        )
    elif 'Convex Hull (CH) Predictor' in strategy:
        sado = state.sado_ch
        result = sado.predict_optimal_design(
            df,
            max_iter=iterations
        )
    else:
        raise ValueError(f"Unknown optimization strategy: '{strategy}'.")

    # Apply Optimization Results
    # Update the DataFrame with the new capacities to prepare for OPEX prediction
    optimized_capacities = result['x']

    hthp_index = sado.processes.index('HTHP')
    # Double HTHP capacity (as nominal capacity in optimized_capacities refers to electricity,
    # but we want to display heat)
    optimized_capacities[hthp_index] = optimized_capacities[hthp_index] * 2

    df[state.processes] = optimized_capacities

    # Construct Final Output
    return {
        **df[state.processes].iloc[0].to_dict(),
        "TAC_pred": result['fun'],
        "CAPEX_pred": sado.cost_predictor.calc_capex(optimized_capacities),
        "OPEX_pred": float(state.sm_opex.predict(df).squeeze())
    }


# ==========================================================================================
# region Plotting helpers
# ==========================================================================================
def create_plot(title: str, y_label: str, data_map: Dict[str, pd.DataFrame], trace_styles: dict | None = None) -> None:
    """
    Generic helper to create a Plotly line chart.
    """

    # Prepare Data: Combine individual series into one DataFrame
    data_series = [df.iloc[:, 0].rename(name) for name, df in data_map.items()]
    all_data = pd.concat(data_series, axis=1)

    # Create Basic Figure
    fig = px.line(all_data, y=all_data.columns)

    # Apply Custom Styles (Colors/Dashes)
    if trace_styles:
        for name, style in trace_styles.items():
            fig.update_traces(selector=dict(name=name), line=style)

    # Apply Layout Formatting
    fig.update_layout(
        title = {'text':title, 'x':0.0, 'xanchor':"left"},
        xaxis_title="time step",
        yaxis_title=y_label,
        dragmode=False,
        margin=dict(l=20, r=10, t=30, b=25),
        height=250, # FROM 320
        legend=LEGEND_STYLE,
        legend_title_text='',
        paper_bgcolor='rgba(0,0,0,0)',
        autosize = True
    )

    st.plotly_chart(fig,  width='stretch', config=PLOT_CONFIG)

def display_simulation_plots(scaled_data: Dict[str, pd.DataFrame], unscaled_data: Dict[str, pd.DataFrame]) -> None:
    """
    Defines which plots to show and HOW they look.
    """

    # Plot 1: Demands
    create_plot(
        title="Energy Demands",
        y_label="demand in kW",
        data_map={
            'electricity': scaled_data['ts_demand_el'],
            'heating':     scaled_data['ts_demand_heat'],
            'cooling':     scaled_data['ts_demand_cold']
        },
        trace_styles={
            'electricity': {'color': 'green'},
            'heating':     {'color': 'red'},
            'cooling':     {'color': 'blue'}
        }
    )

    # Plot 2: Prices
    create_plot(
        title="Prices",
        y_label="price in EUR/kWh",
        data_map={
            'electricity': scaled_data['ts_price_el_buy'],
            'natural gas': scaled_data['ts_price_gas_buy']
        },
        trace_styles={
            'electricity': {'color': 'green'},
            'natural gas': {'color': 'orange'}
        }
    )

    # Plot 3: PV Availability
    create_plot(
        title="PV Availability",
        y_label=r"availability in kW/kW peak",
        data_map={
            'PV profile Flensburg': unscaled_data['ts_availability_photovoltaics'],
            'PV profile Freiburg':  unscaled_data['ts_availability_interpolation_photovoltaics'],
            'PV interpolated':      scaled_data['ts_availability_photovoltaics']
        },
        trace_styles={
            'PV profile Flensburg':  {'color': 'blue',  'dash': 'dot'},
            'PV profile Freiburg':   {'color': 'blue',  'dash': 'dash'},
            'PV interpolated':       {'color': 'black', 'dash': 'solid'}
        }
    )


# ==========================================================================================
# region UI helpers
# ==========================================================================================
def get_production_storage_inputs(key_suffix: str = "A") -> dict:
    """
    Renders Production and Storage sliders.
    key_suffix: A unique string (e.g., 'opt1', 'opt2') to prevent widget ID conflicts.
    """
    inputs = {}

    st.header("Production Processes (d)")
    for process, process_dict in DESIGN_HELPER.items():
        if process_dict['category'] != 'production':
            continue

        name = process_dict['name']
        if process_dict['affect_feasibility']:
            name = f"{name} {FEASIBILITY_ICON}"

        inputs[process] = st.slider(
            name, 0,
            process_dict['potential_capacity'],
            process_dict['default_capacity'],
            key=f"{process}_{key_suffix}")

        # Divide capacity of HTHP by two as the reference product is electricity instead of heat
        if process == 'HTHP':
            inputs[process] = inputs[process] / 2

    st.header("Storage Processes (d)")
    for process, process_dict in DESIGN_HELPER.items():
        if process_dict['category'] != 'storage':
            continue

        name = process_dict['name']
        if process_dict['affect_feasibility']:
            name = f"{name} {FEASIBILITY_ICON}"

        inputs[process] = st.slider(
            name, 0,
            process_dict['potential_capacity'],
            process_dict['default_capacity'],
            key=f"{process}_{key_suffix}")

    st.markdown("\n\n")
    with st.container(border=True):
        st.markdown(f"""<span style="font-size: 16px;"><b>{FEASIBILITY_ICON} = affects feasibility</b></span>""", unsafe_allow_html=True)

    return inputs

def get_demand_price_inputs(key_suffix: str = "A") -> dict:
    """
    Renders Demand, Price, and Renewable sliders.
    key_suffix: A unique string (e.g., 'opt1', 'opt2') to prevent widget ID conflicts.
    """
    inputs = {}

    st.header("Energy Demands (s)")
    for ts_name, my_dict in {
        key: value for key,value in TS_SCALING_HELPER.items() if value['ts_type'] == 'demand'
        }.items():

        name = my_dict['name']
        if my_dict['affect_feasibility']:
            name = f"{name} {FEASIBILITY_ICON}"

        inputs[ts_name] = st.slider(name, my_dict['min'], my_dict['max'],
                                    my_dict['default'], key=f"{ts_name}_{key_suffix}")

    st.header("Energy Prices (s)")
    for ts_name, my_dict in {
        key: value for key,value in TS_SCALING_HELPER.items() if value['ts_type'] == 'pricebuy'
        }.items():

        inputs[ts_name] = st.slider( my_dict['name'], my_dict['min'], my_dict['max'],
                                    my_dict['default'], key=f"{ts_name}_{key_suffix}")

    st.header("Renewable Availability (s)")
    for ts_name, my_dict in {
        key: value for key,value in TS_SCALING_HELPER.items() if value['ts_type'] == 'availability'
        }.items():

        inputs[ts_name] = st.slider( my_dict['name'], my_dict['min'], my_dict['max'],
                                    my_dict['default'], key=f"{ts_name}_{key_suffix}")

    # Warnings logic
    if inputs['demand_electricity_peak'] < inputs['demand_electricity_mean']:
        st.toast("Invalid input:\n Mean > Peak for **Electricity**.", icon="⚠️")
    if inputs['demand_heat_peak'] < inputs['demand_heat_mean']:
        st.toast("Invalid input: Mean > Peak for **Heating**.", icon="⚠️")

    return inputs

def _render_metric_box(label: str, value: float) -> None:
    """Renders the HTML container and Streamlit metric."""
    # Preserving your custom CSS class injection
    st.markdown('<div class="metric-box"></div>', unsafe_allow_html=True)
    st.metric(label=label, value=f"{value / 1e6 :.3f} Mio. €")

def display_design(dat_results: dict) -> None:
    """
    Renders the design results including capacities and financial metrics.
    """
    ## Design Capacities
    for key, value in dat_results.items():
        if key in DESIGN_HELPER:
            name    = DESIGN_HELPER[key]['name']
            max_val = DESIGN_HELPER[key]['potential_capacity']

            # Calculate ratio safely (0.0 to 1.0)
            ratio = min(max(value / max_val, 0.0), 1.0)

            st.markdown(f"**{name}**: {value:.1f} / {max_val}")
            st.progress(ratio)


    ## Economic Metrics
    st.header("Economic Metrics")

    # Row 1: OPEX and TAC
    col1, col2 = st.columns(2)
    # Row 2: CAPEX (Left aligned)
    col3, _ = st.columns(2)

    with col1:
        _render_metric_box('OPEX', dat_results['OPEX_pred'])
    with col2:
        _render_metric_box('TAC', dat_results['TAC_pred'])
    with col3:
        _render_metric_box('CAPEX', dat_results['CAPEX_pred'])

def strategy_selection() -> tuple[str, float, int, bool]:
    """
    Renders the strategy selection UI cards and optimization controls.
    """

    st.header("Select Feasibility Predictor")

    # State Initialization
    if "selected_strategy" not in st.session_state:
        st.session_state.selected_strategy = STRATEGY_OPTIONS[0]["title"]

    current_selection = st.session_state.selected_strategy
    threshold = 0  # Default return value

    # Render Feasibility Predictor Cards
    for strategy in STRATEGY_OPTIONS:
        title = strategy["title"]
        is_selected = (current_selection == title)

        with st.container(border=True):
            # --- Row 1: Title ---
            display_title = f"✅ {title}" if is_selected else title
            st.markdown(f"##### {display_title}")

            # --- Row 2: Features (left) + Button (right) ---
            col_feat, col_btn = st.columns([0.75, 0.25], vertical_alignment="top")

            with col_feat:
                for feature in strategy["features"]:
                    if feature["type"] == "good":
                        st.markdown(f":green[**+ {feature['text']}**]")
                    else:
                        st.markdown(f":red[**- {feature['text']}**]")

            with col_btn:
                st.write("")  # tiny spacer so button doesn't stick to title
                if st.button(
                    label="✅" if is_selected else "Select",
                    key=f"btn_{title}",
                    type="primary" if is_selected else "secondary",
                    disabled=is_selected,
                    use_container_width=True,
                ):
                    st.session_state.selected_strategy = title
                    st.rerun()

            # --- Row 3: Only for OC ---
            if title == "Overcapacity (OC) Predictor":
                threshold = st.number_input(
                    "Minimum Overcapacity Threshold (%)",
                    value=10,
                    step=1,
                )
    st.divider()

    # Number of iterations
    iterations = st.number_input("**Number of Iterations**", value=100, step=1)

    st.divider()

    # Optimization Trigger
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        # Injection for CSS targeting
        st.markdown('<span id="button-marker"></span>', unsafe_allow_html=True)

        run_optimization = st.button(
            "OPTIMIZE ☺",
            key="run_optimization_btn",
            type="secondary",
            use_container_width=True
        )

    return current_selection, threshold, iterations, run_optimization

def display_kpi_dashboard(kpis: dict, background_image: str) -> None:
    """
    Generates the SVG overlay and renders it in Streamlit.
    """

    # Define size of small and big boxes
    SMALL_W, SMALL_H = 13, 4
    BIG_W, BIG_H = 25, 8

    # Feasibility Logic
    if kpis['feasibility']:
        feas_color = "#4CAF50" # Green
        feas_text = "Feasible\n(Guarantee)"
    else:
        feas_color = "#F44336" # Red
        feas_text = "Infeasible"

    # Map values to screen coordinates (Top/Left %)
    kpi_map = {
        "opex": {
            "value": f"{kpis['opex']:.2f}",
            "color": "white", "text_color": "black",
            "top": 17.55, "left": 74, "width": SMALL_W, "height": SMALL_H
        },
        "feasibility": {
            "value": feas_text,
            "color": feas_color, "text_color": "white",
            "top": 46 + BIG_H/2, "left": 80, "width": BIG_W, "height": BIG_H
        },
        "OC": {
            "value": f"{kpis['OC']:.2f}",
            "color": "#4CAF50" if kpis['OC'] > 0 else "#F44336",
            "text_color": "white",
            "top": 67 + SMALL_H/2, "left": 74, "width": SMALL_W, "height": SMALL_H
        },
        "ENS": {
            "value": f"{kpis['ENS']:.2f}",
            "color": "#4CAF50" if kpis['ENS'] < 0 else "#F44336",
            "text_color": "white",
            "top": 86.3 + SMALL_H/2, "left": 74, "width": SMALL_W, "height": SMALL_H
        }
    }

    # Build SVG String
    img_b64 = get_img_base64(background_image)
    shapes = ""

    for _, item in kpi_map.items():
        # Extract properties
        val = item['value']
        w, h = item['width'], item['height']
        x_rect = item['left'] - (w / 2)
        y_rect = item['top'] - (h / 2)
        f_size = item.get('font_size', 2.4)

        # Handle multiline text
        if '\n' in val:
            lines = val.split('\n')
            text_content = (f'<tspan x="{item["left"]}%" dy="-0.6em">{lines[0]}</tspan>'
                            f'<tspan x="{item["left"]}%" dy="1.2em">{lines[1]}</tspan>')
        else:
            text_content = val

        # Add Rectangle & Text Group
        shapes += (
            f'<g>'
            f'<rect x="{x_rect}%" y="{y_rect}%" width="{w}%" height="{h}%" '
            f'fill="{item["color"]}" fill-opacity="1" stroke="black" stroke-width="0.4" />'
            f'<text x="{item["left"]}%" y="{item["top"]}%" '
            f'font-family="Arial, sans-serif" font-size="{f_size}" font-weight="bold" '
            f'fill="{item["text_color"]}" text-anchor="middle" dominant-baseline="middle">'
            f'{text_content}'
            f'</text>'
            f'</g>'
        )

    # Assemble & Render HTML
    html_code = (
        f'<div style="position: relative; width: 100%; margin: 0; padding: 0;">'
        f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; display: block; height: auto;" />'
        f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" '
        f'style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">'
        f'{shapes}'
        f'</svg>'
        f'</div>'
    )

    st.markdown(html_code, unsafe_allow_html=True)


# ==========================================================================================
# region Streamlit Pages
# ==========================================================================================
@st.fragment
def operational_surrogate_model() -> None:
    """
    Renders the full operational surrogate model interface.
    """
    state: ResourceState = st.session_state.get("state")
    if state is None:
        st.error("Resources not loaded yet. Please restart the app.")
        return

    col1, col2, col3, col4 = st.columns([7, 7, 10, 10])

    # --------------------------------------------------------------------------
    # User Inputs
    # --------------------------------------------------------------------------
    inputs = {}

    # Production & Storage - Inputs
    with col1:
        st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
        prod_storage_data = get_production_storage_inputs(key_suffix="scen1")
        inputs.update(prod_storage_data)

    # Demands & Prices - Inputs
    with col2:
        st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
        demand_price_data = get_demand_price_inputs(key_suffix="scen1")
        inputs.update(demand_price_data)

    # --------------------------------------------------------------------------
    # Perform Timeseries Scaling and Plot Timeseries
    # --------------------------------------------------------------------------
    with col3:
        st.markdown('<div class="gray-box"></div>', unsafe_allow_html=True)
        st.header("Visualization of Time-Series Scaling")
        scaled_ts_data = scale_all_timeseries(state.unscaled_ts_data, inputs)
        display_simulation_plots(scaled_ts_data, state.unscaled_ts_data)

    # --------------------------------------------------------------------------
    # Perform Surrogate Predictions, Display Results
    # --------------------------------------------------------------------------
    with col4:
        st.markdown('<div class="gray-box"></div>', unsafe_allow_html=True)
        kpi_results = calculate_kpis(inputs, state)
        display_kpi_dashboard(
            kpi_results, background_image=os.path.join(BASE_DIR, "outputs_layout.png"))

    print("New >>Operational Surrogate Model<< run completed.")

@st.fragment
def surrogate_assisted_design_optimization() -> None:
    """
    Renders the full Surrogate-Assisted design optimization interface.
    """

    state: ResourceState = st.session_state.get("state")
    if state is None:
        st.error("Resources not loaded yet. Please restart the app.")
        return

    # Initialize session state for results if not already present
    if "sado_results" not in st.session_state:
        st.session_state.sado_results = None

    col1, col2, col3, col4 = st.columns([7, 10, 8.5, 8.5])

    # --------------------------------------------------------------------------
    # User Inputs
    # --------------------------------------------------------------------------
    inputs = {}
    with col1:
        st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
        demand_price_data = get_demand_price_inputs(key_suffix="scen2")
        inputs.update(demand_price_data)

    # --------------------------------------------------------------------------
    # Timeseries Scaling
    # --------------------------------------------------------------------------
    with col2:
        st.markdown('<div class="gray-box"></div>', unsafe_allow_html=True)
        st.header("Visualization of Time-Series Scaling")
        scaled_ts_data = scale_all_timeseries(state.unscaled_ts_data, inputs)
        display_simulation_plots(scaled_ts_data, state.unscaled_ts_data)

    # --------------------------------------------------------------------------
    # Strategy Selection & Button
    # --------------------------------------------------------------------------
    with col3:
        st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
        strategy, threshold, iterations, run_optimization = strategy_selection()

    # --------------------------------------------------------------------------
    # Display Design
    # --------------------------------------------------------------------------
    with col4:
        # Check here if the button was pressed
        if run_optimization:
            st.session_state.sado_results = calculate_design(
                inputs, strategy, threshold, iterations, state
            )

        st.markdown('<div class="gray-box"></div>', unsafe_allow_html=True)
        st.header("Design Results Overview")

        # Check if results exist in memory/session state
        if st.session_state.sado_results is not None:
            display_design(st.session_state.sado_results)
            print("New >>Surrogate-Assisted Design Optimization<< run completed.")

        else:
            # Placeholder message if optimization hasn't run yet
            st.info("Press 'OPTIMIZE' to see Results.")

def home_page():
    """ Configurates the homepage. """
    st.markdown(
    """
    # Welcome to the Interactive Surrogate Model Demonstrator

    This web application provides an interactive exploration of the surrogate models introduced in
    the accompanying publication (see reference below).  
    The purpose of this demonstrator is to make the behavior, assumptions, and limitations of the
    presented methods transparent, and to enable analysis beyond what is possible
    with static figures.

    ## Scope of the Demonstrator

    The application covers two components:

    - **Operational Surrogate Models (OSMs)**  
    The OSMs emulate an operational optimization. 
    Given a **scenario** (s) and a **design** (d),  
    the cost predictor estimates the cost-optimal operational expenditure (OPEX).  
    We compare three feasibility predictors that estimate feasibility via  
    (i) the energy-not-supplied, (ii) the overcapacity,
    and (iii) a binary feasibility classification based on a convex-hull approximation.

    - **Surrogate-Assisted Design Optimization (SADO)**  
    SADO emulates **design optimization** by embedding the OSMs into an optimization problem.  
    Given a **scenario** (s), it predicts a **cost-optimal design** subject to feasibility
    constraints derived from the selected feasibility predictor.

    For methodological details and validation results, we refer to the publication below.

    ## Publication
    **Enhancing Feasibility of Cost-Optimal Energy Systems in Surrogate-Assisted Design
    Optimization**  
    B. Jürgens, K. Reents, N. Baumeister, H. Seele, H. Schricker, N. von der Assen (2026).  
    *Preprint on SSRN.*  
    DOI: [10.2139/ssrn.6477860](https://doi.org/10.2139/ssrn.6477860)

    ## Funding
    The methods presented in this demonstrator have been developed as part of the research project
    **SuMo4EnSy**.  
    This project is supported by the German **Federal Ministry for Economic Affairs and Energy**
    (Grant number: 03EI1084B).  
    For more information about the project, please visit [SuMo4EnSy](
    https://www.ltt.rwth-aachen.de/cms/ltt/forschung/forschung-am-ltt/energiesystemtechnik/aktuelle-projekte/~biwuzn/sumo4ensy-bewertung-von-dekarbonisieru/?lidx=1
    ).
    """
    )
    st.markdown("---")

    st.markdown(
        """
        <small>
        Brought to you by <br>
        <a href="https://www.ltt.rwth-aachen.de" target="_blank" rel="noopener noreferrer">
            <img src="https://www.ltt.rwth-aachen.de/global/show_picture.asp?id=aaaaaaaaaanvmia" alt="Institute Logo" height="100">
        </a>
        </small>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================================
# region Page Selector
# ==========================================================================================
def web_app() -> None:
    # ''' Main WebApp function to select between modes. '''

    with st.sidebar:
        page = st.radio(
            "Navigation",
            ["Home", "OSM", "SADO"],
        )
        #
        if page == "Home":
            st.caption("")
        elif page == "OSM":
            st.caption("Operational Surrogate Model")
        else:
            st.caption("Surrogate-Assisted Design Optimization")

    if page == "Home":
        home_page()
    elif page == "OSM":
        operational_surrogate_model()
    else:
        surrogate_assisted_design_optimization()

    st.markdown("---")
    st.markdown(
        """
        <small>
        <a href="https://apps.ltt.rwth-aachen.de/impressum.html" target="_blank" rel="noopener noreferrer">
            Legal Notice / Imprint
        </a>
        </small>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================================
# region Main
# ==========================================================================================
if __name__ == "__main__":
    # Load resource state once and store in session
    if "state" not in st.session_state:
        st.session_state.state = load_surrogate_models_and_unscaled_ts(
            data_in= DATA_IN,
            project_file=PROJECT_FILE
        )

    setup_styling()

    web_app()
