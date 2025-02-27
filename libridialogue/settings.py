# /libridialogue/settings.py

import yaml
from pathlib import Path

# Define the path to the project root (one level up from /libridialogue)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the path to the config.yml file located at /config.yml
CONFIG_PATH = PROJECT_ROOT / "config.yml"

# Verify that the config.yaml exists
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

# Load the YAML configuration file
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# LIBRISPEECH settings
LIBRISPEECH_PATH = config["librispeech"]["path"]
LIBRISPEECH_CSV = config["librispeech"]["csv"]

# Random seed
RANDOM_SEED = config["random_seed"]

# Data overwrite setting
OVERWRITE_GENERATED_DATA = config["overwrite_generated_data"]

# Separation settings
SEPARATE_COSY = config["separate"]["cosy"]
SEPARATE_MOSSFORMER2 = config["separate"]["mossformer2"]
SEPARATE_CONVTASNET = config["separate"]["convtasnet"]

# Analysis settings
ANALYZE_CLEAN = config["analyze"]["clean"]
ANALYZE_REVERB_DUAL = config["analyze"]["reverb_dual"]
ANALYZE_COSY = config["analyze"]["cosy"]
ANALYZE_MOSSFORMER2 = config["analyze"]["mossformer2"]
ANALYZE_CONVTASNET = config["analyze"]["convtasnet"]

# LIBRIDIALOGUE settings
LIBRIDIALOGUE_SIZE = config["libridialogue"]["size"]
LIBRIDIALOGUE_PATH = config["libridialogue"]["path"]
LIBRIDIALOGUE_SEPARATED_PATH = config["libridialogue"]["separated_path"]
LIBRIDIALOGUE_MIN_OVERLAP = config["libridialogue"]["min_overlap"]
LIBRIDIALOGUE_MAX_OVERLAP = config["libridialogue"]["max_overlap"]

# LIBRIDIALOGUE room settings
LIBRIDIALOGUE_ROOM_MIN_TEMPERATURE = config["libridialogue"]["room"]["min_temperature"]
LIBRIDIALOGUE_ROOM_MAX_TEMPERATURE = config["libridialogue"]["room"]["max_temperature"]
LIBRIDIALOGUE_ROOM_MIN_HUMIDITY = config["libridialogue"]["room"]["min_humidity"]
LIBRIDIALOGUE_ROOM_MAX_HUMIDITY = config["libridialogue"]["room"]["max_humidity"]
LIBRIDIALOGUE_ROOM_MIN_X = config["libridialogue"]["room"]["min_x"]
LIBRIDIALOGUE_ROOM_MAX_X = config["libridialogue"]["room"]["max_x"]
LIBRIDIALOGUE_ROOM_XY_DIFF = config["libridialogue"]["room"]["xy_diff"]
LIBRIDIALOGUE_ROOM_MIN_Z = config["libridialogue"]["room"]["min_z"]
LIBRIDIALOGUE_ROOM_MAX_Z = config["libridialogue"]["room"]["max_z"]
LIBRIDIALOGUE_ROOM_STEP_SIZE_SOURCE = config["libridialogue"]["room"][
    "step_size_source"
]
LIBRIDIALOGUE_ROOM_RT60_TGT = config["libridialogue"]["room"]["rt60_tgt"]
LIBRIDIALOGUE_ROOM_SOURCE_Z = config["libridialogue"]["room"]["source_z"]

# COSY settings
COSY_COMPRESSOR_PLUGIN_FILE = config["cosy"]["compressor_plugin_file"]
COSY_GATE_PLUGIN_FILE = config["cosy"]["gate_plugin_file"]
COSY_CONFIG_FILE = config["cosy"]["config_file"]
COSY_OPTIMIZATION_TRIALS = config["cosy"]["optimization_trials"]
COSY_OPTIMIZATION_INITIAL_CONFIG_FILE = config["cosy"][
    "optimization_initial_config_file"
]
COSY_OPTIMIZATION_PARAMETER_STEP_SIZE = config["cosy"][
    "optimization_parameter_step_size"
]

# Output files
ANALYSIS_CSV = config["analysis_csv"]
COMPUTATION_TIMES_CSV = config["computation_times_csv"]
