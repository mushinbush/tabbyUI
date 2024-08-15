import json

config_file = "data/config.json"
load_config_file = "data/load-config.json"
parameters_config_file = "data/parameters-config.json"

def load_config():
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"url": "", "api_key": ""}

def save_config(url, api_key):
    with open(config_file, "w") as file:
        json.dump({"url": url, "api_key": api_key}, file)

def load_load_config():
    try:
        with open(load_config_file, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_load_config(max_seq_len, gpu_split_auto, gpu_split, cache_mode, max_cache_size, rope_scale, rope_alpha, draft_rope_scale, draft_rope_alpha, draft_model):
    config = {
        "Max Seq Len": max_seq_len,
        "GPU Split Auto": gpu_split_auto,
        "GPU Split": gpu_split,
        "Cache Mode": cache_mode,
        "Max Cache Size": max_cache_size,
        "Rope Scale": rope_scale,
        "Rope Alpha": rope_alpha,
        "Draft Rope Scale": draft_rope_scale,
        "Draft Rope Alpha": draft_rope_alpha,
        "Draft Model": draft_model
    }
    with open(load_config_file, 'w') as config_file:
        json.dump(config, config_file)

def load_parameters_config():
    try:
        with open(parameters_config_file, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_parameters_config(parameters):
    with open(parameters_config_file, 'w') as param_config_file:
        json.dump(parameters, param_config_file)

def get_default_parameters():
    return {
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
        "typical_p": 1.0,
        "min_p": 0.0,
        "top_a": 0.0,
        "tfs": 1.0,
        "repetition_penalty": 1.0,
        "rep_pen_range": 0,
        "rep_pen_decay": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }