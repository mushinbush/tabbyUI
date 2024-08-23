import streamlit as st
import requests, json

def current_model(url, api_key):
    api_url = f"{url}/v1/model"
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            return response.json()

        else:
            #st.error(f"Request failed, status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def fetch_model_list(url, api_key):
    api_url = f"{url}/v1/model/list"
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        else:
            st.error(f"Request failed, status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error: {e}")
        return []
    
def fetch_draft_model_list(url, api_key):
    api_url = f"{url}/v1/model/draft/list"
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return [None] + models if models else [None]
        else:
            st.error(f"Request failed, status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error: {e}")
        return []
    
def load_model(url, api_key, model_id, config):
    api_url = f"{url}/v1/model/load"
    headers = {
        "x-admin-key": api_key,
        "Authorization": ""
    }
    draft_model = config.get("Draft Model")

    payload = {
        "name": model_id,
        "max_seq_len": config.get("Max Seq Len", 4096),
        "override_base_seq_len": None,
        "cache_size": config.get("Cache Size", None),
        "gpu_split_auto": config.get("GPU Split Auto", True),
        "autosplit_reserve": None,
        "tensor_parallel": config.get("Tensor Parallel", None),
        "gpu_split": config.get("GPU Split", None),
        "rope_scale": config.get("Rope Scale", None),
        "rope_alpha": config.get("Rope Alpha", None),
        "cache_mode": config.get("Cache Mode", None),
        "chunk_size": None,
        "prompt_template": None,
        "num_experts_per_token": None,
        "fasttensors": None,
        "skip_queue": False,
        "draft": {
            "draft_model_name": draft_model,
            "draft_rope_scale": config.get("Draft Rope Scale"),
            "draft_rope_alpha": config.get("Draft Rope Scale"),
            "draft_cache_mode": config.get("Cache Mode")
        } if draft_model else None
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True)
        
        if response.status_code == 200:
            return response.iter_lines(), None
        else:
            return None, f"Request failed, status code: {response.status_code}"

    except Exception as e:
        st.error(f"Error: {e}")
        return None

def unload_model(url, api_key):
    api_url = f"{url}/v1/model/unload"
    headers = {
        "x-admin-key": api_key,
        "Authorization": ""
    }
    try:
        response = requests.post(api_url, headers=headers)
        if response.status_code == 200:
            message = ("success", "Model unloaded successfully!")
            return message
        else:
            message = ("error", f"Request failed, status code: {response.status_code}")
            return message
    except Exception as e:
        st.error(f"Error: {e}")
        return False
    
def request_completion(url, api_key, prompt, parameters, chat):
    api_url = f"{url}/v1/completions"
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    payload = param_payload(parameters,prompt)
    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        if chat == False:
            yield prompt # add prompt first, write -> write_stream needed

        for line in response.iter_lines(): # 逐行處理流式輸出的文本
            if line:
                line_content = line.decode('utf-8').replace("data: ", "") # 刪除 "data: " 開頭
                if line_content.strip() != "[DONE]":
                    completion_data = json.loads(line_content) # 解析JSON並取出"choices"內的"text"
                    text = completion_data['choices'][0]['text']
                    yield text  # 使用yield以生成器方式返回結果

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        yield ""

def param_payload(parameters,prompt):
    return {
        "prompt": prompt,
        "max_tokens": 512,
        "stream": True,
        "add_bos_token": True,
        "temperature_last": True,
        "temperature": parameters.get("temperature", None),
        "top_k": parameters.get("top_k", None),
        "top_p": parameters.get("top_p", None),
        "top_a": parameters.get("top_a", None),
        "min_p": parameters.get("min_p", None),
        "tfs": parameters.get("tfs", None),
        "typical": parameters.get("typical_p", None),
        "frequency_penalty": parameters.get("frequency_penalty", None),
        "presence_penalty": parameters.get("presence_penalty", None),
        "repetition_penalty": parameters.get("repetition_penalty", None),
        "penalty_range": parameters.get("rep_pen_range", None),
        "repetition_decay": parameters.get("rep_pen_decay", None),
    }

# Completion additions
'''
        "min_tokens": None,
        "generate_window": None,
        "stop": None,
        "banned_strings": None,
        "banned_tokens": None,
        "token_healing": None,
        "temperature_last": None,
        "smoothing_factor": None,
        "skew": None,
        "mirostat_mode": None,
        "mirostat_tau": None,
        "mirostat_eta": None,
        "add_bos_token": None,
        "ban_eos_token": None,
        "skip_special_tokens": None,
        "logit_bias": None,
        "negative_prompt": None,
        "json_schema": None,
        "regex_pattern": None,
        "grammar_string": None,
        "speculative_ngram": None,
        "cfg_scale": None,
        "max_temp": None,
        "min_temp": None,
        "temp_exponent": None,
        "model": None,
        "stream": True,
        "stream_options": None,
        "logprobs": None,
        "response_format": None,
        "n": None,
        "best_of": None,
        "echo": None,
        "suffix": None,
        "user": None,
'''