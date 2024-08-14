import streamlit as st
from modules.api import current_model, fetch_model_list, load_model, unload_model, request_completion
from modules.configs import load_config, save_config, save_load_config, load_load_config, save_parameters_config, load_parameters_config, get_default_parameters

config = load_config()
message = None

# Streamlit configs
st.set_page_config(layout="wide")

# Reducing whitespace on the top of the page
st.markdown("""
    <style>
        .block-container{
            padding-top: 1rem;
            padding-bottom: 0rem;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    url_input = st.text_input("API URL", config["url"], placeholder = "e.g. http://127.0.0.1:5000")
    api_key_input = st.text_input("API Key", config["api_key"], type = "password")

    if st.button("Save & Connect!", use_container_width=True):
        save_config(url_input,api_key_input)
        if fetch_model_list(url_input, api_key_input):
            data = current_model(url_input, api_key_input)
            if data:
                st.success("Saved & Connected!")
                st.write("Model:", data.get("id", "Cannot get Model name"))
            else:
                st.success("Saved & Connected!")
                st.write("Model: None")

    model_list = st.selectbox("Select Model", options=fetch_model_list(url_input, api_key_input))

    with st.expander("Configuration"):
        load_config = load_load_config()
        max_seq_len = st.number_input("Max Seq Len", min_value=1, value=load_config.get("Max Seq Len", 4096))
        gpu_split_auto = True
        valid_split_input = True
        gpu_split = st.text_input("GPU Split (comma-separated, leave blank for Auto-Split)", ", ".join(map(str, load_config.get("GPU Split", ""))))
        cache_mode_option = st.selectbox("Cache Mode", options=["FP16", "Q8", "Q6", "Q4"], index=["FP16", "Q8", "Q6", "Q4"].index(load_config.get("Cache Mode", "FP16")))
        
        if st.button("Save Configuration", use_container_width=True):
            if gpu_split.strip(): # valid split input
                try:
                    gpu_split_list = [float(x.strip()) for x in gpu_split.split(",")]
                    gpu_split_auto = False
                except ValueError:
                    st.error("Invalid input: Please enter a comma-separated list of integers or floats.")
                    gpu_split_list = load_config.get("GPU Split", "")
                    valid_split_input = False
            else:
                gpu_split_list = load_config.get("GPU Split", "")
                gpu_split_auto = True

            if valid_split_input:
                save_load_config(max_seq_len, gpu_split_auto, gpu_split_list, cache_mode_option)
                st.success("Configuration saved successfully!")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Model", use_container_width=True):
            load_config = load_load_config()
            message = load_model(url_input, api_key_input, model_list, load_config)

    with col2:
        if st.button("Unload Model", use_container_width=True):
            message = unload_model(url_input, api_key_input)

    if message:
        if message[0] == "success":
            st.success(message[1])
        elif message[0] == "error":
            st.error(message[1])

# Main page
tab1, tab2 = st.tabs(["Completions", "Parameters"])
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        prompt = st.text_area(label="Input Box", height=700)
    with col2:
        result_area = st.empty()
        result_area.markdown("")
    if st.button("Start Completion"):
        with col2:
            result = ""
            parameters = load_parameters_config()

            result = prompt
            for chunk in request_completion(url_input, api_key_input, prompt, parameters):
                result += chunk
                result_area.markdown(result)

with tab2:
    parameters = load_parameters_config()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temperature = st.number_input("Temperature", 0.0, 5.0, parameters.get("temperature", 0.7))
        min_p = st.number_input("Min P", 0.0, 1.0, parameters.get("min_p", 0))
        rep_pen_range = st.number_input("Rep Pen Range", 0, 204800, parameters.get("rep_pen_range", 0))
        
    with col2:
        top_k = st.number_input("Top K", 0, 200, parameters.get("top_k", 0))
        top_a = st.number_input("Top A", 0.0, 1.0, parameters.get("top_a", 0))
        rep_pen_decay = st.number_input("Rep Pen Decay", 0, 204800, parameters.get("rep_pen_decay", 0))

    with col3:
        top_p = st.number_input("Top P", 0.0, 1.0, parameters.get("top_p", 1.0))
        tfs = st.number_input("TFS", 0.0, 1.0, parameters.get("tfs", 1.0))       
        frequency_penalty = st.number_input("Frequency Penalty", -2.0, 2.0, parameters.get("frequency_penalty", 0.0))

    with col4:
        typical_p = st.number_input("Typical P", 0.0, 1.0, parameters.get("typical_p", 1.0))
        repetition_penalty = st.number_input("Repetition Penalty", 1.0, 3.0, parameters.get("repetition_penalty", 1.0))
        presence_penalty = st.number_input("Presence Penalty", -2.0, 2.0, parameters.get("presence_penalty", 0.0))

    col1, col2 = st.columns([1,1])

    with col1:
        if st.button("Save Parameters", use_container_width=True):
            parameters = {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "typical_p": typical_p,
                "min_p": min_p,
                "top_a": top_a,
                "tfs": tfs,
                "repetition_penalty": repetition_penalty,
                "rep_pen_range": rep_pen_range,
                "rep_pen_decay": rep_pen_decay,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
            save_parameters_config(parameters)
            st.success("Parameters saved successfully!")
    with col2:
        if st.button("Reset to Default", use_container_width=True):
            save_parameters_config(get_default_parameters())