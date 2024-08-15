import streamlit as st
import json
from modules.api import *
from modules.configs import *

# Inits
config = load_config()
message = None
progress = 0
progress_bar = None
response_iter = None

# Streamlit configs
st.set_page_config(
    page_title="tabbyUI",
    page_icon="🐈",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "https://github.com/mushinbush"
        }
    )

# Reducing whitespace on the top of the page
st.markdown("""
    <style>
        .block-container{
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin-top: 1rem;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        .small-font {
            font-size: 14px !important;
            display: inline-block;
            line-height: 0;
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

    @st.dialog("Advanced Configuration")
    def advanced_config(max_seq_len):
        st.write("Enter additional configuration options:")
        st.session_state.max_cache_size = st.number_input("Max Cache Size", min_value=1, value=st.session_state.get("max_cache_size"), placeholder="Auto")
        st.session_state.rope_scale = st.number_input("Rope Scale", min_value=0.0, value=st.session_state.get("rope_scale"), format="%.2f", placeholder="Auto")
        st.session_state.rope_alpha = st.number_input("Rope Alpha", min_value=0.0, value=st.session_state.get("rope_alpha"), format="%.2f", placeholder="Auto")
        
        if st.button("Confirm"):
            st.rerun()

    with st.expander("Configuration"):
        load_config = load_load_config()
        max_seq_len = st.number_input("Max Seq Len", min_value=1, value=load_config.get("Max Seq Len", 4096))
        gpu_split_auto = True
        valid_split_input = True
        gpu_split = st.text_input("GPU Split", ", ".join(map(str, load_config.get("GPU Split", ""))), placeholder = "Auto")
        cache_mode_option = st.selectbox("Cache Mode", options=["FP16", "Q8", "Q6", "Q4"], index=["FP16", "Q8", "Q6", "Q4"].index(load_config.get("Cache Mode", "FP16")))
        
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.button("Save Config", use_container_width=True)
        
        with col2:
            if st.button("Advanced", use_container_width=True):
                advanced_config(max_seq_len)

        if save_button:
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
                save_load_config(
                    max_seq_len, 
                    gpu_split_auto, 
                    gpu_split_list, 
                    cache_mode_option,
                    st.session_state.get("max_cache_size"),
                    st.session_state.get("rope_scale"),
                    st.session_state.get("rope_alpha")
                )
                st.success("Configuration saved successfully!")


    # Model loading
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Model", use_container_width=True):
            load_config = load_load_config()
            response_iter, error_message = load_model(url_input, api_key_input, model_list, load_config)
            
            if error_message:
                message = ("error", error_message)
            else:
                progress = 0  # Initialize percentage

    with col2:
        if st.button("Unload Model", use_container_width=True):
            message = unload_model(url_input, api_key_input)

    # Progess bar of loading model
    if response_iter:
        progress_bar = st.progress(progress)
        finished = False

        for line in response_iter:
            if line:
                data = json.loads(line.decode('utf-8').split('data: ')[1])
                module = data['module']
                modules = data['modules']
                status = data['status']

                progress = int((module / modules) * 100) # Process percentage
                progress_bar.progress(progress)

                if status == 'finished' and not finished:
                    message = ("success", "Model loaded successfully!")
                    finished = True
                    break

        if not finished:
            message = ("error", "Model loading did not finish successfully.")

    # Error handling
    if message:
        if message[0] == "success":
            st.success(message[1])
        elif message[0] == "error":
            st.error(message[1])

# Main page
tab1, tab2 = st.tabs(["Completions", "Parameters"])

with tab1:
    col1, col2 = st.columns(2)

    with col2:
        st.markdown('<p class="small-font">Output Box</p>', unsafe_allow_html=True)
        completionbox = st.container(height=700)
    with col1:
        prompt = st.text_area(label="Input Box", height=700)

        if st.button("Start Completion"):
            parameters = load_parameters_config()
            
            with col2:
                completionbox.write_stream(request_completion(url_input, api_key_input, prompt, parameters, False))

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