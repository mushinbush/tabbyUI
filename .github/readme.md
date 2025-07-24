⚠️ **DEPRECATED**

This repository is no longer maintained.  

The upstream API seems to have changed its implementation of Streaming Completion, which likely breaks the functionality.  
As a result, and since I’ve switched to a different solution, this repository is no longer functional.  

---

## Description:

This repository features a web UI designed for interacting with [ExllamaV2](https://github.com/turboderp/exllamav2)'s API ([TabbyAPI](https://github.com/theroyallab/tabbyAPI/)). Features:

- Easy one-click installation with venv, simply download and deploy directly (requires python installed).
- Load or unload ExllamaV2 models (exl2 models).
- Support for basic features for engaging with LLMs, including text generation, conversation (under construction).

## Usage:
### Windows
1. Install [Python](https://www.python.org/). Make sure to check "Add Python X.X to PATH" during the installation.
2. Set up [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) according to its instructions. You will obtain the API URL and API KEY from this step.
3. Clone this repository to your local machine, or simply download it (Code -> Download ZIP).
4. Run the `start.bat` file. This will create a virtual environment (venv) and install necessary dependencies.
5. Open the URL indicated in the terminal in your browser (e.g., http://localhost:8501).
6. All set! You can input the API URL and API key in the left sidebar, as well as switch between models.
