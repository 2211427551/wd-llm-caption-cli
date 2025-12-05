# WD LLM Caption

A versatile, Python-based tool with a user-friendly Gradio GUI for generating image captions. It supports local models and any OpenAI-compatible API, allowing for flexible deployment from powerful local machines to lightweight, API-driven setups.

<img alt="GUI Demo" src="DEMO/DEMO_GUI.png" width="800"/>

## Features

- **Multiple Captioning Modes**:
  - **WD Tagging**: Generate Danbooru-style tags using various [WD series](https://huggingface.co/SmilingWolf) models.
  - **LLM Captioning**: Create descriptive, natural-language captions.
  - **Combined Mode (WD+LLM)**: Enhance LLM captions with context from WD tags for more accurate and detailed results.

- **Flexible Model Support**:
  - **Local Models**: Supports a wide range of popular open-source vision models (e.g., Llama-3.2-Vision, Qwen2-VL, Florence-2, Mini-CPM).
  - **OpenAI-Compatible API Support**:
      - This feature allows the tool to act as a client for **any** model service that follows the OpenAI API standard.
      - It is not limited to OpenAI's official models. You can connect to third-party providers or locally hosted models (e.g., via SiliconFlow, vLLM, Ollama) by providing an API endpoint, an API key (if required), and a model name.

- **User-Friendly Interfaces**:
  - **GUI**: An intuitive Gradio web interface for interactive single-image or batch processing.
  - **API**: A FastAPI-based server mode to allow other applications to use its captioning capabilities.

- **Highly Configurable**:
  - Easily adjust WD-tagger thresholds, post-processing, and LLM parameters like prompts and temperature directly in the GUI.
  - Advanced model parameters can be configured via JSON files in the `wd_llm_caption/configs/` directory.

---

## 🚀 Quick Start (Using an OpenAI-Compatible API)

This setup is perfect if you want to use the tool without a powerful local GPU. It relies on an external API service for captioning.

### 1. Installation

Open a terminal and follow these steps:

```shell
# Clone this repository
git clone https://github.com/fireicewolf/wd-llm-caption-cli.git
cd wd-llm-caption-cli

# Create and activate a Python virtual environment
python -m venv .venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install the minimal dependencies for API usage
pip install -r requirements_api.txt
```

### 2. Usage

Launch the GUI and start captioning!

```shell
# Run the GUI
python -m wd_llm_caption.gui

# (Optional) To allow remote connections, use --listen
python -m wd_llm_caption.gui --listen
```

In the GUI:
1.  Set `Caption method` to `LLM` or `WD+LLM`.
2.  Select **`OpenAI`** from the `Choice LLM` dropdown.
3.  The **"OpenAI API Settings"** section will appear.
4.  Enter your API service details:
    *   **API Endpoint**: The URL of your service (e.g., `http://127.0.0.1:8000/v1`).
    *   **API Key**: Your secret API key (if required by the service).
    *   **Model Selection**: Either select a model from the auto-fetched list or enter the name in `Custom Model Name`.
5.  Load the model and you are ready to generate captions using the API!

---

## 🛠️ Full Installation (For Local Models)

This setup is for users who want to run models locally. It requires a compatible GPU and significant disk space for model downloads.

### 1. Prerequisites

- Python 3.10+
- A CUDA-enabled GPU is highly recommended.

### 2. Installation

```shell
# Clone the repository and enter the directory
git clone https://github.com/fireicewolf/wd-llm-caption-cli.git
cd wd-llm-caption-cli

# Create and activate a virtual environment
python -m venv .venv
.\venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 1. Install PyTorch
# Find the correct command for your system from https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install base and GUI dependencies
pip install -r requirements.txt
pip install -r requirements_gui.txt

# 3. Install dependencies for the models you need
# For WD Tagging (ONNX models)
# For CUDA 11.8
pip install -r requirements_onnx_cu118.txt
# For CUDA 12.x
pip install -r requirements_onnx_cu12x.txt

# For Local LLMs (Transformers-based)
pip install -r requirements_llm.txt
```

---

## 📖 Usage Guide

### GUI

The easiest way to use the tool is via the Gradio GUI.

```shell
python -m wd_llm_caption.gui
```

**GUI Options:**
- `--port <number>`: Set the web UI port (default: `8282`).
- `--listen`: Allow remote network connections.
- `--share`: Create a public link via Gradio.
- `--inbrowser`: Automatically open the UI in a browser.
- `--log_level <level>`: Set the console log level (e.g., `INFO`, `DEBUG`).

### Command-Line (CLI)

For batch processing, the CLI is recommended.

```shell
# Example: Caption all images in a folder using an OpenAI-compatible API
python caption.py --data_path /path/to/your/images \
                  --caption_method llm \
                  --llm_choice openai \
                  --api_endpoint "http://localhost:8000/v1" \
                  --api_model "your-custom-model-name"

# Example: Caption using a local WD tagger and a local LLM
python caption.py --data_path /path/to/your/images \
                  --caption_method wd+llm \
                  --wd_model_name "wd-swinv2-tagger-v3" \
                  --llm_choice llama \
                  --llm_model_name "Llama-3.2-11B-Vision-Instruct"
```

For a full list of command-line options, run `python caption.py --help`.

---

## 🙏 Credits

This project is a fork of and has been significantly updated from the original [wd-llm-caption-cli by fireicewolf](https://github.com/fireicewolf/wd-llm-caption-cli). All credit for the foundational work goes to the original author.

This tool is also built upon the fantastic work of the open-source community. Special thanks to:
- [SmilingWolf](https://huggingface.co/SmilingWolf) for the WD tagger models.
- [fancyfeast](https://huggingface.co/fancyfeast) for the Joy-Caption models.
- [Meta](https://huggingface.co/meta-llama), [Qwen](https://huggingface.co/Qwen), [OpenBMB](https://huggingface.co/openbmb), and [Microsoft](https://huggingface.co/microsoft) for their powerful vision models.
