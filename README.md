# WD LLM Caption

[![PyPI - Version](https://img.shields.io/pypi/v/wd-llm-caption.svg)](https://pypi.org/project/wd-llm-caption)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wd-llm-caption.svg)](https://pypi.org/project/wd-llm-caption)

A versatile, Python-based tool with a user-friendly Gradio GUI for generating image captions. It supports both local models and OpenAI-compatible APIs, allowing for flexible deployment from powerful local machines to lightweight, API-driven setups.

<img alt="GUI Demo" src="DEMO/DEMO_GUI.png" width="800"/>

## Features

- **Multiple Captioning Modes**:
  - **WD Tagging**: Generate Danbooru-style tags using various [WD series](https://huggingface.co/SmilingWolf) models.
  - **LLM Captioning**: Create descriptive, natural-language captions.
  - **Combined Mode**: Enhance LLM captions with context from WD tags.
- **Flexible Model Support**:
  - **Local Models**: Supports a wide range of popular open-source vision models (e.g., Llama-3.2-Vision, Qwen2-VL, Florence-2, Mini-CPM).
  - **API Models**: Seamlessly integrates with any OpenAI-compatible API service (e.g., OpenAI, vLLM, Ollama).
- **User-Friendly Interfaces**:
  - **CLI**: A powerful command-line interface for batch processing and automation.
  - **GUI**: An intuitive Gradio web interface for interactive use.
- **Optimized for Different Setups**: Install only what you need. Use the lightweight API-only setup or the full-featured local model setup.

---

## 🚀 Quick Start (API Users)

This setup is perfect if you want to use the tool without a powerful local GPU. It relies on an external OpenAI-compatible API for captioning.

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
python gui.py

# (Optional) To allow remote connections, use --listen
python gui.py --listen
```

In the GUI:
1.  Select **`openai`** from the "LLM Choice" dropdown.
2.  Go to the **"LLM Advanced Options"** tab, then the **"OpenAI API Settings"** sub-tab.
3.  Enter your API service details:
    *   **API Endpoint**: The URL of your service (e.g., `https://api.openai.com/v1`).
    *   **API Key**: Your secret API key.
    *   **API Model**: The name of the model you want to use (e.g., `gpt-4o`).
4.  You are now ready to generate captions using the API!

---

## 🛠️ Full Installation (Local Models)

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

# 4. (Optional) Install model downloaders
# To download from Hugging Face (recommended)
pip install -r requirements_huggingface.txt
# To download from ModelScope
pip install -r requirements_modelscope.txt
```

---

## 📖 Usage Guide

### GUI

The easiest way to use the tool is via the Gradio GUI.

```shell
python gui.py
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
# Example: Caption all images in a folder using the API
python caption.py --data_path /path/to/your/images \
                  --caption_method llm \
                  --llm_choice openai \
                  --api_endpoint "http://localhost:8000/v1" \
                  --api_model "your-model-name"

# Example: Caption using a local WD tagger and a local LLM
python caption.py --data_path /path/to/your/images \
                  --caption_method wd+llm \
                  --wd_model_name "wd-swinv2-v3" \
                  --llm_choice llama \
                  --llm_model_name "Llama-3.2-11B-Vision-Instruct"
```

For a full list of command-line options, run:
```shell
python caption.py --help
```
<details>
<summary>Click to see all CLI options</summary>

`--data_path`: Path to your image dataset.
`--recursive`: Process images in subdirectories.
`--log_level`: Set console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
`--save_logs`: Save logs to a file.
`--model_site`: Download models from `huggingface` or `modelscope`.
`--models_save_path`: Directory to save local models.
`--use_sdk_cache`: Use the default cache directory of the download SDK.
`--force_download`: Force re-download of existing models.
`--caption_method`: Choose `wd` (tags), `llm` (description), or `wd+llm`.
`--run_method`: For `wd+llm`, run in `sync` (image by image) or `queue` (batch by batch).
`--caption_extension`: Extension for caption files (default: `.txt`).
`--not_overwrite`: Prevent overwriting existing caption files.
`--custom_caption_save_path`: Save caption files to a custom directory.

**WD Tagger Options:**
`--wd_config`: Path to WD model config JSON.
`--wd_model_name`: WD model to use.
`--wd_force_use_cpu`: Force CPU for WD model inference.
`--wd_caption_extension`: Extension for WD tag files in combined mode (default: `.wdcaption`).
`--wd_remove_underscore`: Replace `_` with spaces in tags.
`--wd_undesired_tags`: Comma-separated tags to exclude.
`--wd_threshold`: General confidence threshold for tags (default: `0.35`).
`--wd_character_threshold`: Confidence threshold for character tags.
`--wd_caption_separator`: Separator between tags (default: `, `).

**LLM Options:**
`--llm_choice`: Choose LLM type (`joy`, `llama`, `qwen`, `minicpm`, `florence`, `openai`).
`--llm_config`: Path to LLM model config JSON.
`--llm_model_name`: LLM to use.
`--llm_patch`: Apply a LoRA patch to the LLM.
`--llm_use_cpu`: Force CPU for LLM inference.
`--llm_dtype`: LLM data type (`fp16`, `bf16`, `fp32`).
`--llm_qnt`: LLM quantization (`none`, `4bit`, `8bit`).
`--llm_caption_extension`: Extension for LLM caption files in combined mode (default: `.llmcaption`).
`--llm_user_prompt`: Custom user prompt for the LLM.
`--llm_temperature`: LLM temperature (default: `0` for model's default).
`--llm_max_tokens`: LLM max new tokens (default: `0` for model's default).

**API Options:**
`--api_endpoint`: URL for the OpenAI-compatible API.
`--api_key`: API key for the service.
`--api_model`: Model name to use with the API.

</details>

---

## Supported Models

This tool supports a wide variety of models from Hugging Face and ModelScope.

<details>
<summary>Click to see all supported models</summary>

### WD Caption Models

|            Model             |
|:----------------------------:|:-------------------------------------------------------------------------------:|
|   wd-eva02-large-tagger-v3   |
|    wd-vit-large-tagger-v3    |
|     wd-swinv2-tagger-v3      |
|       wd-vit-tagger-v3       |
|    wd-convnext-tagger-v3     |
|    wd-v1-4-moat-tagger-v2    |
|   wd-v1-4-swinv2-tagger-v2   |
| wd-v1-4-convnextv2-tagger-v2 |
|    wd-v1-4-vit-tagger-v2     |
|  wd-v1-4-convnext-tagger-v2  |
|      wd-v1-4-vit-tagger      |
|   wd-v1-4-convnext-tagger    |
|      Z3D-E621-Convnext       |

### LLM Models

|               Model                |
|:----------------------------------:|:-------------------------------------------------------------------------------------:|
|       joy-caption-pre-alpha        |
|       Joy-Caption-Alpha-One        |
|       Joy-Caption-Alpha-Two        |
|    Joy-Caption-Alpha-Two-Llava     |
| siglip-so400m-patch14-384(Google)  |
|         Meta-Llama-3.1-8B          |
| unsloth/Meta-Llama-3.1-8B-Instruct |
|  Llama-3.1-8B-Lexi-Uncensored-V2   |
|  Llama-3.2-11B-Vision-Instruct  |
|  Llama-3.2-90B-Vision-Instruct  |
| Llama-3.2-11b-vision-uncensored |
| Qwen2-VL-7B-Instruct  |
| Qwen2-VL-72B-Instruct |
| MiniCPM-V-2_6 |
|  Florence-2-large   |
|   Florence-2-base   |

</details>

---

## 🙏 Credits

This project is a fork of and has been significantly updated from the original [wd-llm-caption-cli by fireicewolf](https://github.com/fireicewolf/wd-llm-caption-cli). All credit for the foundational work goes to the original author.

This tool is also built upon the fantastic work of the open-source community. Special thanks to:
- [SmilingWolf](https://huggingface.co/SmilingWolf) for the WD tagger models.
- [fancyfeast](https://huggingface.co/fancyfeast) for the Joy-Caption models.
- [Meta](https://huggingface.co/meta-llama), [Qwen](https://huggingface.co/Qwen), [OpenBMB](https://huggingface.co/openbmb), and [Microsoft](https://huggingface.co/microsoft) for their powerful vision models.