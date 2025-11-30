# WD LLM Caption (图像描述生成工具)

[![PyPI - Version](https://img.shields.io/pypi/v/wd-llm-caption.svg)](https://pypi.org/project/wd-llm-caption)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wd-llm-caption.svg)](https://pypi.org/project/wd-llm-caption)

一个功能多样、基于 Python 的图像描述生成工具，配备了友好的 Gradio 图形界面。它同时支持本地模型和与 OpenAI 兼容的 API，允许从强大的本地计算机到轻量级的 API 驱动环境的灵活部署。

<img alt="GUI Demo" src="DEMO/DEMO_GUI.png" width="800"/>

## ✨ 功能特性

- **多种描述模式**:
  - **WD 标签**: 使用各种 [WD 系列](https://huggingface.co/SmilingWolf)模型生成 Danbooru 风格的标签。
  - **LLM 描述**: 创建描述性的、符合自然语言习惯的文本描述。
  - **混合模式**: 结合 WD 标签的上下文信息来增强 LLM 的描述效果。
- **灵活的模型支持**:
  - **本地模型**: 支持多种流行的开源视觉模型（例如 Llama-3.2-Vision, Qwen2-VL, Florence-2, Mini-CPM）。
  - **API 模型**: 与任何兼容 OpenAI 的 API 服务（如 OpenAI, vLLM, Ollama）无缝集成。
- **友好的用户界面**:
  - **CLI**: 功能强大的命令行界面，用于批量处理和自动化。
  - **GUI**: 直观的 Gradio 网页界面，用于交互式操作。
- **为不同配置优化**: 只安装您需要的。可选择轻量级的纯 API 配置或功能齐全的本地模型配置。

---

## 🚀 快速上手 (API 用户)

此配置非常适合希望在没有强大本地 GPU 的情况下使用该工具的用户。它依赖外部的 OpenAI 兼容 API 来生成图像描述。

### 1. 安装

打开终端并按照以下步骤操作：

```shell
# 克隆此仓库
git clone https://github.com/fireicewolf/wd-llm-caption-cli.git
cd wd-llm-caption-cli

# 创建并激活 Python 虚拟环境
python -m venv .venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 安装 API 使用所需的最小依赖
pip install -r requirements_api.txt
```

### 2. 使用

启动 GUI，开始生成描述！

```shell
# 运行 GUI
python gui.py

# (可选) 若要允许远程连接，请使用 --listen 参数
python gui.py --listen
```

在图形界面中:
1.  从 “LLM Choice” (LLM 选择) 下拉菜单中选择 **`openai`**。
2.  进入 **"LLM Advanced Options"** (LLM 高级选项) 选项卡，然后点击 **"OpenAI API Settings"** (OpenAI API 设置) 子选项卡。
3.  输入您的 API 服务信息:
    *   **API Endpoint**: 您的服务 URL (例如, `https://api.openai.com/v1`)。
    *   **API Key**: 您的 API 密钥。
    *   **API Model**: 您希望使用的模型名称 (例如, `gpt-4o`)。
4.  现在您已准备好使用 API 生成描述了！

---

## 🛠️ 完整安装 (本地模型)

此配置适用于希望在本地运行模型的高级用户。它需要兼容的 GPU 和用于下载模型的大量磁盘空间。

### 1. 系统要求

- Python 3.10+
- 强烈推荐使用支持 CUDA 的 GPU。

### 2. 安装

```shell
# 克隆仓库并进入目录
git clone https://github.com/fireicewolf/wd-llm-caption-cli.git
cd wd-llm-caption-cli

# 创建并激活虚拟环境
python -m venv .venv
.\venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 1. 安装 PyTorch
# 从 https://pytorch.org/get-started/locally/ 找到适合您系统的命令
# 例如，针对 CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 安装基础和 GUI 依赖
pip install -r requirements.txt
pip install -r requirements_gui.txt

# 3. 根据需要安装模型依赖
# 用于 WD 标签 (ONNX 模型)
# CUDA 11.8
pip install -r requirements_onnx_cu118.txt
# CUDA 12.x
pip install -r requirements_onnx_cu12x.txt

# 用于本地 LLM (基于 Transformers)
pip install -r requirements_llm.txt

# 4. (可选) 安装模型下载器
# 从 Hugging Face 下载 (推荐)
pip install -r requirements_huggingface.txt
# 从 ModelScope 下载
pip install -r requirements_modelscope.txt
```

---

## 📖 使用指南

### 图形界面 (GUI)

使用该工具最简单的方式是通过 Gradio 图形界面。

```shell
python gui.py
```

**GUI 启动选项:**
- `--port <端口号>`: 设置 Web UI 端口 (默认为 `8282`)。
- `--listen`: 允许远程网络连接。
- `--share`: 通过 Gradio 创建一个公共访问链接。
- `--inbrowser`: 自动在浏览器中打开 UI。
- `--log_level <级别>`: 设置控制台日志级别 (例如, `INFO`, `DEBUG`)。

### 命令行 (CLI)

对于批量处理，推荐使用命令行界面。

```shell
# 示例: 使用 API 为文件夹中的所有图像生成描述
python caption.py --data_path /path/to/your/images \
                  --caption_method llm \
                  --llm_choice openai \
                  --api_endpoint "http://localhost:8000/v1" \
                  --api_model "your-model-name"

# 示例: 使用本地 WD 标签器和本地 LLM 生成描述
python caption.py --data_path /path/to/your/images \
                  --caption_method wd+llm \
                  --wd_model_name "wd-swinv2-v3" \
                  --llm_choice llama \
                  --llm_model_name "Llama-3.2-11B-Vision-Instruct"
```

要获取完整的命令行选项列表，请运行：
```shell
python caption.py --help
```
<details>
<summary>点击查看所有 CLI 选项</summary>

`--data_path`: 您的图像数据集路径。
`--recursive`: 处理子目录中的图像。
`--log_level`: 设置控制台日志级别 (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)。
`--save_logs`: 将日志保存到文件。
`--model_site`: 从 `huggingface` 或 `modelscope` 下载模型。
`--models_save_path`: 保存本地模型的目录。
`--use_sdk_cache`: 使用下载 SDK 的默认缓存目录。
`--force_download`: 强制重新下载已存在的模型。
`--caption_method`: 选择 `wd` (标签), `llm` (描述), 或 `wd+llm` (混合)。
`--run_method`: 对于 `wd+llm` 模式，选择 `sync` (逐图处理) 或 `queue` (批量处理)。
`--caption_extension`: 描述文件的扩展名 (默认为 `.txt`)。
`--not_overwrite`: 防止覆盖已存在的描述文件。
`--custom_caption_save_path`: 将描述文件保存到自定义目录。

**WD 标签器选项:**
`--wd_config`: WD 模型配置 JSON 文件的路径。
`--wd_model_name`: 要使用的 WD 模型名称。
`--wd_force_use_cpu`: 强制使用 CPU 进行 WD 模型推理。
`--wd_caption_extension`: 在混合模式下，WD 标签文件的扩展名 (默认为 `.wdcaption`)。
`--wd_remove_underscore`: 将标签中的 `_` 替换为空格。
`--wd_undesired_tags`: 要排除的标签列表，以逗号分隔。
`--wd_threshold`: 添加标签的通用置信度阈值 (默认为 `0.35`)。
`--wd_character_threshold`: 角色标签的置信度阈值。
`--wd_caption_separator`: 标签之间的分隔符 (默认为 `, `)。

**LLM 选项:**
`--llm_choice`: 选择 LLM 类型 (`joy`, `llama`, `qwen`, `minicpm`, `florence`, `openai`)。
`--llm_config`: LLM 模型配置 JSON 文件的路径。
`--llm_model_name`: 要使用的 LLM 名称。
`--llm_patch`: 为 LLM 应用 LoRA 补丁。
`--llm_use_cpu`: 强制使用 CPU 进行 LLM 推理。
`--llm_dtype`: LLM 数据类型 (`fp16`, `bf16`, `fp32`)。
`--llm_qnt`: LLM 量化 (`none`, `4bit`, `8bit`)。
`--llm_caption_extension`: 在混合模式下，LLM 描述文件的扩展名 (默认为 `.llmcaption`)。
`--llm_user_prompt`: 用于 LLM 的自定义用户提示。
`--llm_temperature`: LLM 温度 (默认为 `0`，表示使用模型自己的默认值)。
`--llm_max_tokens`: LLM 输出的最大 token 数 (默认为 `0`，表示使用模型自己的默认值)。

**API 选项:**
`--api_endpoint`: 兼容 OpenAI 的 API 的 URL。
`--api_key`: API 服务的密钥。
`--api_model`: 通过 API 使用的模型名称。

</details>

---

## 📦 支持的模型

本工具支持来自 Hugging Face 和 ModelScope 的多种模型。

<details>
<summary>点击查看所有支持的模型</summary>

### WD 描述模型

|            模型             |                                Hugging Face 链接                                |
|:----------------------------:|:-------------------------------------------------------------------------------:|
|   wd-eva02-large-tagger-v3   |   [Hugging Face](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)   |
|    wd-vit-large-tagger-v3    |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3)    |
|     wd-swinv2-tagger-v3      |     [Hugging Face](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)      |
|       wd-vit-tagger-v3       |       [Hugging Face](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3)       |
|    wd-convnext-tagger-v3     |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3)     |
|    wd-v1-4-moat-tagger-v2    |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)    |
|   wd-v1-4-swinv2-tagger-v2   |   [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)   |
| wd-v1-4-convnextv2-tagger-v2 | [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2) |
|    wd-v1-4-vit-tagger-v2     |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)     |
|  wd-v1-4-convnext-tagger-v2  |  [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)  |
|      wd-v1-4-vit-tagger      |      [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger)      |
|   wd-v1-4-convnext-tagger    |   [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger)    |
|      Z3D-E621-Convnext       |         [Hugging Face](https://huggingface.co/toynya/Z3D-E621-Convnext)         |

### LLM 模型

|               模型                |                                   Hugging Face 链接                                   |
|:----------------------------------:|:-------------------------------------------------------------------------------------:|
|       joy-caption-pre-alpha        |    [Hugging Face](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)     |
|       Joy-Caption-Alpha-One        |    [Hugging Face](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one)     |
|       Joy-Caption-Alpha-Two        |    [Hugging Face](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)     |
|    Joy-Caption-Alpha-Two-Llava     | [Hugging Face](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) |
| siglip-so400m-patch14-384(Google)  |        [Hugging Face](https://huggingface.co/google/siglip-so400m-patch14-384)        |
|         Meta-Llama-3.1-8B          |          [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)          |
| unsloth/Meta-Llama-3.1-8B-Instruct |       [Hugging Face](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)       |
|  Llama-3.1-8B-Lexi-Uncensored-V2   |   [Hugging Face](https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2)   |
|  Llama-3.2-11B-Vision-Instruct  |  [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)   |
|  Llama-3.2-90B-Vision-Instruct  |  [Hugging Face](https://huggingface.co/meta-llama/LLama-3.2-90B-Vision-Instruct)   |
| Llama-3.2-11b-vision-uncensored | [Hugging Face](https://huggingface.co/Guilherme34/Llama-3.2-11b-vision-uncensored) |
| Qwen2-VL-7B-Instruct  | [Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)  |
| Qwen2-VL-72B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) |
| MiniCPM-V-2_6 | [Hugging Face](https://huggingface.co/openbmb/MiniCPM-V-2_6) |
|  Florence-2-large   |  [Hugging Face](https://huggingface.co/microsoft/Florence-2-large)   |
|   Florence-2-base   |   [Hugging Face](https://huggingface.co/microsoft/Florence-2-base)   |

</details>

---

## 🙏 致谢

本项目是 [fireicewolf/wd-llm-caption-cli](https://github.com/fireicewolf/wd-llm-caption-cli) 的一个分支，并在此基础上进行了大量更新。所有基础性工作的功劳归于原作者。

本工具的实现也离不开开源社区的杰出工作。特别感谢：
- [SmilingWolf](https://huggingface.co/SmilingWolf) 提供的 WD 标签模型。
- [fancyfeast](https://huggingface.co/fancyfeast) 提供的 Joy-Caption 模型。
- [Meta](https://huggingface.co/meta-llama)、[Qwen](https://huggingface.co/Qwen)、[OpenBMB](https://huggingface.co/openbmb) 和 [Microsoft](https://huggingface.co/microsoft) 提供的强大的视觉模型。
