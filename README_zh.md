# WD LLM Caption

这是一个利用大型语言模型（LLM）为图片生成详细描述（Caption）的工具。

本项目修改自 [llm-caption](https://github.com/Maki橘子/llm-caption)，并在此基础上进行二次开发。

## 主要功能

1.  **多种模型支持**：项目支持多种主流的视觉语言模型（VLM），包括：
    *   **WD-tagger**：一个专门用于动漫风格图片的优秀标签器，可作为前置处理步骤。
    *   **Qwen-VL** (通义千问-VL)
    *   **LLaMA-3.2V**
    *   **Florence-2**
    *   **MiniCPM-Llama3-V 2.5**
    *   **Joy-Caption**
    *   **兼容 OpenAI API 的模型服务**：
        *   这并非特指 OpenAI 官方模型（如 GPT-4o），而是指本项目可以作为一个客户端，调用任何提供 OpenAI 兼容 API 接口的第三方或本地模型服务（例如 SiliconFlow, Ollama, vLLM 等）。
        *   你只需在界面中提供服务的 `API Endpoint`、`API Key`（如果需要）和 `模型名称` 即可接入。

2.  **灵活的组合模式**：
    *   **WD+LLM**：先使用 WD-tagger 生成标签，然后将这些标签作为参考信息，引导 LLM 生成更丰富、更准确的描述。
    *   **仅 WD**：只使用 WD-tagger 生成关键词标签。
    *   **仅 LLM**：直接使用视觉语言模型生成描述。

3.  **多种运行界面**：
    *   **GUI 模式**：提供了一个基于 Gradio 的图形用户界面，支持单张图片和文件夹批处理，操作直观方便。
    *   **API 模式**：可以通过 FastAPI 启动一个 API 服务，让其他程序调用此项目的图片描述能力。

4.  **高度可配置**：
    *   用户可以通过 JSON 配置文件 (`wd_llm_caption/configs/`) 来自定义模型的参数。
    *   在 GUI 中可以方便地调整 WD-tagger 的阈值、后处理选项以及 LLM 的提示词（Prompt）、温度等超参数。

## TODO

- [ ] 优化日志和输出
- [ ] 增加更多模型支持
- [ ] 增加 LoRA 支持
- [ ] 增加视频打标支持

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/wd-llm-caption-cli.git
cd wd-llm-caption-cli

# 2. 创建虚拟环境 (推荐)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate  # Windows

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 根据你需要使用的模型，安装对应的依赖包
# 例如，使用 Qwen-VL 和 ONNX:
pip install -r requirements_llm.txt
pip install -r requirements_onnx_cu12x.txt # 根据你的 CUDA 版本选择
```

## 使用

```bash
# 启动 Gradio 图形界面
python -m wd_llm_caption.gui
```

访问 `http://127.0.0.1:8282` 即可打开操作界面。

**在 GUI 中使用兼容 OpenAI 的 API：**

1.  在 `Caption method` 中选择包含 `LLM` 的模式。
2.  在 `Choice LLM` 中选择 `OpenAI`。
3.  此时会显示 `OpenAI API Settings` 区域。
4.  **API Endpoint**：填入你的服务地址，例如 `http://127.0.0.1:8000/v1`。
5.  **API Key**：填入你的 API 密钥（如果服务需要）。
6.  **Model Selection**：
    *   你可以点击 `Get Models from Endpoint` 按钮自动从你的服务获取可用的模型列表。
    *   或者，在 `Custom Model Name` 中手动输入你想使用的模型名称。
7.  加载模型并开始推理。