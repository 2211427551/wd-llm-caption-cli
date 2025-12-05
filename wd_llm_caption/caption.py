import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from .utils.download import download_models
from .utils.image import get_image_paths
from .utils.inference import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_WITH_WD,
    DEFAULT_USER_PROMPT_WITHOUT_WD,
    LLM,
    Tagger,
    get_caption_file_path,
)
from .utils.logger import Logger, print_title


DEFAULT_MODELS_SAVE_PATH = str(os.path.join(os.getcwd(), "models"))


class Caption:
    def __init__(self):
        # Set flags
        self.use_wd = False
        self.use_llm = False # Simplified flag

        self.my_logger = None

        self.wd_model_path = None
        self.wd_tags_csv_path = None
        self.llm_models_paths = None

        self.my_tagger = None
        self.my_llm = None

    def check_path(
            self,
            args: argparse.Namespace
    ):
        if not args.data_path:
            print("`data_path` not defined, use `--data_path` add your datasets path!!!")
            raise ValueError
        if not os.path.exists(args.data_path):
            print(f"`{args.data_path}` not exists!!!")
            raise FileNotFoundError

    def set_logger(
            self,
            args: argparse.Namespace
    ):
        # Set logger
        if args.save_logs:
            workspace_path = os.getcwd()
            data_dir_path = Path(args.data_path)

            log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

            if args.custom_caption_save_path:
                log_file_path = Path(args.custom_caption_save_path)

            log_time = datetime.now().strftime('%Y%m%d_%H%M%S')

            if os.path.exists(data_dir_path):
                log_name = os.path.basename(data_dir_path)
            else:
                print(f'{data_dir_path} NOT FOUND!!!')
                raise FileNotFoundError

            log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
            log_file = os.path.join(log_file_path, log_file) \
                if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
        else:
            log_file = None

        if str(args.log_level).lower() in 'debug, info, warning, error, critical':
            self.my_logger = Logger(args.log_level, log_file).logger
            self.my_logger.info(f'Set log level to "{args.log_level}"')
        else:
            self.my_logger = Logger('INFO', log_file).logger
            self.my_logger.warning('Invalid log level, set log level to "INFO"!')

        if args.save_logs:
            self.my_logger.info(f'Log file will be saved as "{log_file}".')

    def download_models(
            self,
            args: argparse.Namespace
    ):
        # Set flags
        self.use_wd = True if args.caption_method in ["wd", "wd+llm"] else False
        self.use_llm = True if args.caption_method in ["llm", "wd+llm"] else False
        
        # Set models save path
        if os.path.exists(Path(args.models_save_path)):
            models_save_path = Path(args.models_save_path)
        else:
            self.my_logger.warning(
                f"Models save path not defined or not exists, will download models into `{DEFAULT_MODELS_SAVE_PATH}`...")
            models_save_path = Path(DEFAULT_MODELS_SAVE_PATH)

        if self.use_wd:
            wd_config_file = Path(args.wd_config) if args.wd_config else os.path.join(Path(__file__).parent, 'configs', 'default_wd.json')
            self.wd_model_path, self.wd_tags_csv_path = download_models(
                logger=self.my_logger, models_type="wd", args=args,
                config_file=wd_config_file, models_save_path=models_save_path,
            )

        if self.use_llm:
            llm_configs = {
                "joy": "default_joy.json",
                "llama": "default_llama_3.2V.json",
                "qwen": "default_qwen2_vl.json",
                "minicpm": "default_minicpm.json",
                "florence": "default_florence.json",
                "openai": "default_openai.json",
            }
            config_filename = llm_configs.get(args.llm_choice)
            if config_filename:
                llm_config_file = Path(args.llm_config) if args.llm_config else os.path.join(Path(__file__).parent, 'configs', config_filename)
                if args.llm_choice == "openai":
                    self.my_logger.info("Using OpenAI-compatible API, no model download required.")
                    self.llm_models_paths = None
                else:
                    self.llm_models_paths = download_models(
                        logger=self.my_logger, models_type=args.llm_choice, args=args,
                        config_file=llm_config_file, models_save_path=models_save_path,
                    )

    def load_wd_model(self, args: argparse.Namespace):
        if self.my_tagger is None:
            self.my_logger.info("Loading WD Tagger model...")
            self.my_tagger = Tagger(
                logger=self.my_logger, args=args,
                model_path=self.wd_model_path, tags_csv_path=self.wd_tags_csv_path
            )
            self.my_tagger.load_model()
            self.my_logger.info("WD Tagger model loaded.")

    def load_llm_model(self, args: argparse.Namespace):
        if self.my_llm is None:
            self.my_logger.info(f"Loading LLM ({args.llm_choice}) model...")
            self.my_llm = LLM(
                logger=self.my_logger, models_type=args.llm_choice,
                models_paths=self.llm_models_paths, args=args,
            )
            self.my_llm.load_model()
            self.my_logger.info(f"LLM ({args.llm_choice}) model loaded.")

    def load_models(self, args: argparse.Namespace):
        """Loads all models required by the caption_method."""
        if self.use_wd:
            self.load_wd_model(args)
        if self.use_llm:
            self.load_llm_model(args)

    def _process_image_wd(self, image_path, args):
        try:
            wd_caption_file = get_caption_file_path(
                self.my_logger, data_path=args.data_path, image_path=Path(image_path),
                custom_caption_save_path=args.custom_caption_save_path, caption_extension=args.wd_caption_extension
            )
            if args.skip_exists and os.path.isfile(wd_caption_file):
                return f"Skipped (exists): {wd_caption_file}"

            image = Image.open(image_path).convert("RGB")
            tag_text, _, _, _ = self.my_tagger.get_tags(image=image)

            if not (args.not_overwrite and os.path.isfile(wd_caption_file)):
                with open(wd_caption_file, "w", encoding="utf-8") as f:
                    f.write(tag_text + "\n")
                return f"Tagged: {wd_caption_file}"
            else:
                return f"Skipped (overwrite disabled): {wd_caption_file}"
        except Exception as e:
            self.my_logger.error(f"Failed to WD-tag {image_path}: {e}")
            return f"Failed: {image_path}"

    def _process_image_llm(self, image_path, args):
        try:
            # Define file paths
            final_caption_ext = args.caption_extension
            llm_caption_file = get_caption_file_path(
                self.my_logger, data_path=args.data_path, image_path=Path(image_path),
                custom_caption_save_path=args.custom_caption_save_path,
                caption_extension=args.llm_caption_extension if args.save_caption_together else final_caption_ext
            )

            if args.skip_exists and os.path.isfile(llm_caption_file):
                 # Also check for the final combined file if that option is enabled
                if args.save_caption_together:
                    together_caption_file = get_caption_file_path(
                        self.my_logger, data_path=args.data_path, image_path=Path(image_path),
                        custom_caption_save_path=args.custom_caption_save_path, caption_extension=final_caption_ext
                    )
                    if os.path.isfile(together_caption_file):
                        return f"Skipped (exists): {together_caption_file}"
                else:
                    return f"Skipped (exists): {llm_caption_file}"

            # Read WD tags if needed
            tag_text = ""
            if args.caption_method == "wd+llm" or (args.caption_method == "llm" and args.llm_read_wd_caption):
                wd_caption_file = get_caption_file_path(
                    self.my_logger, data_path=args.data_path, image_path=Path(image_path),
                    custom_caption_save_path=args.custom_caption_save_path, caption_extension=args.wd_caption_extension
                )
                if os.path.exists(wd_caption_file):
                    with open(wd_caption_file, "r", encoding="utf-8") as f:
                        tag_text = f.read().strip()
                else:
                    self.my_logger.warning(f"WD caption file not found for {image_path}, proceeding without tags.")

            # Get LLM caption
            image = Image.open(image_path).convert("RGB")
            user_prompt = str(args.llm_user_prompt).format(wd_tags=tag_text)
            caption_text = self.my_llm.get_caption(
                image=image, system_prompt=str(args.llm_system_prompt),
                user_prompt=user_prompt, temperature=args.llm_temperature, max_new_tokens=args.llm_max_tokens
            )

            # Save captions
            if args.save_caption_together:
                together_caption_file = get_caption_file_path(
                    self.my_logger, data_path=args.data_path, image_path=Path(image_path),
                    custom_caption_save_path=args.custom_caption_save_path, caption_extension=final_caption_ext
                )
                if not (args.not_overwrite and os.path.isfile(together_caption_file)):
                    with open(together_caption_file, "w", encoding="utf-8") as f:
                        f.write(f"{tag_text} {args.save_caption_together_seperator} {caption_text}\n")
                    return f"Saved combined caption to {together_caption_file}"
                else:
                    return f"Skipped combined (overwrite disabled): {together_caption_file}"
            else:
                if not (args.not_overwrite and os.path.isfile(llm_caption_file)):
                    with open(llm_caption_file, "w", encoding="utf-8") as f:
                        f.write(caption_text + "\n")
                    return f"Saved LLM caption to {llm_caption_file}"
                else:
                    return f"Skipped LLM caption (overwrite disabled): {llm_caption_file}"

        except Exception as e:
            self.my_logger.error(f"Failed to LLM-caption {image_path}: {e}")
            return f"Failed: {image_path}"

    def run_inference(self, args: argparse.Namespace):
        start_inference_time = time.monotonic()
        image_paths = get_image_paths(logger=self.my_logger, path=Path(args.data_path), recursive=args.recursive)
        self.my_logger.info(f"Found {len(image_paths)} images to process.")

        # --- Stage 1: WD Tagging ---
        if self.use_wd:
            self.my_logger.info("--- Starting Stage 1: WD Tagging ---")
            self.load_wd_model(args)
            
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(self._process_image_wd, img_path, args) for img_path in image_paths]
                for future in tqdm(as_completed(futures), total=len(image_paths), desc="WD Tagging"):
                    self.my_logger.debug(future.result())
            
            if self.use_llm:
                self.my_logger.info("Unloading WD model to free VRAM for LLM...")
                self.unload_models(unload_llm=False) # Unload only WD

        # --- Stage 2: LLM Captioning ---
        if self.use_llm:
            self.my_logger.info("--- Starting Stage 2: LLM Captioning ---")
            self.load_llm_model(args)

            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(self._process_image_llm, img_path, args) for img_path in image_paths]
                for future in tqdm(as_completed(futures), total=len(image_paths), desc="LLM Captioning"):
                    self.my_logger.debug(future.result())
        
        total_inference_time = time.monotonic() - start_inference_time
        # ... (time formatting logic remains the same)
        days = total_inference_time // (24 * 3600)
        total_inference_time %= (24 * 3600)
        hours = total_inference_time // 3600
        total_inference_time %= 3600
        minutes = total_inference_time // 60
        seconds = total_inference_time % 60
        days_str = f"{days:.0f} Day(s) " if days > 0 else ""
        hours_str = f"{hours:.0f} Hour(s) " if hours > 0 or (days and hours == 0) else ""
        minutes_str = f"{minutes:.0f} Min(s) " if minutes > 0 or (hours and minutes == 0) else ""
        seconds_str = f"{seconds:.2f} Sec(s)"
        self.my_logger.info(f"All work done in {days_str}{hours_str}{minutes_str}{seconds_str}.")

    def unload_models(self, unload_wd=True, unload_llm=True):
        unloaded = False
        if self.use_wd and unload_wd and self.my_tagger:
            self.my_tagger.unload_model()
            self.my_tagger = None
            self.my_logger.info("WD Tagger model unloaded.")
        if self.use_llm and unload_llm and self.my_llm:
            unloaded = self.my_llm.unload_model()
            self.my_llm = None
            self.my_logger.info("LLM model unloaded.")
        
        if unload_wd and unload_llm:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if self.my_logger: self.my_logger.info("CUDA cache emptied.")
            except ImportError:
                if self.my_logger: self.my_logger.debug("Torch not installed, skipping CUDA cache clear.")
            except Exception as e:
                if self.my_logger: self.my_logger.error(f"Failed to empty CUDA cache: {e}")
        return unloaded


def setup_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    base_args = args.add_argument_group("Base")
    base_args.add_argument('--data_path', type=str, help='path for data.')
    base_args.add_argument('--recursive', action='store_true', help='Include recursive dirs')

    log_args = args.add_argument_group("Logs")
    log_args.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='set log level, default is `INFO`')
    log_args.add_argument('--save_logs', action='store_true', help='save log file.')

    download_args = args.add_argument_group("Download")
    download_args.add_argument('--model_site', type=str, choices=['huggingface', 'modelscope'], default='huggingface', help='download models from model site, default is `huggingface`.')
    download_args.add_argument('--models_save_path', type=str, default=DEFAULT_MODELS_SAVE_PATH, help='path to save models, default is `models`.')
    download_args.add_argument('--use_sdk_cache', action='store_true', help='use sdk\'s cache dir to store models.')
    download_args.add_argument('--download_method', type=str, choices=["SDK", "URL"], default='SDK', help='download models via SDK or URL, default is `SDK`.')
    download_args.add_argument('--force_download', action='store_true', help='force download even file exists.')
    download_args.add_argument('--skip_download', action='store_true', help='skip download if exists.')

    caption_args = args.add_argument_group("Caption")
    caption_args.add_argument('--caption_method', type=str, default='wd+llm', choices=['wd', 'llm', 'wd+llm'], help='method for caption, default is `wd+llm`.')
    caption_args.add_argument('--num_workers', type=int, default=1, help='Number of concurrent workers for batch processing. Default is 1.')
    caption_args.add_argument('--caption_extension', type=str, default='.txt', help='extension of caption file.')
    caption_args.add_argument('--save_caption_together', action='store_true', help='Save WD tags and LLM captions in one file.')
    caption_args.add_argument('--save_caption_together_seperator', default='|', help='Seperator between WD and LLM captions.')
    caption_args.add_argument('--image_size', type=int, default=1024, help='resize image to suitable, default is `1024`.')
    caption_args.add_argument('--skip_exists', action='store_true', help='not caption file if caption exists.')
    caption_args.add_argument('--not_overwrite', action='store_true', help='not overwrite caption file if exists.')
    caption_args.add_argument('--custom_caption_save_path', type=str, default=None, help='custom caption file save path.')
    # Deprecate run_method for new logic
    caption_args.add_argument('--run_method', type=str, default='sync', choices=['sync', 'queue'], help='DEPRECATED. Concurrency is now default for batch mode.')

    wd_args = args.add_argument_group("WD Caption")
    wd_args.add_argument('--wd_config', type=str, help='configs json for wd tagger models, default is `default_wd.json`')
    wd_args.add_argument('--wd_model_name', type=str, help='wd tagger model name, default is `wd-eva02-large-tagger-v3`.')
    wd_args.add_argument('--wd_force_use_cpu', action='store_true', help='force use cpu for wd models inference.')
    wd_args.add_argument('--wd_caption_extension', type=str, default=".wdcaption", help='extension for wd captions files, default is `.wdcaption`.' )
    wd_args.add_argument('--wd_remove_underscore', action='store_true', help='replace underscores with spaces in the output tags.')
    wd_args.add_argument("--wd_undesired_tags", type=str, default='', help='comma-separated list of undesired tags to remove.')
    wd_args.add_argument('--wd_tags_frequency', action='store_true', help='Show frequency of tags for images.')
    wd_args.add_argument('--wd_threshold', type=float, default=0.35, help='threshold of confidence to add a tag, default is `0.35`.')
    wd_args.add_argument('--wd_general_threshold', type=float, default=None, help='threshold for general category.')
    wd_args.add_argument('--wd_character_threshold', type=float, default=None, help='threshold for character category.')
    wd_args.add_argument('--wd_add_rating_tags_to_first', action='store_true', help='Adds rating tags to the first.')
    wd_args.add_argument('--wd_add_rating_tags_to_last', action='store_true', help='Adds rating tags to the last.')
    wd_args.add_argument('--wd_character_tags_first', action='store_true', help='Put character tags before the general tags.')
    wd_args.add_argument('--wd_always_first_tags', type=str, default=None, help='comma-separated list of tags to always put at the beginning.')
    wd_args.add_argument('--wd_caption_separator', type=str, default=', ', help='Separator for tags, default is `, `.')
    wd_args.add_argument('--wd_tag_replacement', type=str, default=None, help='tag replacement in the format of `source1,target1;...`.')
    wd_args.add_argument('--wd_character_tag_expand', action='store_true', help='expand tag tail parenthesis to another tag for character tags.')

    llm_args = args.add_argument_group("LLM Caption")
    llm_args.add_argument('--llm_choice', type=str, default='llama', choices=['joy', 'llama', 'qwen', 'minicpm', 'florence', 'openai'], help='select llm models, default is `llama`.')
    llm_args.add_argument('--llm_config', type=str, help='config json for LLM Caption models.')
    llm_args.add_argument('--llm_model_name', type=str, help='model name for inference.')
    llm_args.add_argument('--llm_patch', action='store_true', help='patch llm with lora for uncensored.')
    llm_args.add_argument('--llm_use_cpu', action='store_true', help='load LLM models use cpu.')
    llm_args.add_argument('--llm_dtype', type=str, choices=["fp16", "bf16", "fp32"], default='fp16', help='choice joy LLM load dtype, default is `fp16`.')
    llm_args.add_argument('--llm_qnt', type=str, choices=["none", "4bit", "8bit"], default='none', help='Enable quantization for LLM, default is `none`.')
    llm_args.add_argument('--llm_caption_extension', type=str, default='.llmcaption', help='extension of LLM caption file, default is `.llmcaption`')
    llm_args.add_argument('--llm_read_wd_caption', action='store_true', help='LLM will read wd tags for inference.')
    llm_args.add_argument('--llm_caption_without_wd', action='store_true', help='LLM will not read WD tags for inference.')
    llm_args.add_argument('--llm_system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help='system prompt for llm caption.')
    llm_args.add_argument('--llm_user_prompt', type=str, default=DEFAULT_USER_PROMPT_WITHOUT_WD, help='user prompt for llm caption.')
    llm_args.add_argument('--llm_temperature', type=float, default=0, help='temperature for LLM model, default is `0`.')
    llm_args.add_argument('--llm_max_tokens', type=int, default=0, help='max tokens for LLM model output, default is `0`.')

    openai_args = args.add_argument_group("OpenAI API")
    openai_args.add_argument('--api_endpoint', type=str, help='OpenAI-compatible API endpoint URL')
    openai_args.add_argument('--api_key', type=str, help='API key for OpenAI-compatible API')
    openai_args.add_argument('--api_model', type=str, help='Model name for OpenAI-compatible API')

    gradio_args = args.add_argument_group("Gradio dummy args, no effects")
    gradio_args.add_argument('--theme', type=str, default="default")
    gradio_args.add_argument('--port', type=int, default="8282")
    gradio_args.add_argument('--listen', action='store_true')
    gradio_args.add_argument('--share', action='store_true')
    gradio_args.add_argument('--inbrowser', action='store_true')
    return args.parse_args()


def main():
    print_title()
    get_args = setup_args()
    my_caption = Caption()
    my_caption.check_path(get_args)
    my_caption.set_logger(get_args)
    my_caption.download_models(get_args)
    # Models are now loaded on-demand in run_inference
    my_caption.run_inference(get_args)
    my_caption.unload_models()


if __name__ == "__main__":
    main()