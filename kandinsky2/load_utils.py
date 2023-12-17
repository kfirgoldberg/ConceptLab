import os
from copy import deepcopy
from pathlib import Path

from huggingface_hub import hf_hub_url, cached_download
from omegaconf import DictConfig

from kandinsky2.configs import CONFIG_2_1
from kandinsky2.kandinsky2_1_model import Kandinsky2_1


def download_models_if_not_exist(
        task_type="text2img",
        cache_dir="/tmp/kandinsky2",
        use_auth_token=None,
):
    '''
    Download models in cache folder without model creation.
    '''
    cache_dir = os.path.join(cache_dir, "2_1")
    if task_type == "text2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id="ai-forever/Kandinsky_2.1", filename=model_name)
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id="ai-forever/Kandinsky_2.1", filename=model_name)
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=model_name,
        use_auth_token=use_auth_token,
    )
    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(repo_id="ai-forever/Kandinsky_2.1", filename=prior_name)
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=prior_name,
        use_auth_token=use_auth_token,
    )
    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(repo_id="ai-forever/Kandinsky_2.1", filename=f"text_encoder/{name}")
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en,
            force_filename=name,
            use_auth_token=use_auth_token,
        )
    config_file_url = hf_hub_url(repo_id="ai-forever/Kandinsky_2.1", filename="movq_final.ckpt")
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="movq_final.ckpt",
        use_auth_token=use_auth_token,
    )
    config_file_url = hf_hub_url(repo_id="ai-forever/Kandinsky_2.1", filename="ViT-L-14_stats.th")
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="ViT-L-14_stats.th",
        use_auth_token=use_auth_token,
    )


def get_model(cache_root: Path, device: str) -> Kandinsky2_1:
    try:
        download_models_if_not_exist(task_type="text2img", cache_dir=cache_root)
    except Exception as e:
        print(f'Failed to save model to cache, will try to load if exists, error was {e}')

    config = DictConfig(deepcopy(CONFIG_2_1))
    cache_dir = cache_root / "2_1"

    config["model_config"]["up"] = False
    config["model_config"]["use_fp16"] = True
    config["model_config"]["inpainting"] = False
    config["model_config"]["cache_text_emb"] = False
    config["model_config"]["use_flash_attention"] = False

    config["tokenizer_name"] = os.path.join(cache_dir, "text_encoder")
    config["text_enc_params"]["model_path"] = os.path.join(cache_dir, "text_encoder")
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")

    model_path = os.path.join(cache_dir, "decoder_fp16.ckpt")
    prior_path = os.path.join(cache_dir, "prior_fp16.ckpt")

    # Create model and check that initializer_token exists in both tokenizers and
    # placeholder_token doesn't exist in both tokenizers
    model = Kandinsky2_1(config=config,
                         model_path=model_path,
                         prior_path=prior_path,
                         device=device,
                         task_type="text2img")

    return model
