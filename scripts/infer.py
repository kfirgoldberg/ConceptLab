from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import PIL
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from PIL import Image

from kandinsky2 import load_utils
from training.dataset import object_templates


@dataclass
class InferConfig:
    # Output path
    output_dir: Path
    # Path to checkpoint to load.
    learned_embeds_path: Optional[str] = None
    # Prompts, seperated by ,
    prompts: Optional[List[str]] = None
    # Number of samples per prompt
    samples_per_prompt: int = 1
    # Path to pretrained model WITHOUT 2_1 folder
    cache_root: Path = Path('/tmp/kandinsky2')
    # GPU device
    device: str = "cuda:0"
    # The resolution for input images, all the images will be resized to this size
    img_size: int = 512
    # Batch size (per device) for the training dataloader.
    batch_size: int = 1
    # Negative prompt
    negative_prompt: str = ""
    # Save embeds
    save_embeds: bool = False




def save_images(model, save_path, prompt, img_prompt=None, img_size=512, seed=None, negative_prompt: str = "") \
        -> List[PIL.Image.Image]:
    if img_prompt is None:
        img_prompt = prompt
    images = model.generate_text2img(
        prompt,
        num_steps=50,
        batch_size=1,
        guidance_scale=7.5,
        h=img_size,
        w=img_size,
        sampler="p_sampler",
        prior_cf_scale=4,
        prior_steps="5",
        img_prompt=img_prompt,
        negative_prior_prompt=negative_prompt,
        seed=seed
    )
    images = [*images]
    gen_images = np.hstack([np.array(img) for img in images])
    Image.fromarray(gen_images).save(save_path)
    return images


@pyrallis.wrap()
def main(cfg: InferConfig):
    model = load_utils.get_model(cfg.cache_root, cfg.device)

    if cfg.learned_embeds_path is not None:
        learned_embeds_dict = torch.load(cfg.learned_embeds_path)

        # Convert the initializer_token, placeholder_token to ids for tokenizer2
        # and add placeholder_token to tokenizer2
        for token, embedding in learned_embeds_dict['t2'].items():
            print(f'Adding {token} to tokenizer2...')
            t2p_index_to_add = len(model.tokenizer2.encoder)
            model.tokenizer2.encoder[token] = t2p_index_to_add
            model.tokenizer2.decoder[t2p_index_to_add] = token
            model.tokenizer2.cache[token] = token

            t2_place_token_id = model.tokenizer2.encode(token)[0]

            old_vocab_size, t2_embed_size = model.clip_model.token_embedding.weight.shape

            # Create new embeddings
            # Copy old weights to the new embeddings and initialize new token
            new_embed = nn.Embedding(old_vocab_size + 1, t2_embed_size).to(cfg.device)

            current_embeds = model.clip_model.token_embedding.weight.data

            new_embed.weight.data[:old_vocab_size, :] = current_embeds.clone()
            new_embed.weight.data[t2_place_token_id, :] = embedding

            model.clip_model.token_embedding = deepcopy(new_embed)

    output_dir = cfg.output_dir
    images_root = output_dir / "infer_images"
    images_root.mkdir(exist_ok=True, parents=True)

    prompts = cfg.prompts
    if prompts is None:
        prompts = [p.format(a='a', token='{}') for p in set(object_templates)]
    seeds = range(cfg.samples_per_prompt)
    for prompt in prompts:
        if cfg.learned_embeds_path is not None:
            img_prompt = prompt.format(token)
        else:
            img_prompt = prompt
        text_prompt = f""
        neg_prompt = cfg.negative_prompt
        for seed in seeds:
            image_save_path = images_root / f"{img_prompt.replace(' ', '_')}_images_seed_{seed}.jpg"
            print(image_save_path)
            images = save_images(model,
                                 save_path=str(image_save_path),
                                 prompt=text_prompt,
                                 img_prompt=img_prompt,
                                 seed=seed,
                                 negative_prompt=neg_prompt)
            if cfg.save_embeds:
                embeds = [model.encode_images(image, is_pil=True) for image in images]
                embeds = torch.stack(embeds)
                torch.save(embeds, output_dir / f"{img_prompt.replace(' ', '_')}_embeds_seed_{seed}.pt")
    print('Done')


if __name__ == "__main__":
    main()
