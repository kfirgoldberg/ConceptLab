import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from training.templates import style_templates, object_templates, \
    LearnableProperties, PREFIXES


class ConceptDataset(Dataset):
    def __init__(
            self,
            placeholder_token: str,
            learnable_property: LearnableProperties = LearnableProperties.object,
    ):
        self.placeholder_token = placeholder_token
        if learnable_property == LearnableProperties.style:
            self.templates = style_templates
        elif learnable_property == LearnableProperties.object:
            self.templates = object_templates
        else:
            raise ValueError(f"Unknown learnable property {learnable_property}")

    def __len__(self):
        return 5  # Doesn't really matter as we use steps

    def __getitem__(self, i: int):
        example = {}
        placeholder_string = self.placeholder_token
        template = random.choice(self.templates)
        if '{a}' in template:
            template = template.format(a=random.choice(PREFIXES), token='{token}')
        text = template.format(token=placeholder_string)
        example["template"] = template
        example["text"] = text
        return example


class EvolutionDataset(Dataset):

    def __init__(
            self,
            placeholder_token: str,
            embeds_dirs: Optional[List[Path]] = None,
            embeds: Optional[List[torch.Tensor]] = None,
            learnable_property: LearnableProperties = LearnableProperties.object,
    ):
        assert (embeds_dirs is not None and embeds is None) or (
                embeds_dirs is None and embeds is not None
        ), "Either embeds_dirs or embeds must be provided"
        if embeds_dirs is not None:
            self.embeds_dirs = embeds_dirs
            self.embeds: List[torch.Tensor] = load_embeds(embeds_dirs)
        else:
            self.embeds = embeds
        self.placeholder_token = placeholder_token
        if learnable_property == LearnableProperties.style:
            self.templates = style_templates
        elif learnable_property == LearnableProperties.object:
            self.templates = object_templates
        else:
            raise ValueError(f"Unknown learnable property {learnable_property}")

    def __len__(self):
        return min([embed.shape[0] for embed in self.embeds])

    def __getitem__(self, i: int):
        example = {}
        chosen_embeds = []
        for embed in self.embeds:
            curr_rand_embed = random.choice(embed)
            chosen_embeds.append(curr_rand_embed)
        placeholder_string = self.placeholder_token
        template = random.choice(self.templates)
        if '{a}' in template:
            template = template.format(a=random.choice(PREFIXES), token='{token}')
        text = template.format(token=placeholder_string)
        example["template"] = template
        example["text"] = text
        example["positive_embeds"] = chosen_embeds
        return example


def load_embeds(embeds_dirs: List[Path]) -> List[torch.Tensor]:
    all_embeds = []
    for embeds_dir in embeds_dirs:
        embeds_dir = Path(embeds_dir)
        embeds_paths = list(embeds_dir.glob("*.pt"))
        if len(embeds_paths) == 0:
            raise ValueError(f"Could not find any embeddings in {embeds_dir}")
        curr_dir_embeds = []
        for embed_path in embeds_paths:
            curr_embed = torch.load(embed_path)
            curr_dir_embeds.append(curr_embed)
        all_embeds.append(torch.cat(curr_dir_embeds, dim=0))
    return all_embeds
