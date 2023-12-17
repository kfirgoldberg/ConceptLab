from dataclasses import dataclass, field
import matplotlib
import pyrallis
import sys
import torch
from PIL import Image
from accelerate.utils import set_seed
from pathlib import Path
from typing import List, Optional
from training.coach import Coach
from training.dataset import EvolutionDataset
from training.templates import LearnableProperties

matplotlib.use('Agg')  # Set the backend to non-interactive (Agg)


@dataclass
class TrainEvolutionConfig:
    # A token to use as initializer word.
    initializer_token: str
    # list of dirs paths to parents images
    parents_images_dirs: List[Path]
    # A token to use as a placeholder for the concept.
    placeholder_token: str = 'myevolution'
    # Weights for each of the mixed objects
    mix_weights: Optional[List[float]] = None
    # Path to pretrained model WITHOUT 2_1 folder
    cache_root: Path = Path('/tmp/kandinsky2')
    # Defines which prompts to use
    learnable_property: LearnableProperties = LearnableProperties.object
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: Path = Path('concept-lab-evolution')
    # GPU device
    device: str = 'cuda:0'
    # The resolution for input images, all the images will be resized to this size
    img_size: int = 512
    # Batch size (per device) for the training dataloader
    train_batch_size: int = 1
    # Number of steps to train for
    steps: int = 2000
    # Initial learning rate (after the potential warmup period) to use
    learning_rate: float = 1e-4
    # Dataloader num workers.
    num_workers: int = 0
    # The beta1 parameter for the Adam optimizer.
    adam_beta1: float = 0.9
    # The beta2 parameter for the Adam optimizer
    adam_beta2: float = 0.999
    # Weight decay to use
    adam_weight_decay: float = 1e-2
    # Epsilon value for the Adam optimizer
    adam_epsilon: float = 1e-08
    # How often save images. Values less zero - disable saving
    log_image_frequency: int = 250
    # How often save embeddings
    log_embed_frequency: int = 250
    # A seed for reproducible training
    seed: Optional[int] = None
    # Comma-separated seeds for reproducible inference.
    inference_seeds: List[int] = field(default_factory=lambda: [42, 500])

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.images_root = self.output_dir / "images"
        self.images_root.mkdir(exist_ok=True, parents=True)
        if self.mix_weights is None:
            self.mix_weights = [1 / len(self.parents_images_dirs)] * len(self.parents_images_dirs)
        assert len(self.mix_weights) == len(self.parents_images_dirs), "Weights and parents images dirs must have the same length"


class CoachEvolution(Coach):
    def __init__(self, config: TrainEvolutionConfig):
        self.cfg = config
        (self.cfg.output_dir / 'run_cfg.yaml').write_text(pyrallis.dump(self.cfg))
        (self.cfg.output_dir / 'run.sh').write_text(f'python {Path(__file__).name} {" ".join(sys.argv)}')
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)
        self.model = self.get_model()
        self.t2_eos_token_id = len(self.model.tokenizer2.encoder) - 1
        self.t2_place_token_id = self.update_tokenizer()
        self.set_model_gradient_flow()
        self.optimizer = self.get_optimizer()
        self.train_dataloader = self.get_train_dataloader()
        # Save original embeddings from both models
        self.orig_t2_params = self.model.clip_model.token_embedding.weight.data.clone()
        self.weight_dtype = self.model.model.dtype
        self.train_step = 0

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        embeds = []
        for parent_dir in self.cfg.parents_images_dirs:
            # load all .jpg images from dir
            curr_parent_embeds = []
            for img_path in parent_dir.glob('*.jpg'):
                curr_image_embeds = self.model.encode_images(Image.open(str(img_path)).convert('RGB'), is_pil=True)
                curr_parent_embeds.append(curr_image_embeds)
            embeds.append(torch.cat(curr_parent_embeds, dim=0))
        dataset = EvolutionDataset(
            embeds=embeds,
            placeholder_token=self.cfg.placeholder_token,
            learnable_property=self.cfg.learnable_property
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers
        )
        return train_dataloader

    def train(self):

        while self.train_step < self.cfg.steps:
            self.model.text_encoder.train()
            self.model.clip_model.train()
            for sample_idx, batch in enumerate(self.train_dataloader):
                print(f'For step #{self.train_step}')
                image_emb, txt_emb = self.generate_clip_emb(
                    prompt=batch["text"][0],
                    batch_size=self.cfg.train_batch_size,
                    prior_cf_scale=4,
                    prior_steps="5",
                    negative_prior_prompt="",
                )
                image_emb_norm = image_emb.norm(p=2, dim=-1, keepdim=True)
                image_emb_normed = image_emb / image_emb_norm
                all_sims = []
                for pos_embed in batch["positive_embeds"]:
                    pos_embed_norm = pos_embed.norm(p=2, dim=-1, keepdim=True)
                    pos_embed_normed = pos_embed / pos_embed_norm
                    curr_sim = (pos_embed_normed.detach() @ image_emb_normed.T).mean()
                    all_sims.append(curr_sim)
                print('Positive similarities:', [sim.item() for sim in all_sims])
                weighted_sims = sum([self.cfg.mix_weights[i] * sim for i, sim in enumerate(all_sims)])
                loss = 1 - weighted_sims
                print(f'loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # We don't need update all embeddings weights. Only new embeddings.
                with torch.no_grad():
                    index_no_updates_t2 = torch.arange(
                        self.model.clip_model.token_embedding.weight.shape[0]) != self.t2_place_token_id
                    self.model.clip_model.token_embedding.weight[index_no_updates_t2] = \
                        self.orig_t2_params[index_no_updates_t2]

                self.train_step += 1

                if self.train_step % self.cfg.log_embed_frequency == 0:
                    embed_save_path = self.cfg.output_dir / f"{self.train_step}_step_embeds.bin"
                    self.save_embeds(embed_save_path)

                if self.cfg.log_image_frequency > 0 and (self.train_step % self.cfg.log_image_frequency == 0):
                    self.save_images(save_dir=self.cfg.images_root,
                                     save_prefix=f'{self.train_step}_step_images')

        self.save_embeds(self.cfg.output_dir / "learned_embeds.bin")
        self.save_images(save_dir=self.cfg.images_root, save_prefix=f'images')
