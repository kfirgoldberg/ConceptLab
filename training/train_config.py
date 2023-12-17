from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from training.templates import LearnableProperties


@dataclass
class TrainConfig:
    # A token to use as a placeholder for the concept.
    placeholder_token: str
    # A token to use as initializer word.
    initializer_token: str
    # Path to pretrained model WITHOUT 2_1 folder
    cache_root: Path = Path('/tmp/kandinsky2')
    # Defines which prompts to use
    learnable_property: LearnableProperties = LearnableProperties.object
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: Path = Path('concept-lab')
    # GPU device
    device: str = 'cuda:0'
    # The resolution for input images, all the images will be resized to this size
    img_size: int = 512
    # Batch size (per device) for the training dataloader
    train_batch_size: int = 1
    # Number of steps to train for
    steps: int = 1000
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
    log_image_frequency: int = -1
    # How often save embeddings
    log_embed_frequency: int = 2500
    # Positive class, if not set use the same as init token
    positive_classes: List[str] = field(default_factory=list)
    # Positive weights, if not set use 1/num_classes
    positive_weights: Optional[List[float]] = None
    # Negative prompts
    negative_classes: List[str] = field(default_factory=list)
    # Whether to use live negatives
    live_negatives: bool = False
    # A seed for reproducible training
    seed: Optional[int] = None
    # Comma-separated seeds for reproducible inference.
    inference_seeds: List[int] = field(default_factory=lambda: [42, 500])
    # Whether to use gradual negatives
    gradual_negatives: bool = False
    # Max cosine similarity to optimize for
    max_cosine_thr: float = 0.28
    # Min cosine similarity to optimize for
    min_cosine_thr: float = 0.2
    # Factor of loss between positive and negative constraints
    pos_to_neg_loss_factor: float = 1.0
    # Whether to use early stopping for live negatives
    negatives_early_stop: bool = False
    # Whether to use CLIP-ConceptLab and optimize only in CLIP text space
    optimize_in_text_space: bool = False

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if len(self.positive_classes) == 0:
            print('Set positive class to initializer token')
            self.positive_classes = [self.initializer_token]

        if self.positive_weights is None:
            self.positive_weights = [1 / len(self.positive_classes) for _ in self.positive_classes]
        if len(self.positive_classes) != len(self.positive_weights):
            raise ValueError('num positive_weights != num positive_classes')
        self.images_root = self.output_dir / "images"
        self.images_root.mkdir(exist_ok=True, parents=True)

