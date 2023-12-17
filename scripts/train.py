import pyrallis

from training.coach import Coach
from training.train_config import TrainConfig


@pyrallis.wrap()
def main(cfg: TrainConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == "__main__":
    main()
