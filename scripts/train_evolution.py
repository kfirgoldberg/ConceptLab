import pyrallis
import sys


from training.coach_evolution import TrainEvolutionConfig, CoachEvolution


@pyrallis.wrap()
def main(cfg: TrainEvolutionConfig):
    coach = CoachEvolution(cfg)
    coach.train()


if __name__ == "__main__":
    main()
