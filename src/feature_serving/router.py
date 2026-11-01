import random
from dataclasses import dataclass

import numpy as np


@dataclass
class ThompsonRouter:
    arms: int
    rng: random.Random
    successes: list[int]
    failures: list[int]

    @classmethod
    def create(cls, arms: int, seed: int) -> "ThompsonRouter":
        return cls(arms, random.Random(seed), [0] * arms, [0] * arms)

    def choose(self) -> int:
        draws = [self.rng.betavariate(1 + s, 1 + f) for s, f in zip(self.successes, self.failures)]
        return int(np.argmax(draws))

    def update(self, arm: int, reward: int) -> None:
        self.successes[arm] += reward
        self.failures[arm] += 1 - reward


def simulate(rewards: np.ndarray, requests: int, seed: int, adaptive: bool) -> float:
    rng = random.Random(seed)
    arms = rewards.shape[1]
    router = ThompsonRouter.create(arms, seed)
    best_mean = rewards.mean(axis=0).max()
    regret = 0.0
    for _ in range(requests):
        row = rewards[rng.randrange(len(rewards))]
        arm = router.choose() if adaptive else rng.randrange(arms)
        router.update(arm, int(row[arm]))
        regret += best_mean - row[arm]
    return regret
