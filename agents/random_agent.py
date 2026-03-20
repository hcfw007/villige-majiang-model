"""随机智能体，作为 RL 训练的基准对手"""

import random
from env.majiang_env import MajiangEnv


class RandomAgent:
    def __init__(self, seat: int):
        self.seat = seat

    def act(self, obs, env: MajiangEnv) -> int:
        legal = env.legal_actions(self.seat)
        return random.choice(legal) if legal else 38  # 38=PASS
