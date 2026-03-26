"""
单智能体 Gym 包装器。

RL 智能体始终控制 seat=0，其余两家由 opponent_policy 控制。
支持 sb3-contrib MaskablePPO 所需的 action_masks() 接口。
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.majiang_env import (
    MajiangEnv, NUM_ACTIONS,
    ACT_DISCARD_BASE, ACT_KONG_SELF, ACT_PONG,
    ACT_KONG_DISCARD, ACT_WIN, ACT_PASS,
)
from env.game import new_game, action_draw, settle, check_win
from env.tile import NUM_TILE_TYPES
from env.shanten import calc_shanten, _shanten_qidui
from env.scorer import calc_kong_score
from env.tile import tile_suit, SUIT_ZI

OBS_DIM = 382

# ── 中间奖励系数 ──────────────────────────────────────
SHANTEN_REWARD_SCALE = 0.4   # 每减少1向听的奖励
TENPAI_BONUS         = 3.0   # 有效听牌（底分≥门槛）额外奖励
# 无效路径（当前底分潜力不足）时的奖励缩小倍数
INVALID_PATH_SCALE   = 0.1
WIN_BONUS_BASE       = 10.0  # 终局胜利基础奖励
LIUJU_PENALTY        = -3.0  # 流局惩罚（降低保守流局倾向）
KONG_REWARD          = 2.0   # 每次杠牌即时奖励（鼓励积极杠牌）


class SingleAgentMajiangEnv(gym.Env):
    """
    3 人麻将，固定控制 seat=0，对手使用 opponent_policy。

    opponent_policy: callable(obs, legal_actions) -> action
                     默认为随机策略。
    """

    metadata = {'render_modes': []}

    def __init__(self, opponent_policy=None, dealer: int = -1):
        super().__init__()
        self.opponent_policy = opponent_policy or _random_policy
        self.dealer = dealer  # -1 表示每局随机

        self.observation_space = spaces.Box(
            low=0.0, high=136.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._env = MajiangEnv(dealer=dealer)
        self._last_obs = None
        self._prev_shanten = 8  # 上一步的向听数，用于计算中间奖励

    # ─────────────────────────────────────────────
    # Gym 接口
    # ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        dealer = random.randint(0, 2) if self.dealer == -1 else self.dealer
        self._env = MajiangEnv(dealer=dealer)
        obs, info = self._env.reset()
        # 对局内策略行为计数器
        self._act_pong_taken   = 0
        self._act_pong_skipped = 0
        self._act_kong_taken   = 0
        self._act_win_declared = 0
        self._act_win_skipped  = 0
        # 如果第一个行动玩家不是 seat=0，先让对手行动
        obs = self._run_opponents_until_agent(obs, info)
        self._prev_shanten = self._get_shanten()
        self._last_obs = obs
        return obs, {}

    def step(self, action: int):
        env = self._env
        state = env.state

        # 记录策略行为
        self._record_action(action, env.legal_actions(0))

        # 执行 seat=0 的动作
        obs, rewards, done, info = env.step(action)

        if done:
            if env.state.winner == 0:
                terminal_r = WIN_BONUS_BASE + env.state.win_result.base_score
            else:
                terminal_r = float(rewards[0])
                if env.state.winner is None:
                    terminal_r += self._liuju_bonus()
            return obs, terminal_r, True, False, self._episode_info()

        # 让对手行动，直到轮到 seat=0 或游戏结束
        obs = self._run_opponents_until_agent(obs, info)
        done = env.state.phase == 'end'
        if done:
            deltas = settle(env.state)
            if env.state.winner == 0:
                terminal_r = WIN_BONUS_BASE + env.state.win_result.base_score
            else:
                terminal_r = float(deltas[0])
                if env.state.winner is None:
                    terminal_r += self._liuju_bonus()
            return obs, terminal_r, True, False, self._episode_info()

        # 计算中间奖励（基于向听数变化）
        new_shanten = self._get_shanten()
        step_reward = self._shanten_reward(self._prev_shanten, new_shanten)
        self._prev_shanten = new_shanten

        # 杠牌即时奖励
        if action in (ACT_KONG_SELF, ACT_KONG_DISCARD):
            step_reward += KONG_REWARD

        self._last_obs = obs
        return obs, step_reward, False, False, {}

    def _get_shanten(self) -> int:
        """计算 seat=0 当前的向听数"""
        state = self._env.state
        p = state.players[0]
        counts, wilds = p.hand_counts(state.wild_idx)
        n_melds_fixed = sum(
            1 for m in p.melds if m.meld_type in ('pong', 'kong', 'pong_wild', 'kong_wild')
        )
        return calc_shanten(counts, wilds, n_melds_fixed)

    def _liuju_bonus(self) -> float:
        """流局惩罚（固定负值，消除保守流局倾向）"""
        return LIUJU_PENALTY

    def _estimate_score_potential(self) -> int:
        """
        估算当前手牌路径的底分潜力，用于过滤无意义的向听奖励。

        返回估算的最低可达底分：
          - 已有杠 → 杠分（≥10即可胡）
          - 追七对路线 → 10分
          - 追清一色路线（≥80%非字牌同花色） → 10分
          - 其他无杠手牌 → 0（凑完也不能胡）
        """
        state = self._env.state
        p = state.players[0]
        counts, wilds = p.hand_counts(state.wild_idx)

        # 1. 已有杠的分数
        kong_count = sum(1 for m in p.melds if 'kong' in m.meld_type)
        score = calc_kong_score(kong_count)
        if score >= 10:
            return score

        # 2. 七对路线（向听≤3时认为在追）
        if _shanten_qidui(counts, wilds) <= 3:
            return 10

        # 3. 清一色路线（非字牌中≥80%集中在同一花色）
        suit_tile_counts = [0, 0, 0]
        for i, c in enumerate(counts):
            s = tile_suit(i)
            if s != SUIT_ZI and c > 0:
                suit_tile_counts[s] += c
        total_suit = sum(suit_tile_counts)
        if total_suit > 0 and max(suit_tile_counts) / total_suit >= 0.8:
            return 10

        return 0  # 无法达到胡牌门槛

    def _shanten_reward(self, prev: int, new: int) -> float:
        """
        根据向听数变化返回中间奖励。
        只有当前手牌路径有实际胡牌潜力（底分可达门槛）时才给予全额奖励。
        """
        if prev <= 0 or new < 0:
            return 0.0
        improvement = prev - new
        if improvement <= 0:
            return 0.0

        potential = self._estimate_score_potential()
        # 万能牌≥2时门槛为20，否则为10
        state = self._env.state
        p = state.players[0]
        _, wilds = p.hand_counts(state.wild_idx)
        wild_total = wilds + sum(
            4 if 'kong_wild' in m.meld_type else
            3 if m.meld_type == 'pong_wild' else 0
            for m in p.melds
        )
        threshold = 20 if wild_total >= 2 else 10
        # shanten≤1（接近或已听牌）时也视为有效路径，鼓励完成手牌
        valid = potential >= threshold or new <= 1

        scale = 1.0 if valid else INVALID_PATH_SCALE
        reward = improvement * SHANTEN_REWARD_SCALE * scale

        # 有效听牌（向听=0且底分足够）给予额外奖励
        if new == 0 and valid:
            reward += TENPAI_BONUS
        return reward

    def _record_action(self, action: int, legal: list):
        if action == ACT_WIN:
            self._act_win_declared += 1
        elif action == ACT_PONG:
            self._act_pong_taken += 1
        elif action in (ACT_KONG_SELF, ACT_KONG_DISCARD):
            self._act_kong_taken += 1
        elif action == ACT_PASS and ACT_PONG in legal:
            self._act_pong_skipped += 1
        if ACT_WIN in legal and action != ACT_WIN:
            self._act_win_skipped += 1

    def _episode_info(self) -> dict:
        """对局结束时的统计信息，供 StatsCallback 使用"""
        state = self._env.state
        return {
            "winner":       state.winner,
            "win_result":   state.win_result,
            "is_dealer":    state.players[0].is_dealer,
            "pong_taken":   self._act_pong_taken,
            "pong_skipped": self._act_pong_skipped,
            "kong_taken":   self._act_kong_taken,
            "win_declared": self._act_win_declared,
            "win_skipped":  self._act_win_skipped,
        }

    def action_masks(self) -> np.ndarray:
        """MaskablePPO 所需：返回合法动作 bool 掩码"""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for a in self._env.legal_actions(0):
            mask[a] = True
        # 确保至少有一个合法动作
        if not mask.any():
            mask[ACT_PASS] = True
        return mask

    # ─────────────────────────────────────────────
    # 内部：代替对手行动
    # ─────────────────────────────────────────────

    def _run_opponents_until_agent(self, obs, info) -> np.ndarray:
        """持续让对手行动，直到当前玩家为 seat=0 或游戏结束"""
        env = self._env
        while env.state.phase not in ('end',) and env.state.current_player != 0:
            seat = env.state.current_player
            legal = env.legal_actions(seat)
            opp_obs = env._observe(seat)
            action = self.opponent_policy(opp_obs, legal)
            obs, rewards, done, info = env.step(action)
            if done:
                break
        return env._observe(0)


# ─────────────────────────────────────────────
# 内置策略
# ─────────────────────────────────────────────

def _random_policy(obs: np.ndarray, legal_actions: list) -> int:
    import random
    return random.choice(legal_actions) if legal_actions else ACT_PASS


def make_model_policy(model):
    """将训练好的 SB3 模型包装成 policy 函数"""
    def policy(obs: np.ndarray, legal_actions: list) -> int:
        action, _ = model.predict(obs, deterministic=True)
        if int(action) not in legal_actions:
            import random
            return random.choice(legal_actions)
        return int(action)
    return policy
