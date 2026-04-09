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
from env.tile import tile_suit, SUIT_ZI, tile_name

OBS_DIM = 382

# ── 中间奖励系数 ──────────────────────────────────────
SHANTEN_REWARD_SCALE  = 0.4   # 每减少1向听的奖励
TENPAI_BONUS          = 3.0   # 有效听牌（底分≥门槛）额外奖励
# 无效路径（当前底分潜力不足）时的奖励缩小倍数
INVALID_PATH_SCALE    = 0.1
# 有杠潜力但尚未杠到时的奖励缩小倍数
KONG_POTENTIAL_SCALE  = 0.5
KONG_REWARD           = 2.0   # 每次杠牌即时奖励（鼓励积极杠牌）
JIAKANG_TENPAI_BONUS  = 4.0   # 听牌状态下加杠（碰升杠）的额外奖励，激励杠上开花
LIUJU_PENALTY         = 3.0   # 流局额外惩罚（叠加在结算分之上）
LIUJU_TENPAI_DISCOUNT = 0.3   # 流局时已听牌，惩罚缩小为原来的此比例
PONG_REWARD           = 0.3   # 碰牌即时奖励（仅当第4张未见时）



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
        self._replay_log: list = []

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
        # 牌谱
        self._replay_log = [{"e": "init",
                              "wild": tile_name(self._env.state.wild_idx),
                              "dealer": dealer}]
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

        # 记录牌谱：记录本次轮到seat0时的摸牌及动作
        self._log_agent_turn(action, state)

        # 记录策略行为
        self._record_action(action, env.legal_actions(0))

        # 行动前保存需要的信息（state 对象会被 env.step 原地修改）
        shanten_before = self._prev_shanten
        is_jiakang = (action == ACT_KONG_SELF) and self._is_jiakang(state)

        # 执行 seat=0 的动作
        obs, rewards, done, info = env.step(action)

        if done:
            deltas = settle(env.state)
            terminal_r = float(deltas[0])
            if env.state.winner is None:
                terminal_r -= self._liuju_penalty()
            return obs, terminal_r, True, False, self._episode_info(deltas)

        # 立即计算行动后向听数（行动前 vs 行动后，直接衡量本次动作的影响）
        post_action_shanten = self._get_shanten()
        step_reward = self._shanten_reward(shanten_before, post_action_shanten)

        # 碰牌奖励：仅当第4张尚未可见时
        if action == ACT_PONG:
            pong_r = self._pong_reward()
            if pong_r > 0:
                step_reward += pong_r

        # 让对手行动，直到轮到 seat=0 或游戏结束
        obs = self._run_opponents_until_agent(obs, info)
        done = env.state.phase == 'end'
        if done:
            deltas = settle(env.state)
            terminal_r = float(deltas[0])
            if env.state.winner is None:
                terminal_r -= self._liuju_penalty()
            return obs, terminal_r, True, False, self._episode_info(deltas)

        # 更新 prev_shanten 为下一步"行动前"的状态（已摸牌）
        self._prev_shanten = self._get_shanten()

        # 杠牌即时奖励
        if action in (ACT_KONG_SELF, ACT_KONG_DISCARD):
            step_reward += KONG_REWARD
            # 听牌状态下加杠（碰升杠）→ 额外奖励，激励杠上开花策略
            if is_jiakang and shanten_before <= 0:
                step_reward += JIAKANG_TENPAI_BONUS

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

    def _is_jiakang(self, state) -> bool:
        """
        检测当前 ACT_KONG_SELF 是否为加杠（将已有碰牌升为杠）。
        加杠 = 手中持有与某个碰牌相同的第4张，可以升级为杠。
        """
        p = state.players[0]
        counts, wilds = p.hand_counts(state.wild_idx)
        for m in p.melds:
            if m.meld_type == 'pong':
                if counts[m.tiles[0]] >= 1:
                    return True
            elif m.meld_type == 'pong_wild':
                if wilds >= 1:
                    return True
        return False


    def _pong_reward(self) -> float:
        """
        碰牌即时奖励。仅当被碰牌的第4张尚未在场面上可见时才给奖励，
        此时碰→加杠路线仍有可能，值得鼓励。
        """
        state = self._env.state
        p = state.players[0]
        # 找到刚碰的面子（最后一个 pong/pong_wild）
        if not p.melds:
            return 0.0
        last_meld = p.melds[-1]
        if last_meld.meld_type not in ('pong', 'pong_wild'):
            return 0.0
        tile_idx = last_meld.tiles[0]
        # 统计该牌在场面上可见的数量：碰的3张 + 弃牌堆 + 其他人面子
        visible = 3  # 碰本身
        for seat in range(3):
            visible += state.discards[seat].count(tile_idx)
        for seat in range(3):
            for m in state.players[seat].melds:
                if m is not last_meld:
                    visible += m.tiles.count(tile_idx)
        # 第4张未见 → 奖励
        if visible < 4:
            return PONG_REWARD
        return 0.0

    def _liuju_penalty(self) -> float:
        """流局惩罚，听牌时减免"""
        shanten = self._get_shanten()
        if shanten <= 0:
            potential = self._estimate_score_potential()
            state = self._env.state
            p = state.players[0]
            _, wilds = p.hand_counts(state.wild_idx)
            wild_total = wilds + sum(
                4 if 'kong_wild' in m.meld_type else
                3 if m.meld_type == 'pong_wild' else 0
                for m in p.melds
            )
            threshold = 20 if wild_total >= 2 else 10
            if potential >= threshold:
                return LIUJU_PENALTY * LIUJU_TENPAI_DISCOUNT
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

        # 4. 一条龙路线（某花色1-9中，缺的点数 ≤ 手中万能牌数）
        for s in range(3):
            missing = sum(1 for n in range(9) if counts[s * 9 + n] == 0)
            if missing <= wilds:
                return 10

        # 5. 有杠潜力：手里有三张同牌，或碰了一对且第四张未全部可见
        for i, c in enumerate(counts):
            if c >= 3:
                return 5
        for m in p.melds:
            if m.meld_type in ('pong', 'pong_wild'):
                tile_idx = m.tiles[0]
                seen = 3  # 碰牌本身占3张
                for seat in range(3):
                    seen += state.discards[seat].count(tile_idx)
                for player in state.players:
                    for other_m in player.melds:
                        if other_m is not m:
                            seen += other_m.tiles.count(tile_idx)
                if seen < 4:
                    return 5

        return 0  # 无法达到胡牌门槛

    def _shanten_reward(self, prev: int, new: int) -> float:
        """
        根据向听数变化返回中间奖励（可为负）。

        比较的是行动前（14张）vs 行动后（13张）的向听数：
          - 向听数减少 → 正奖励（打得好）
          - 向听数不变 → 0
          - 向听数增加 → 负奖励（打坏了，如拆散对子）

        路径有效性缩放仍然生效：无效路径奖惩均按比例缩小。
        """
        if new < 0:
            return 0.0  # 已胡牌，由终局奖励处理
        improvement = prev - new
        if improvement == 0:
            return 0.0

        potential = self._estimate_score_potential()
        state = self._env.state
        p = state.players[0]
        _, wilds = p.hand_counts(state.wild_idx)
        wild_total = wilds + sum(
            4 if 'kong_wild' in m.meld_type else
            3 if m.meld_type == 'pong_wild' else 0
            for m in p.melds
        )
        threshold = 20 if wild_total >= 2 else 10
        # shanten≤1 视为有效路径（接近听牌时惩罚力度不打折）
        valid = potential >= threshold or new <= 1

        if valid:
            scale = 1.0
        elif potential > 0:
            scale = KONG_POTENTIAL_SCALE
        else:
            scale = INVALID_PATH_SCALE

        reward = improvement * SHANTEN_REWARD_SCALE * scale

        # 达到有效听牌时给予额外奖励（仅向好方向变化时）
        if new == 0 and improvement > 0 and valid:
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

    def _episode_info(self, deltas: dict = None) -> dict:
        """对局结束时的统计信息，供 StatsCallback 使用"""
        state = self._env.state
        # 收尾牌谱：记录终局事件
        if state.winner is None:
            self._replay_log.append({"e": "end", "result": "流局", "deltas": deltas})
        elif state.winner == 0:
            wr = state.win_result
            types_zh = "+".join(wr.win_types) if wr else "?"
            self._replay_log.append({"e": "end", "result": "seat0胡",
                                     "types": types_zh,
                                     "score": int(wr.base_score) if wr else 0,
                                     "deltas": deltas})
        else:
            self._replay_log.append({"e": "end", "result": f"seat{state.winner}胡",
                                     "deltas": deltas})
        return {
            "winner":       state.winner,
            "win_result":   state.win_result,
            "is_dealer":    state.players[0].is_dealer,
            "pong_taken":   self._act_pong_taken,
            "pong_skipped": self._act_pong_skipped,
            "kong_taken":   self._act_kong_taken,
            "win_declared": self._act_win_declared,
            "win_skipped":  self._act_win_skipped,
            "replay":       self._replay_log,
        }

    def _log_agent_turn(self, action: int, state):
        """记录 seat0 本轮的摸牌（若有）和动作"""
        p = state.players[0]
        hand = " ".join(sorted(tile_name(t) for t in p.hand))
        if state.phase == 'action':
            drawn = tile_name(state.last_drawn) if state.last_drawn is not None else "?"
            self._replay_log.append({"e": "draw", "tile": drawn,
                                     "sht": self._prev_shanten, "hand": hand})
            act_str = self._act_str(action, state)
        else:  # respond
            act_str = self._act_str(action, state)
        self._replay_log.append({"e": "act", "a": act_str})

    def _act_str(self, action: int, state) -> str:
        if action < 34:
            return f"打{tile_name(action)}"
        elif action == ACT_KONG_SELF:
            return "暗杠"
        elif action == ACT_PONG:
            t = state.last_discard
            return f"碰{tile_name(t)}" if t is not None else "碰"
        elif action == ACT_KONG_DISCARD:
            t = state.last_discard
            return f"杠明{tile_name(t)}" if t is not None else "杠明"
        elif action == ACT_WIN:
            return "胡牌"
        elif action == ACT_PASS:
            t = state.last_discard
            return f"过({tile_name(t)})" if t is not None else "过"
        return f"act{action}"

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
            # 记录对手关键动作（碰/杠/胡）
            if action in (ACT_PONG, ACT_KONG_DISCARD):
                tile = env.state.last_discard
                self._replay_log.append({"e": "opp", "seat": seat,
                                         "a": ("碰" if action == ACT_PONG else "杠明") + tile_name(tile)})
            elif action == ACT_KONG_SELF:
                self._replay_log.append({"e": "opp", "seat": seat, "a": "暗杠"})
            elif action == ACT_WIN:
                self._replay_log.append({"e": "opp", "seat": seat, "a": "胡牌"})
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
