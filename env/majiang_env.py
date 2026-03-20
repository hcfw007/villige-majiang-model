"""
Gym 风格的麻将环境。

observation_space: 每个玩家视角的观测向量
action_space: 离散动作集合

动作编码：
  0-33      : 打出第 i 种牌
  34        : 宣布杠（自摸杠/加杠，由环境判断杠哪张）
  35        : 碰
  36        : 从弃牌杠
  37        : 胡牌
  38        : 跳过（不碰/不杠/不胡）
"""

import numpy as np
from env.game import (
    GameState, new_game,
    action_draw, action_discard, action_kong_self,
    action_pong, action_kong_from_discard,
    action_declare_win, settle,
    check_win,
)
from env.tile import NUM_TILE_TYPES, tile_name
from env.hand import WinResult
from env.shanten import calc_shanten

NUM_ACTIONS = 39

# 动作 ID
ACT_DISCARD_BASE = 0    # 0-33: 打出牌
ACT_KONG_SELF    = 34
ACT_PONG         = 35
ACT_KONG_DISCARD = 36
ACT_WIN          = 37
ACT_PASS         = 38


class MajiangEnv:
    """
    三人麻将环境。

    每次 step() 由当前 active_player 执行一个动作。
    外部 RL 智能体轮流为三名玩家提供动作。
    """

    def __init__(self, dealer: int = 0):
        self.dealer = dealer
        self.state: GameState = None

    def reset(self):
        self.state = new_game(dealer=self.dealer)
        # 庄家先摸牌
        action_draw(self.state)
        obs = self._observe(self.state.current_player)
        return obs, self._info()

    def step(self, action: int):
        state = self.state
        seat  = state.current_player

        done    = False
        rewards = {i: 0.0 for i in range(3)}

        if state.phase == 'action':
            done = self._handle_action(seat, action)
        elif state.phase == 'respond':
            done = self._handle_respond(seat, action)

        if state.phase == 'end' or done:
            deltas = settle(state)
            for i in range(3):
                rewards[i] = float(deltas[i])
            done = True
            obs = self._observe(seat)
            return obs, rewards, done, self._info()

        # 推进到下一个需要行动的玩家
        if state.phase == 'draw':
            drawn = action_draw(state)
            if drawn is None:
                # 流局
                deltas = settle(state)
                rewards = {i: float(deltas[i]) for i in range(3)}
                return self._observe(state.current_player), rewards, True, self._info()

        obs = self._observe(state.current_player)
        return obs, rewards, False, self._info()

    def _handle_action(self, seat: int, action: int) -> bool:
        """处理 'action' 阶段（摸牌后行动）"""
        state = self.state
        p = state.players[seat]

        if action == ACT_WIN:
            success = action_declare_win(state, seat)
            if not success:
                # 非法胡牌：打出第一张牌
                action_discard(state, seat, p.hand[0])
            return state.phase == 'end'

        if action == ACT_KONG_SELF:
            counts, wilds = p.hand_counts(state.wild_idx)
            kong_tile = None
            if wilds >= 4 or (wilds >= 1 and any(
                m.meld_type == 'pong_wild' for m in p.melds
            )):
                kong_tile = state.wild_idx
            else:
                for i in range(NUM_TILE_TYPES):
                    if counts[i] >= 4:
                        kong_tile = i
                        break
                    if counts[i] >= 1 and any(
                        m.meld_type == 'pong' and m.tiles[0] == i for m in p.melds
                    ):
                        kong_tile = i
                        break
            if kong_tile is not None:
                action_kong_self(state, seat, kong_tile)
            else:
                action_discard(state, seat, p.hand[0])
            return state.phase == 'end'

        if 0 <= action <= 33:
            tile = action
            if tile not in p.hand:
                tile = p.hand[0]
            action_discard(state, seat, tile)
            return False

        # 未知动作：打第一张
        action_discard(state, seat, p.hand[0])
        return False

    def _handle_respond(self, seat: int, action: int) -> bool:
        """处理 'respond' 阶段（对他人打牌的响应）"""
        state = self.state

        if action == ACT_PONG:
            success = action_pong(state, seat)
            if success:
                return False  # 碰牌成功，phase 已变为 'action'
            # 碰牌失败，继续下一个
        elif action == ACT_KONG_DISCARD:
            success = action_kong_from_discard(state, seat)
            if success:
                return state.phase == 'end'
            # 杠牌失败，继续下一个

        # ACT_PASS 或操作失败：移到队列下一个
        self._advance_respond_queue()
        return False

    def _advance_respond_queue(self):
        """从 respond_queue 移除当前玩家，推进到下一个或进入 draw 阶段"""
        state = self.state
        if state.respond_queue:
            state.respond_queue.pop(0)
        if state.respond_queue:
            state.current_player = state.respond_queue[0]
        else:
            # 所有人都已响应，进入摸牌阶段
            next_player = (state.last_discard_player + 1) % 3
            state.current_player = next_player
            state.phase = 'draw'

    # ─────────────────────────────────────────────
    # 观测空间
    # ─────────────────────────────────────────────

    def _observe(self, seat: int) -> np.ndarray:
        """
        构建观测向量（从 seat 玩家视角）。

        向量结构（共 382 维）：
          [0:34]     自己手牌计数（含万能牌）
          [34:68]    自己公开面子计数
          [68:102]   左家公开面子计数
          [102:136]  右家公开面子计数
          [136:170]  自己弃牌计数
          [170:204]  左家弃牌计数
          [204:238]  右家弃牌计数
          [238:272]  场上所有已见牌计数（手牌+面子+弃牌）
          [272:306]  万能牌 one-hot
          [306]      自己是否庄家
          [307]      左家是否庄家
          [308]      右家是否庄家
          [309]      剩余牌数 / 136
          [310]      自己杠次数 / 4
          [311]      左家杠次数 / 4
          [312]      右家杠次数 / 4
          [313]      自己 max_wilds_held / 4
          [314]      自己当前向听数 / 8（归一化）
          [315]      自己当前杠分（归一化 /25）
          [316]      自己是否碰了万能牌
          [317]      上张弃牌是否是万能牌
          [318:352]  上张弃牌 one-hot（34维，0=无）
          [352:382]  自己手牌是否含各种牌（binary，30维：仅数牌1-9×3花色）
        长度：382
        """
        state = self.state
        p     = state.players[seat]
        left  = (seat + 1) % 3
        right = (seat + 2) % 3
        obs   = np.zeros(382, dtype=np.float32)

        # 手牌
        counts, wilds = p.hand_counts(state.wild_idx)
        for i, c in enumerate(counts):
            obs[i] = c
        obs[state.wild_idx] += wilds

        # 面子
        for m in p.melds:
            for t in m.tiles:
                obs[34 + t] += 1
        for m in state.players[left].melds:
            if m.meld_type == 'kong' and not m.from_discard:
                continue  # 暗杠：对其他玩家不可见
            for t in m.tiles:
                obs[68 + t] += 1
        for m in state.players[right].melds:
            if m.meld_type == 'kong' and not m.from_discard:
                continue  # 暗杠：对其他玩家不可见
            for t in m.tiles:
                obs[102 + t] += 1

        # 弃牌历史
        for t in state.discards[seat]:
            obs[136 + t] += 1
        for t in state.discards[left]:
            obs[170 + t] += 1
        for t in state.discards[right]:
            obs[204 + t] += 1

        # 已见牌总计（帮助推断剩余牌）
        for i in range(34):
            obs[238 + i] = obs[i] + obs[34+i] + obs[68+i] + obs[102+i] \
                         + obs[136+i] + obs[170+i] + obs[204+i]

        # 万能牌
        obs[272 + state.wild_idx] = 1.0

        # 庄家标记
        obs[306] = 1.0 if p.is_dealer else 0.0
        obs[307] = 1.0 if state.players[left].is_dealer else 0.0
        obs[308] = 1.0 if state.players[right].is_dealer else 0.0

        # 剩余牌数、杠次数
        obs[309] = len(state.wall) / 136.0
        obs[310] = state.kong_count[seat] / 4.0
        obs[311] = state.kong_count[left] / 4.0
        obs[312] = state.kong_count[right] / 4.0
        obs[313] = p.max_wilds_held / 4.0

        # 向听数
        n_melds_fixed = sum(1 for m in p.melds if m.meld_type in ('pong', 'kong', 'pong_wild'))
        shanten = calc_shanten(counts, wilds, n_melds_fixed)
        obs[314] = max(0, shanten) / 8.0

        # 杠分
        from env.scorer import calc_kong_score
        obs[315] = calc_kong_score(state.kong_count[seat]) / 25.0

        # 碰了万能牌
        obs[316] = 1.0 if any(m.meld_type == 'pong_wild' for m in p.melds) else 0.0

        # 上张弃牌
        if state.last_discard is not None:
            obs[317] = 1.0 if state.last_discard == state.wild_idx else 0.0
            obs[318 + state.last_discard] = 1.0

        # 手牌是否含各数牌（binary，用于快速感知牌型）
        for i in range(27):  # 万筒条各9张
            obs[352 + i] = 1.0 if counts[i] > 0 else 0.0

        return obs

    def legal_actions(self, seat: int) -> list[int]:
        """返回当前玩家的合法动作列表"""
        state = self.state
        p     = state.players[seat]
        acts  = []

        if state.phase == 'action':
            counts, wilds = p.hand_counts(state.wild_idx)

            # 可以打出手中任意牌
            for t in set(p.hand):
                acts.append(t)  # 0-33

            # 可以胡牌
            if check_win(state, seat) is not None:
                acts.append(ACT_WIN)

            # 可以杠
            for i in range(NUM_TILE_TYPES):
                if counts[i] >= 4:
                    acts.append(ACT_KONG_SELF)
                    break
                if counts[i] >= 1 and any(
                    m.meld_type == 'pong' and m.tiles[0] == i for m in p.melds
                ):
                    acts.append(ACT_KONG_SELF)
                    break
            if wilds >= 4 or (wilds >= 1 and any(
                m.meld_type == 'pong_wild' for m in p.melds
            )):
                if ACT_KONG_SELF not in acts:
                    acts.append(ACT_KONG_SELF)

        elif state.phase == 'respond':
            acts.append(ACT_PASS)
            tile = state.last_discard
            if tile is not None:
                counts, wilds = p.hand_counts(state.wild_idx)
                cnt = wilds if tile == state.wild_idx else counts[tile]
                if cnt >= 2:
                    acts.append(ACT_PONG)
                if cnt >= 3:
                    acts.append(ACT_KONG_DISCARD)

        return acts

    def _info(self) -> dict:
        state = self.state
        return {
            'wild_idx':       state.wild_idx,
            'wild_name':      tile_name(state.wild_idx),
            'phase':          state.phase,
            'current_player': state.current_player,
            'winner':         state.winner,
            'win_result':     state.win_result,
            'wall_remaining': len(state.wall),
        }
