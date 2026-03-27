"""
游戏状态管理。

流程：
  1. 初始化牌墙，确定万能牌
  2. 发牌（每人13张）
  3. 轮流：
     a. 摸牌
     b. 检测胡牌（自摸/杠上开花）
     c. 动作：打牌 / 宣布杠
     d. 其他玩家响应：碰 / 杠万能牌（杠即胡）
  4. 流局 or 胡牌结算
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from env.tile import NUM_TILE_TYPES, TILES_PER_TYPE, tile_name
from env.hand import evaluate_hand, WinResult
from env.scorer import calc_kong_score, calc_payments


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class Meld:
    meld_type: str   # 'pong' | 'kong' | 'pong_wild' | 'kong_wild'
    tiles: list[int]
    from_discard: bool = False  # 是否由别人打出触发


@dataclass
class PlayerState:
    seat: int
    hand: list[int]            # 手牌（含万能牌，以 wild_idx 标识）
    melds: list[Meld] = field(default_factory=list)
    score: int = 0
    is_dealer: bool = False

    # 万能牌追踪（用于胡牌门槛）
    max_wilds_held: int = 0    # 本局持有过的最多万能牌数（含已打出）

    def hand_counts(self, wild_idx: int) -> tuple[list[int], int]:
        """返回 (counts, wilds)，从手牌中分离万能牌"""
        counts = [0] * NUM_TILE_TYPES
        wilds = 0
        for t in self.hand:
            if t == wild_idx:
                wilds += 1
            else:
                counts[t] += 1
        return counts, wilds

    def update_max_wilds(self, wild_idx: int):
        _, wilds = self.hand_counts(wild_idx)
        if wilds > self.max_wilds_held:
            self.max_wilds_held = wilds


@dataclass
class GameState:
    wall: list[int]            # 牌墙（剩余）
    players: list[PlayerState]
    wild_idx: int              # 本局万能牌索引
    current_player: int        # 当前行动玩家
    dealer: int                # 庄家座位
    last_discard: Optional[int] = None          # 上一张打出的牌
    last_discard_player: Optional[int] = None   # 打出者
    last_drawn: Optional[int] = None            # 刚摸的牌
    kong_count: list[int] = field(default_factory=lambda: [0, 0, 0])  # 每人杠次数
    lingshang_trigger: int = -1   # 杠上开花触发者（-1=无）
    phase: str = 'draw'           # 'draw' | 'action' | 'respond' | 'end'
    winner: Optional[int] = None
    win_result: Optional[WinResult] = None
    is_lingshang: bool = False
    respond_queue: list[int] = field(default_factory=list)  # 还需响应的玩家队列
    lingshang_pending: bool = False  # 杠后待胡标记
    dead_count: int = 14  # 牌墙末尾不可摸的死牌数量（含万能牌指示牌）
    discards: list = field(default_factory=lambda: [[], [], []])  # 每人弃牌历史


# ─────────────────────────────────────────────
# 游戏工厂
# ─────────────────────────────────────────────

def build_wall() -> list[int]:
    """构建并洗牌，返回136张牌的索引列表"""
    wall = [t for t in range(NUM_TILE_TYPES) for _ in range(TILES_PER_TYPE)]
    random.shuffle(wall)
    return wall


def determine_wild(wall: list[int]) -> int:
    """
    掷骰子（1-6），从牌墙末尾倒数定位，其下一张（更靠末尾方向）为万能牌。
    简化：从墙尾取骰子位置的牌，其后一张为万能牌定位牌。
    """
    dice = random.randint(1, 6)
    # 从牌墙末尾数 dice 张，该位置的牌决定万能牌
    indicator_pos = -(dice)          # 指示牌位置
    wild_pos      = -(dice - 1)      # 万能牌位置（指示牌的"下一张"，即更靠末尾）
    if wild_pos == 0:
        # dice=1 时 wild_pos=0，取最后一张
        wild_tile = wall[-1]
    else:
        wild_tile = wall[wild_pos]
    return wild_tile


def new_game(dealer: int = 0) -> GameState:
    """初始化一局游戏"""
    wall = build_wall()
    wild_idx = determine_wild(wall)

    players = []
    for seat in range(3):
        # 发13张牌（从墙头取）
        hand = wall[:13]
        wall = wall[13:]
        p = PlayerState(seat=seat, hand=hand, is_dealer=(seat == dealer))
        p.update_max_wilds(wild_idx)
        players.append(p)

    return GameState(
        wall=wall,
        players=players,
        wild_idx=wild_idx,
        current_player=dealer,
        dealer=dealer,
        kong_count=[0, 0, 0],
        dead_count=14,
    )


# ─────────────────────────────────────────────
# 胡牌检测
# ─────────────────────────────────────────────

def _open_melds_for_hand(player: PlayerState) -> list:
    return [(m.meld_type, m.tiles) for m in player.melds]


def check_win(state: GameState, seat: int) -> Optional[WinResult]:
    p = state.players[seat]
    counts, wilds = p.hand_counts(state.wild_idx)
    open_melds = _open_melds_for_hand(p)
    kong_score = calc_kong_score(state.kong_count[seat])

    # 集齐4张万能牌（特殊：手牌摸到第4张）
    wild_in_open = sum(
        3 if m.meld_type == 'pong_wild' else (4 if m.meld_type == 'kong_wild' else 0)
        for m in p.melds
    )
    total_wilds_in_hand = wilds + wild_in_open
    if total_wilds_in_hand >= 4:
        total_score = 40 + kong_score
        threshold = 20 if p.max_wilds_held >= 2 else 10
        if total_score >= threshold:
            return WinResult(win_types=['four_wilds'], base_score=total_score)

    result = evaluate_hand(
        counts=counts,
        wilds=wilds,
        open_melds=open_melds,
        kong_score=kong_score,
        total_wild_count=p.max_wilds_held,
        last_drawn=state.last_drawn,
    )
    return result


# ─────────────────────────────────────────────
# 动作执行
# ─────────────────────────────────────────────

def _dead_count_for_kongs(total_kongs: int) -> int:
    """根据场上总杠数计算死牌区大小"""
    if total_kongs == 0:
        return 14
    elif total_kongs == 1:
        return 10
    else:
        return 0


def action_draw(state: GameState) -> Optional[int]:
    """从牌墙摸牌，返回摸到的牌，或 None（流局）"""
    # 可摸牌数 = 总牌数 - 死牌区
    if len(state.wall) <= state.dead_count:
        state.phase = 'end'
        return None
    tile = state.wall.pop(0)
    state.players[state.current_player].hand.append(tile)
    state.players[state.current_player].update_max_wilds(state.wild_idx)
    state.last_drawn = tile
    state.phase = 'action'
    return tile


def action_discard(state: GameState, seat: int, tile_idx: int):
    """打出一张牌"""
    p = state.players[seat]
    assert tile_idx in p.hand, f"玩家{seat}手中没有牌{tile_name(tile_idx)}"
    p.hand.remove(tile_idx)
    state.last_discard = tile_idx
    state.last_discard_player = seat
    state.lingshang_trigger = -1
    state.lingshang_pending = False
    state.discards[seat].append(tile_idx)
    # 按顺序让其他两名玩家依次响应
    state.respond_queue = [(seat + 1) % 3, (seat + 2) % 3]
    state.current_player = state.respond_queue[0]
    state.phase = 'respond'


def action_kong_self(state: GameState, seat: int, tile_idx: int) -> bool:
    """
    自摸杠（手中有4张，或碰后补杠）。
    杠万能牌直接胡牌。
    返回 True 表示杠成功（若是万能牌则同时标记胡牌）。
    """
    p = state.players[seat]
    counts, wilds = p.hand_counts(state.wild_idx)

    if tile_idx == state.wild_idx:
        # 杠万能牌 → 直接胡
        p.melds.append(Meld('kong_wild', [tile_idx] * 4))
        for _ in range(min(4, p.hand.count(tile_idx))):
            p.hand.remove(tile_idx)
        kong_score = calc_kong_score(state.kong_count[seat])
        threshold = 20 if p.max_wilds_held >= 2 else 10
        result = WinResult(win_types=['four_wilds'], base_score=40 + kong_score)
        _do_win(state, seat, result, lingshang=False)
        return True

    # 普通杠：手中有4张
    if counts[tile_idx] >= 4:
        for _ in range(4):
            p.hand.remove(tile_idx)
        p.melds.append(Meld('kong', [tile_idx] * 4))
        state.kong_count[seat] += 1
        # 摸补牌（杠上花机会）
        _draw_lingshang(state, seat)
        return True

    # 碰后补杠（加杠）
    for m in p.melds:
        if m.meld_type == 'pong' and m.tiles[0] == tile_idx:
            p.hand.remove(tile_idx)
            m.meld_type = 'kong'
            m.tiles.append(tile_idx)
            state.kong_count[seat] += 1
            _draw_lingshang(state, seat)
            return True

    return False


def _draw_lingshang(state: GameState, seat: int):
    """杠后从墙尾（死牌区）摸补牌（岭上牌），并更新死牌区大小"""
    if not state.wall:
        state.phase = 'end'
        return
    # 补牌从死牌区末尾取（wall[-1]），死牌区消耗 1 张
    tile = state.wall.pop()
    state.players[seat].hand.append(tile)
    state.players[seat].update_max_wilds(state.wild_idx)
    state.last_drawn = tile
    state.current_player = seat
    state.phase = 'action'
    state.lingshang_pending = True

    # 杠后重新计算死牌区大小（总杠数增加已在调用方记录）
    total_kongs = sum(state.kong_count)
    state.dead_count = max(0, _dead_count_for_kongs(total_kongs) - 1)
    # -1 因为刚才已经从死牌区取了 1 张补牌


def action_pong(state: GameState, seat: int) -> bool:
    """
    碰牌。碰万能牌后只能胡集齐4张。
    """
    tile = state.last_discard
    if tile is None:
        return False
    p = state.players[seat]
    counts, wilds = p.hand_counts(state.wild_idx)

    if tile == state.wild_idx:
        if wilds < 2:
            return False
        for _ in range(2):
            p.hand.remove(tile)
        p.melds.append(Meld('pong_wild', [tile] * 3, from_discard=True))
    else:
        if counts[tile] < 2:
            return False
        for _ in range(2):
            p.hand.remove(tile)
        p.melds.append(Meld('pong', [tile] * 3, from_discard=True))

    state.last_discard = None
    state.respond_queue = []
    state.current_player = seat
    state.phase = 'action'
    return True


def action_kong_from_discard(state: GameState, seat: int) -> bool:
    """
    从别人打出的牌进行明杠（需手中有3张）。
    杠万能牌直接胡。
    """
    tile = state.last_discard
    if tile is None:
        return False
    p = state.players[seat]
    counts, wilds = p.hand_counts(state.wild_idx)

    if tile == state.wild_idx:
        if wilds < 3:
            return False
        for _ in range(3):
            p.hand.remove(tile)
        p.melds.append(Meld('kong_wild', [tile] * 4, from_discard=True))
        trigger = state.last_discard_player
        kong_score = calc_kong_score(state.kong_count[seat])
        result = WinResult(win_types=['four_wilds'], base_score=40 + kong_score)
        _do_win(state, seat, result, lingshang=False, trigger=trigger)
        return True

    if counts[tile] < 3:
        return False
    for _ in range(3):
        p.hand.remove(tile)
    p.melds.append(Meld('kong', [tile] * 4, from_discard=True))
    state.kong_count[seat] += 1
    state.lingshang_trigger = state.last_discard_player  # 记录触发者（用于杠上开花付分）
    state.last_discard = None
    state.respond_queue = []
    _draw_lingshang(state, seat)
    return True


def action_declare_win(state: GameState, seat: int) -> bool:
    """宣布胡牌"""
    result = check_win(state, seat)
    if result is None:
        return False
    _do_win(state, seat, result, lingshang=state.lingshang_pending)
    return True


def _do_win(state: GameState, seat: int, result: WinResult, lingshang: bool, trigger: int = -1):
    state.winner = seat
    state.win_result = result
    state.is_lingshang = lingshang
    if trigger != -1:
        state.lingshang_trigger = trigger
    state.phase = 'end'


# ─────────────────────────────────────────────
# 结算
# ─────────────────────────────────────────────

def settle(state: GameState) -> dict:
    """
    结算本局，返回 {seat: score_delta}。
    流局：庄家付给每位闲家 10 分（庄家 -20，闲家各 +10）。
    胡牌：赢家收分，输家付分。
    """
    if state.winner is None:
        # 流局：庄家 -20，另外两家各 +10
        dealer = state.dealer
        return {i: (-20 if i == dealer else 10) for i in range(3)}

    winner = state.winner
    result = state.win_result
    deltas = {i: 0 for i in range(3)}

    winner_is_dealer = state.players[winner].is_dealer

    total_win = 0
    for loser in range(3):
        if loser == winner:
            continue
        loser_is_dealer = state.players[loser].is_dealer
        loser_seat_rel = (loser - winner) % 3  # 相对座位

        payment = calc_payments(
            base_score=result.base_score,
            is_lingshang=state.is_lingshang,
            winner_is_dealer=winner_is_dealer,
            loser_is_dealer=loser_is_dealer,
            lingshang_trigger_player=(state.lingshang_trigger - winner) % 3
                                      if state.lingshang_trigger != -1 else -1,
            loser_seat=loser_seat_rel,
        )
        deltas[loser] -= payment
        total_win += payment

    deltas[winner] += total_win
    for i in range(3):
        state.players[i].score += deltas[i]

    return deltas
