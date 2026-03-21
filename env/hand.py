"""
胡牌检测逻辑。

输入约定：
  counts     : list[int], 长度34，counts[i] 表示手牌中第i种牌的数量（不含已公开的碰/杠）
  wilds      : int，手牌中万能牌的数量
  open_melds : list of (meld_type, tiles)
               meld_type in {'pong', 'kong', 'dragon'}  # dragon=一条龙组
  wild_idx   : int，本局万能牌的牌型索引

胡牌检测返回 WinResult(win_type, score_base) 或 None。
"""

from dataclasses import dataclass
from typing import Optional
from env.tile import (
    NUM_TILE_TYPES, tile_suit, tile_number,
    is_suit_tile, SUIT_ZI, SHISANYAO_TILES,
    SUIT_MAN, SUIT_TONG, SUIT_TIAO,
)


@dataclass
class WinResult:
    win_types: list[str]   # 可能同时满足多种（如清一色+一条龙）
    base_score: int        # 底分（杠分+牌型分，杠上开花翻倍前）


# ─────────────────────────────────────────────
# 辅助：集齐4张万能牌
# ─────────────────────────────────────────────

def check_four_wilds(counts: list[int], wilds: int, open_melds: list) -> bool:
    """检测是否集齐4张万能牌（含碰/杠中的万能牌）"""
    wild_in_open = sum(
        1 for meld_type, _ in open_melds if meld_type in ('pong_wild', 'kong_wild')
    )
    # 手牌中的万能牌 + 公开中的万能牌 = 4 即可胡
    # wilds 已经是手牌中万能牌数量（调用方维护）
    total_wilds = wilds + (3 if any(t == 'pong_wild' for t, _ in open_melds) else 0) \
                        + (4 if any(t == 'kong_wild' for t, _ in open_melds) else 0)
    # 更简洁：由外部传入 total_wild_count
    return False  # 由 game 层处理，此处不重复


# ─────────────────────────────────────────────
# 辅助：十三幺
# ─────────────────────────────────────────────

def check_shisanyao(counts: list[int], wilds: int, open_melds: list) -> bool:
    """十三幺：不能用万能牌，不能有碰/杠"""
    if wilds > 0 or open_melds:
        return False
    # 必须刚好包含13种幺九字牌各至少1张，且其中一种有2张，共14张
    for t in SHISANYAO_TILES:
        if counts[t] == 0:
            return False
    extra = sum(counts[t] for t in SHISANYAO_TILES) - 13
    if extra != 1:
        return False
    # 确保手牌只有这13种牌
    for i in range(NUM_TILE_TYPES):
        if i not in SHISANYAO_TILES and counts[i] > 0:
            return False
    return True


# ─────────────────────────────────────────────
# 辅助：七对
# ─────────────────────────────────────────────

def check_qidui(counts: list[int], wilds: int, open_melds: list) -> bool:
    """七对：7对牌，万能牌可补缺，不能有碰/杠"""
    if open_melds:
        return False
    pairs = sum(c // 2 for c in counts)
    singles = sum(c % 2 for c in counts)
    # wilds 补 singles 中的落单牌，或凑成全野牌对
    wilds_needed = singles  # 每个单张需要1张野牌补齐
    remaining_wilds = wilds - wilds_needed
    if remaining_wilds < 0:
        return False
    total_pairs = pairs + singles  # singles 被野牌补齐
    # 剩余野牌必须成对
    if remaining_wilds % 2 != 0:
        return False
    total_pairs += remaining_wilds // 2
    return total_pairs == 7


# ─────────────────────────────────────────────
# 辅助：标准分解（顺/刻+将，支持野牌）
# ─────────────────────────────────────────────

def _decompose(counts: list[int], wilds: int, need_pair: bool, need_melds: int) -> bool:
    """
    递归判断 counts + wilds 能否分解成 need_melds 组面子 + (need_pair ? 1将 : 0)。
    从最小索引的牌开始处理。
    """
    if need_melds == 0 and not need_pair:
        return wilds == 0

    # 找到第一张有牌的位置
    idx = next((i for i in range(NUM_TILE_TYPES) if counts[i] > 0), -1)

    if idx == -1:
        # 只剩野牌
        wilds_needed = (2 if need_pair else 0) + 3 * need_melds
        return wilds == wilds_needed

    suit = tile_suit(idx)
    num  = tile_number(idx)

    # ── 尝试作为将（pair） ──
    if need_pair:
        # 2张正常牌
        if counts[idx] >= 2:
            counts[idx] -= 2
            if _decompose(counts, wilds, False, need_melds):
                counts[idx] += 2
                return True
            counts[idx] += 2
        # 1张正常牌 + 1张野牌
        if wilds >= 1:
            counts[idx] -= 1
            if _decompose(counts, wilds - 1, False, need_melds):
                counts[idx] += 1
                return True
            counts[idx] += 1

    # ── 尝试作为刻子（triplet） ──
    if need_melds > 0:
        if counts[idx] >= 3:
            counts[idx] -= 3
            if _decompose(counts, wilds, need_pair, need_melds - 1):
                counts[idx] += 3
                return True
            counts[idx] += 3
        if counts[idx] >= 2 and wilds >= 1:
            counts[idx] -= 2
            if _decompose(counts, wilds - 1, need_pair, need_melds - 1):
                counts[idx] += 2
                return True
            counts[idx] += 2
        if counts[idx] >= 1 and wilds >= 2:
            counts[idx] -= 1
            if _decompose(counts, wilds - 2, need_pair, need_melds - 1):
                counts[idx] += 1
                return True
            counts[idx] += 1

        # ── 尝试作为顺子起点（只有数牌可以） ──
        if suit != SUIT_ZI and num <= 7:
            i2 = idx + 1  # 同花色 +1
            i3 = idx + 2  # 同花色 +2
            # 确保 i2, i3 在同一花色
            if tile_suit(i2) == suit and tile_suit(i3) == suit:
                # 3张都有
                if counts[idx] >= 1 and counts[i2] >= 1 and counts[i3] >= 1:
                    counts[idx] -= 1; counts[i2] -= 1; counts[i3] -= 1
                    if _decompose(counts, wilds, need_pair, need_melds - 1):
                        counts[idx] += 1; counts[i2] += 1; counts[i3] += 1
                        return True
                    counts[idx] += 1; counts[i2] += 1; counts[i3] += 1
                # 缺 i3，用1野牌
                if counts[idx] >= 1 and counts[i2] >= 1 and wilds >= 1:
                    counts[idx] -= 1; counts[i2] -= 1
                    if _decompose(counts, wilds - 1, need_pair, need_melds - 1):
                        counts[idx] += 1; counts[i2] += 1
                        return True
                    counts[idx] += 1; counts[i2] += 1
                # 缺 i2，用1野牌
                if counts[idx] >= 1 and counts[i3] >= 1 and wilds >= 1:
                    counts[idx] -= 1; counts[i3] -= 1
                    if _decompose(counts, wilds - 1, need_pair, need_melds - 1):
                        counts[idx] += 1; counts[i3] += 1
                        return True
                    counts[idx] += 1; counts[i3] += 1
                # 缺 i2 和 i3，用2野牌
                if counts[idx] >= 1 and wilds >= 2:
                    counts[idx] -= 1
                    if _decompose(counts, wilds - 2, need_pair, need_melds - 1):
                        counts[idx] += 1
                        return True
                    counts[idx] += 1

    # 该位置的牌无法被消化，失败
    return False


def check_standard(counts: list[int], wilds: int, open_melds: list) -> bool:
    """
    检测是否能组成标准胡牌（4面子+1将）。
    open_melds 中的碰/杠已算作面子，计算还需要几个面子。
    """
    num_open = sum(1 for t, _ in open_melds if t in ('pong', 'kong', 'pong_wild'))
    need_melds = 4 - num_open
    if need_melds < 0:
        return False
    c = counts[:]
    return _decompose(c, wilds, True, need_melds)


# ─────────────────────────────────────────────
# 辅助：字一色
# ─────────────────────────────────────────────

def check_ziyise(counts: list[int], wilds: int, open_melds: list) -> bool:
    """字一色：所有牌（含碰/杠）均为字牌，万能牌可补"""
    for meld_type, tiles in open_melds:
        if meld_type in ('pong_wild', 'kong_wild'):
            continue
        for t in tiles:
            if tile_suit(t) != SUIT_ZI:
                return False
    for i in range(NUM_TILE_TYPES):
        if counts[i] > 0 and tile_suit(i) != SUIT_ZI:
            return False
    return True


# ─────────────────────────────────────────────
# 辅助：三调一
# ─────────────────────────────────────────────

def check_santiaoyun(
    counts: list[int], wilds: int, open_melds: list, last_drawn: Optional[int]
) -> bool:
    """
    三调一：七对手牌中摸到某牌凑成4张（纯正牌，不含万能牌顶替），
    且胡牌张恰好是这张。
    """
    if last_drawn is None:
        return False
    if not check_qidui(counts, wilds, open_melds):
        return False
    # last_drawn 在手牌中出现 ≥ 4 次（counts 只计真实牌，已排除万能牌）
    return counts[last_drawn] >= 4


# ─────────────────────────────────────────────
# 辅助：清一色
# ─────────────────────────────────────────────

def check_qingyise(counts: list[int], wilds: int, open_melds: list) -> bool:
    """清一色：所有牌（含碰/杠）同一数牌花色，万能牌可补"""
    # 检查公开面子是否只有一种花色
    open_suits = set()
    for meld_type, tiles in open_melds:
        if meld_type in ('pong_wild', 'kong_wild'):
            continue  # 万能牌面子稍后统一处理
        for t in tiles:
            if tile_suit(t) == SUIT_ZI:
                return False
            open_suits.add(tile_suit(t))
    if len(open_suits) > 1:
        return False

    target_suits = open_suits if open_suits else {SUIT_MAN, SUIT_TONG, SUIT_TIAO}

    for suit in target_suits:
        # 该花色范围：suit*9 到 suit*9+8
        start = suit * 9
        # 检查手牌中是否只有该花色牌（+ 万能牌）
        ok = True
        for i in range(NUM_TILE_TYPES):
            if counts[i] > 0 and i not in range(start, start + 9):
                ok = False
                break
        if ok:
            return True
    return False


# ─────────────────────────────────────────────
# 辅助：一条龙
# ─────────────────────────────────────────────

def check_yitiaolong(counts: list[int], wilds: int, open_melds: list) -> bool:
    """
    一条龙：手牌包含某一花色1-9的完整顺子，
    剩余牌组成若干面子+1个将。
    """
    for suit in (SUIT_MAN, SUIT_TONG, SUIT_TIAO):
        start = suit * 9
        dragon_tiles = list(range(start, start + 9))  # 该花色1-9

        # 从手牌中尝试取走1-9
        c = counts[:]
        w = wilds
        ok = True
        for t in dragon_tiles:
            if c[t] > 0:
                c[t] -= 1
            elif w > 0:
                w -= 1
            else:
                ok = False
                break
        if not ok:
            continue

        # 剩余牌数 = need_melds*3 + 2(将)
        remaining = sum(c) + w
        if remaining < 2 or (remaining - 2) % 3 != 0:
            continue
        need_melds = (remaining - 2) // 3
        if _decompose(c, w, True, need_melds):
            return True
    return False


# ─────────────────────────────────────────────
# 主检测函数
# ─────────────────────────────────────────────

def evaluate_hand(
    counts: list[int],
    wilds: int,
    open_melds: list,
    kong_score: int,
    total_wild_count: int,  # 含公开碰/杠中的万能牌
    last_drawn: Optional[int] = None,  # 本次摸到的牌（用于三调一检测）
) -> Optional[WinResult]:
    """
    检测当前手牌是否能胡牌，返回 WinResult 或 None。

    counts       : 手牌中各牌型数量（不含万能牌，不含已公开面子）
    wilds        : 手牌中万能牌数量
    open_melds   : 公开面子列表
    kong_score   : 杠的累计分（第1杠10分，后续每个+5分）
    total_wild_count : 本局持有过的最大万能牌数（含已打出的）
    """
    win_types = []
    hand_score = 0

    # ── 特殊：集齐4张万能牌 ──
    # 由 game 层直接处理（杠万能牌即胡），此处检测手牌持有4张的情况
    if total_wild_count >= 4 and wilds + sum(
        3 if t == 'pong_wild' else (4 if t == 'kong_wild' else 0)
        for t, _ in open_melds
    ) >= 4:
        win_types.append('four_wilds')
        hand_score = 40

    # ── 十三幺 ──
    if not win_types and check_shisanyao(counts, wilds, open_melds):
        win_types.append('shisanyao')
        hand_score = 100

    # ── 普通牌型（可叠加） ──
    if not win_types:
        is_qingyise   = check_qingyise(counts, wilds, open_melds)
        is_yitiaolong = check_yitiaolong(counts, wilds, open_melds)
        is_ziyise     = check_ziyise(counts, wilds, open_melds)
        is_santiaoyun = check_santiaoyun(counts, wilds, open_melds, last_drawn)
        is_qidui      = check_qidui(counts, wilds, open_melds)
        is_std        = check_standard(counts, wilds, open_melds)

        if is_qingyise:
            win_types.append('qingyise')
            hand_score += 10
        if is_yitiaolong:
            win_types.append('yitiaolong')
            hand_score += 10
        # 字一色：需同时满足某种合法手型结构
        if is_ziyise and (is_std or is_qidui or is_santiaoyun):
            win_types.append('ziyise')
            hand_score += 20
        # 三调一优先于普通七对（三调一本身已是七对的子集）
        if is_santiaoyun:
            win_types.append('santiaoyun')
            hand_score += 20
        elif is_qidui:
            win_types.append('qidui')
            hand_score += 10
        if is_std and not win_types:
            # 普通4组+1将，只靠杠分
            win_types.append('standard')

    if not win_types:
        return None

    base_score = hand_score + kong_score

    # 胡牌门槛检查
    threshold = 20 if total_wild_count >= 2 else 10
    if base_score < threshold:
        return None

    return WinResult(win_types=win_types, base_score=base_score)
