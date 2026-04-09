"""
向听数计算（距离胡牌还差几步）。

向听数 = 0 表示听牌（再摸一张就可能胡）
向听数 = -1 表示已胡牌
"""

from env.tile import (
    NUM_TILE_TYPES, tile_suit, tile_number,
    is_suit_tile, SUIT_ZI, SHISANYAO_TILES,
)


def _shanten_standard(counts: list[int], wilds: int, n_melds_fixed: int) -> int:
    """
    标准型向听数：需要凑 (4-n_melds_fixed) 个面子 + 1 将。
    返回向听数（-1=已完成）。
    """
    need_melds = 4 - n_melds_fixed
    best = [8]  # 最坏情况

    def dfs(idx: int, wilds_left: int, melds: int, pairs: int, partial: int):
        # 剩余需要的面子数
        need = need_melds - melds
        # 向听数 = need_melds - melds + (1 if pairs==0 else 0) - 1 (partial pairs/melds)
        # 简化估算
        val = need_melds - melds + (1 if pairs == 0 else 0) - partial
        if val < best[0]:
            best[0] = val
        if best[0] <= 0:
            return

        # 找第一张有牌的位置
        while idx < NUM_TILE_TYPES and counts[idx] == 0:
            idx += 1
        if idx == NUM_TILE_TYPES:
            return

        suit = tile_suit(idx)
        num  = tile_number(idx)

        # 尝试作为将
        if pairs == 0:
            if counts[idx] >= 2:
                counts[idx] -= 2
                dfs(idx, wilds_left, melds, 1, partial)
                counts[idx] += 2
            if wilds_left >= 1:
                counts[idx] -= 1
                dfs(idx, wilds_left - 1, melds, 1, partial)
                counts[idx] += 1
            if wilds_left >= 2:
                dfs(idx, wilds_left - 2, melds, 1, partial)

        # 尝试作为刻子
        if counts[idx] >= 3:
            counts[idx] -= 3
            dfs(idx, wilds_left, melds + 1, pairs, partial)
            counts[idx] += 3
        if counts[idx] >= 2 and wilds_left >= 1:
            counts[idx] -= 2
            dfs(idx, wilds_left - 1, melds + 1, pairs, partial)
            counts[idx] += 2
        if counts[idx] >= 1 and wilds_left >= 2:
            counts[idx] -= 1
            dfs(idx, wilds_left - 2, melds + 1, pairs, partial)
            counts[idx] += 1
        if wilds_left >= 3:
            dfs(idx, wilds_left - 3, melds + 1, pairs, partial)

        # 尝试作为顺子
        if suit != SUIT_ZI and num <= 7:
            i2, i3 = idx + 1, idx + 2
            if tile_suit(i2) == suit and tile_suit(i3) == suit:
                if counts[idx] >= 1 and counts[i2] >= 1 and counts[i3] >= 1:
                    counts[idx] -= 1; counts[i2] -= 1; counts[i3] -= 1
                    dfs(idx, wilds_left, melds + 1, pairs, partial)
                    counts[idx] += 1; counts[i2] += 1; counts[i3] += 1

        # 部分顺子（搭子）
        if suit != SUIT_ZI:
            for d in [1, 2]:
                j = idx + d
                if j < NUM_TILE_TYPES and tile_suit(j) == suit and counts[j] >= 1:
                    counts[idx] -= 1; counts[j] -= 1
                    dfs(idx, wilds_left, melds, pairs, partial + 1)
                    counts[idx] += 1; counts[j] += 1

        # 跳过（这张牌无法使用）
        counts[idx] -= 1
        dfs(idx + 1, wilds_left, melds, pairs, partial)
        counts[idx] += 1

    dfs(0, wilds, 0, 0, 0)
    return best[0]


def _shanten_qidui(counts: list[int], wilds: int) -> int:
    """七对向听数"""
    pairs  = sum(1 for c in counts if c >= 2)
    singles = sum(1 for c in counts if c == 1)
    # 野牌优先补单张
    wild_used = min(wilds, singles)
    pairs += wild_used
    remaining_wilds = wilds - wild_used
    pairs += remaining_wilds // 2
    return max(0, 6 - pairs)


def _shanten_shisanyao(counts: list[int], wilds: int) -> int:
    """十三幺向听数（不能用万能牌）"""
    if wilds > 0:
        return 13  # 不可达
    have = sum(1 for t in SHISANYAO_TILES if counts[t] >= 1)
    has_pair = any(counts[t] >= 2 for t in SHISANYAO_TILES)
    return 13 - have - (1 if has_pair else 0)


def _shanten_yitiaolong(counts: list[int], wilds: int, n_melds_fixed: int) -> int:
    """
    一条龙向听数：某花色1-9各一张 + 剩余牌凑面子+将。

    计算方式：
      dragon_gap  = 龙中缺的牌数 - 可用万能牌数（≥0）
      rest_shanten = 剩余手牌凑 (need_melds 面子 + 1将) 的向听数
      总向听 = dragon_gap + rest_shanten
    """
    from env.tile import SUIT_MAN, SUIT_TONG, SUIT_TIAO
    best = 13
    for suit in (SUIT_MAN, SUIT_TONG, SUIT_TIAO):
        start = suit * 9
        # 龙需要的9张牌中缺多少
        missing = 0
        c = counts[:]
        for n in range(9):
            t = start + n
            if c[t] > 0:
                c[t] -= 1
            else:
                missing += 1
        # 万能牌先补龙的缺口
        wilds_for_dragon = min(wilds, missing)
        dragon_gap = missing - wilds_for_dragon
        wilds_left = wilds - wilds_for_dragon
        # 剩余手牌需要凑的面子数
        need_melds = 4 - n_melds_fixed - 3  # 龙本身占3个面子位
        if need_melds < 0:
            # 公开面子太多，不可能组一条龙
            continue
        # 剩余牌的标准向听
        rest = _shanten_standard(c, wilds_left, n_melds_fixed + 3)
        # 调整：_shanten_standard 算的是 need=(4-n_melds_fixed-3) 面子+将
        # dragon_gap 是龙还差几张
        total = dragon_gap + max(0, rest)
        if total < best:
            best = total
    return best


def calc_shanten(counts: list[int], wilds: int, n_melds_fixed: int = 0) -> int:
    """
    计算最小向听数（取所有胡牌型中的最小值）。
    counts: 手牌计数（不含万能牌）
    wilds: 万能牌数量
    n_melds_fixed: 已公开的碰/杠组数
    """
    s_std  = _shanten_standard(counts[:], wilds, n_melds_fixed)
    s_7    = _shanten_qidui(counts, wilds) if n_melds_fixed == 0 else 13
    s_13   = _shanten_shisanyao(counts, 0) if n_melds_fixed == 0 else 13
    s_ytl  = _shanten_yitiaolong(counts, wilds, n_melds_fixed)
    return min(s_std, s_7, s_13, s_ytl)
