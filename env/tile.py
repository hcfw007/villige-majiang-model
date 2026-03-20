"""
牌的基本定义与工具函数。

编码规则：
  0- 8: 万1-9
  9-17: 筒1-9
 18-26: 条1-9
 27-33: 字1-7（东南西北中发白）
"""

NUM_TILE_TYPES = 34
TILES_PER_TYPE = 4
TOTAL_TILES = NUM_TILE_TYPES * TILES_PER_TYPE  # 136

SUIT_MAN  = 0  # 万
SUIT_TONG = 1  # 筒
SUIT_TIAO = 2  # 条
SUIT_ZI   = 3  # 字

SUIT_NAMES = ['万', '筒', '条', '字']
ZI_NAMES   = ['东', '南', '西', '北', '中', '发', '白']


def tile_id(suit: int, number: int) -> int:
    """将花色+点数转为牌型索引 (0-33)"""
    if suit == SUIT_ZI:
        return 27 + number - 1
    return suit * 9 + number - 1


def tile_suit(idx: int) -> int:
    if idx >= 27:
        return SUIT_ZI
    return idx // 9


def tile_number(idx: int) -> int:
    if idx >= 27:
        return idx - 27 + 1
    return idx % 9 + 1


def is_suit_tile(idx: int) -> bool:
    """是否为数牌（万/筒/条）"""
    return idx < 27


def tile_name(idx: int) -> str:
    suit = tile_suit(idx)
    num  = tile_number(idx)
    if suit == SUIT_ZI:
        return ZI_NAMES[num - 1]
    return f"{num}{SUIT_NAMES[suit]}"


def all_tile_types() -> list[int]:
    return list(range(NUM_TILE_TYPES))


# 十三幺所需的13种牌
SHISANYAO_TILES = [
    tile_id(SUIT_MAN,  1), tile_id(SUIT_MAN,  9),
    tile_id(SUIT_TONG, 1), tile_id(SUIT_TONG, 9),
    tile_id(SUIT_TIAO, 1), tile_id(SUIT_TIAO, 9),
    tile_id(SUIT_ZI, 1), tile_id(SUIT_ZI, 2), tile_id(SUIT_ZI, 3),
    tile_id(SUIT_ZI, 4), tile_id(SUIT_ZI, 5), tile_id(SUIT_ZI, 6),
    tile_id(SUIT_ZI, 7),
]
