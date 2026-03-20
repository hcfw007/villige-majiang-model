"""
计分逻辑。

术语：
  base_score    : 底分（杠分 + 牌型分）
  final_score   : 最终分（可能经杠上开花×2、庄家×2）
  payment       : 每位输家付多少分
"""


def calc_kong_score(kong_count: int) -> int:
    """计算杠的累计分值"""
    if kong_count == 0:
        return 0
    return 10 + (kong_count - 1) * 5


def calc_final_score(
    base_score: int,
    is_lingshang: bool,   # 杠上开花
    winner_is_dealer: bool,
) -> int:
    """计算赢家最终收到的总分（三家各自付分之和）"""
    score = base_score
    if is_lingshang:
        score *= 2
    if winner_is_dealer:
        score *= 2
    return score


def calc_payments(
    base_score: int,
    is_lingshang: bool,
    winner_is_dealer: bool,
    loser_is_dealer: bool,
    lingshang_trigger_player: int,  # -1 表示无触发（自摸），否则为触发者的座位索引（相对于赢家）
    loser_seat: int,                # 该输家的座位索引（相对于赢家）
) -> int:
    """
    计算某个输家需要付的分数。

    杠上开花的特殊分摊：
      - 触发者（打出牌导致明杠的玩家）付 score × 2
      - 另一家付 score × 0.5
    """
    score = base_score
    if is_lingshang:
        score *= 2
    if winner_is_dealer:
        score *= 2
    if loser_is_dealer:
        score *= 2

    if is_lingshang and lingshang_trigger_player != -1:
        if loser_seat == lingshang_trigger_player:
            return score * 2  # 触发者付双倍
        else:
            return score // 2  # 另一家付一半（向下取整）

    return score
