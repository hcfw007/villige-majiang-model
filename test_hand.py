"""手牌检测单元测试"""

from env.hand import (
    check_shisanyao, check_qidui, check_standard,
    check_qingyise, check_yitiaolong, evaluate_hand,
)
from env.tile import tile_id, SUIT_MAN, SUIT_TONG, SUIT_TIAO, SUIT_ZI, NUM_TILE_TYPES


def make_counts(*tile_ids) -> list[int]:
    counts = [0] * NUM_TILE_TYPES
    for t in tile_ids:
        counts[t] += 1
    return counts


M = lambda n: tile_id(SUIT_MAN, n)
T = lambda n: tile_id(SUIT_TONG, n)
B = lambda n: tile_id(SUIT_TIAO, n)
Z = lambda n: tile_id(SUIT_ZI, n)


def test_shisanyao():
    # 十三幺：1万9万1筒9筒1条9条东南西北中发白，白多一张
    tiles = [M(1),M(9),T(1),T(9),B(1),B(9),Z(1),Z(2),Z(3),Z(4),Z(5),Z(6),Z(7),Z(7)]
    c = make_counts(*tiles)
    assert check_shisanyao(c, 0, []), "十三幺应该成立"

    # 缺1万
    tiles2 = [M(2),M(9),T(1),T(9),B(1),B(9),Z(1),Z(2),Z(3),Z(4),Z(5),Z(6),Z(7),Z(7)]
    c2 = make_counts(*tiles2)
    assert not check_shisanyao(c2, 0, []), "缺1万不应成立"

    # 有万能牌不能用
    assert not check_shisanyao(c2, 1, []), "有万能牌不应成立"
    print("✓ 十三幺")


def test_qidui():
    # 七对：2万2万 3筒3筒 5筒5筒 1条1条 2条2条 东东 南南
    tiles = [M(2),M(2), T(3),T(3), T(5),T(5), B(1),B(1), B(2),B(2), Z(1),Z(1), Z(2),Z(2)]
    c = make_counts(*tiles)
    assert check_qidui(c, 0, []), "七对应该成立"

    # 六对 + 2万能牌（野牌凑第七对，counts只含12张非野牌）
    tiles2 = [M(2),M(2), T(3),T(3), T(5),T(5), B(1),B(1), B(2),B(2), Z(1),Z(1)]
    c2 = make_counts(*tiles2)
    assert check_qidui(c2, 2, []), "六对+2野牌应该成立"

    # 有碰不能七对
    assert not check_qidui(c, 0, [('pong', [M(1)]*3)]), "有碰不能七对"
    print("✓ 七对")


def test_standard():
    # 标准4组+1将：111万 222万 333万 456万 77万
    tiles = [M(1),M(1),M(1), M(2),M(2),M(2), M(3),M(3),M(3), M(4),M(5),M(6), M(7),M(7)]
    c = make_counts(*tiles)
    assert check_standard(c, 0, []), "标准4+1应该成立"

    # 顺子：123 456 789 123筒 + 将
    tiles2 = [M(1),M(2),M(3), M(4),M(5),M(6), M(7),M(8),M(9), T(1),T(2),T(3), Z(1),Z(1)]
    c2 = make_counts(*tiles2)
    assert check_standard(c2, 0, []), "顺子4组应该成立"

    # 带万能牌：用1张野牌补完一组
    tiles3 = [M(1),M(2), M(4),M(5),M(6), M(7),M(8),M(9), T(1),T(2),T(3), Z(1),Z(1)]
    c3 = make_counts(*tiles3)
    assert check_standard(c3, 1, []), "1野牌补顺子应该成立"
    print("✓ 标准4+1")


def test_qingyise():
    # 清一色：全万
    tiles = [M(1),M(1),M(1), M(2),M(3),M(4), M(5),M(6),M(7), M(8),M(8),M(8), M(9),M(9)]
    c = make_counts(*tiles)
    assert check_qingyise(c, 0, []), "清一色全万应该成立"

    # 混有字牌不行
    tiles2 = [M(1),M(1),M(1), M(2),M(3),M(4), M(5),M(6),M(7), M(8),M(8),M(8), Z(1),Z(1)]
    c2 = make_counts(*tiles2)
    assert not check_qingyise(c2, 0, []), "含字牌不应清一色"
    print("✓ 清一色")


def test_yitiaolong():
    # 一条龙：1-9万 + 123万(面子) + 将
    # 手牌14张：1-9万 + 1万2万3万(做面子) + 东东
    tiles = [M(1),M(2),M(3),M(4),M(5),M(6),M(7),M(8),M(9), M(1),M(2),M(3), Z(1),Z(1)]
    c = make_counts(*tiles)
    assert check_yitiaolong(c, 0, []), "一条龙应该成立"

    # 带野牌补一条龙
    tiles2 = [M(1),M(2),M(3),M(4),M(5),M(7),M(8),M(9), M(1),M(2),M(3), Z(1),Z(1)]
    c2 = make_counts(*tiles2)
    assert check_yitiaolong(c2, 1, []), "1野牌补一条龙应该成立"
    print("✓ 一条龙")


def test_win_threshold():
    # 普通4组+1将，无杠，不应胡（0分 < 10分）
    tiles = [M(1),M(1),M(1), M(2),M(3),M(4), M(5),M(6),M(7), T(1),T(2),T(3), Z(1),Z(1)]
    c = make_counts(*tiles)
    result = evaluate_hand(c, 0, [], kong_score=0, total_wild_count=0)
    assert result is None, "无杠普通手牌不应胡"

    # 普通4组+1将+1杠（10分），应胡
    result2 = evaluate_hand(c, 0, [], kong_score=10, total_wild_count=0)
    assert result2 is not None, "有1杠应胡"
    assert result2.base_score == 10

    # 持有过2张野牌，门槛变20，10分不应胡
    result3 = evaluate_hand(c, 0, [], kong_score=10, total_wild_count=2)
    assert result3 is None, "持有2野牌门槛20，10分不应胡"

    # 清一色（10分），门槛10，应胡
    tiles2 = [M(1),M(1),M(1), M(2),M(3),M(4), M(5),M(6),M(7), M(8),M(8),M(8), M(9),M(9)]
    c2 = make_counts(*tiles2)
    result4 = evaluate_hand(c2, 0, [], kong_score=0, total_wild_count=0)
    assert result4 is not None, "清一色应胡"
    print("✓ 胡牌门槛")


if __name__ == '__main__':
    test_shisanyao()
    test_qidui()
    test_standard()
    test_qingyise()
    test_yitiaolong()
    test_win_threshold()
    print("\n所有测试通过！")
