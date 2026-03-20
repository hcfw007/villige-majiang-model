"""分析当前模型策略：统计动作分布、胡牌类型、碰/杠频率等"""

import random
import numpy as np
from collections import defaultdict
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env.single_agent_env import SingleAgentMajiangEnv, _random_policy
from env.majiang_env import (
    ACT_KONG_SELF, ACT_PONG, ACT_KONG_DISCARD, ACT_WIN, ACT_PASS
)
from env.tile import tile_name

model = MaskablePPO.load("checkpoints/best_model")

stats = {
    'games': 0, 'wins': 0, 'losses': 0, 'draws': 0,
    'win_types': defaultdict(int),
    'win_scores': [],
    'action_counts': defaultdict(int),
    'pong_taken': 0, 'pong_skipped': 0,
    'kong_taken': 0,
    'win_declared': 0, 'win_skipped': 0,
}

N = 2000
for i in range(N):
    random.seed(i)
    env = SingleAgentMajiangEnv(opponent_policy=_random_policy)
    env = ActionMasker(env, lambda e: e.action_masks())
    obs, _ = env.reset(seed=i)
    done = False
    ep_reward = 0.0

    while not done:
        masks = env.action_masks()
        legal = [j for j, m in enumerate(masks) if m]
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        # 统计动作
        if action == ACT_WIN:
            stats['win_declared'] += 1
        elif action == ACT_PONG:
            stats['pong_taken'] += 1
        elif action == ACT_KONG_SELF or action == ACT_KONG_DISCARD:
            stats['kong_taken'] += 1
        elif action == ACT_PASS:
            # 有没有跳过碰牌机会
            if ACT_PONG in legal:
                stats['pong_skipped'] += 1
            if ACT_WIN in legal:
                stats['win_skipped'] += 1
        elif 0 <= action <= 33:
            if ACT_WIN in legal:
                stats['win_skipped'] += 1  # 可胡但选择打牌

        obs, reward, done, _, info = env.step(action)
        ep_reward += reward

    stats['games'] += 1
    if ep_reward > 0:
        stats['wins'] += 1
    elif ep_reward < 0:
        stats['losses'] += 1
    else:
        stats['draws'] += 1

    wr = info.get('win_result')
    if info.get('winner') == 0 and wr:
        for t in wr.win_types:
            stats['win_types'][t] += 1
        stats['win_scores'].append(wr.base_score)

print(f"\n{'='*50}")
print(f"模型策略分析（{N}局）")
print(f"{'='*50}")
print(f"胜率（正分）: {stats['wins']}/{N} = {stats['wins']/N*100:.1f}%")
print(f"负分局:       {stats['losses']/N*100:.1f}%")
print(f"平均得分:     {sum(stats['win_scores'] + [-20]*stats['losses'] + [0]*stats['draws'])/N:.2f}")

print(f"\n── 动作决策 ──")
print(f"宣胡次数:         {stats['win_declared']}")
print(f"可胡但跳过:       {stats['win_skipped']}  ← 应该为0")
print(f"碰牌次数:         {stats['pong_taken']}")
print(f"可碰但跳过:       {stats['pong_skipped']}")
print(f"杠牌次数:         {stats['kong_taken']}")

if stats['win_types']:
    print(f"\n── 胡牌类型（共{stats['wins']}局胡牌）──")
    for t, c in sorted(stats['win_types'].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}局")
    if stats['win_scores']:
        print(f"平均底分: {np.mean(stats['win_scores']):.1f}")
        print(f"底分分布: {sorted(set(stats['win_scores']))}")
