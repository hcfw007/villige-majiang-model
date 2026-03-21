"""
训练统计回调：每 log_episodes 局输出一次小结。

小结内容：
  - 平均得分
  - 终局结果统计（流局 / 各牌型胡牌 / 对手胡牌）
  - 策略行为统计（碰率、杠率、宣胡率）
"""

import json
import os
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback

WIN_TYPE_ZH = {
    "standard":                  "普通（靠杠）",
    "qingyise":                  "清一色",
    "yitiaolong":                "一条龙",
    "qingyise+yitiaolong":       "清一色+一条龙",
    "qidui":                     "七对",
    "santiaoyun":                "三调一",
    "ziyise":                    "字一色",
    "qidui+ziyise":              "字一色七对",
    "santiaoyun+ziyise":         "字一色三调一",
    "shisanyao":                 "十三幺",
    "four_wilds":                "开会",
}

def _type_key(win_types: list) -> str:
    """将 win_types 列表转为统计 key，多牌型组合单独列出"""
    return "+".join(sorted(win_types))


class StatsCallback(BaseCallback):
    def __init__(self, log_episodes: int = 10_000, log_dir: str = "logs", verbose: int = 0):
        super().__init__(verbose)
        self.log_episodes = log_episodes
        self.log_dir      = log_dir
        self.stats_file   = os.path.join(log_dir, "game_stats.jsonl")
        self.summary_file = os.path.join(log_dir, "summary.txt")
        os.makedirs(log_dir, exist_ok=True)

        self._total_episodes = 0   # 历史累计局数（不重置）
        self._last_log_ep    = 0
        self._reset_counters()

        # ── 累计计数器（永不重置）──
        self._cum_liuju        = 0
        self._cum_agent_wins   = 0
        self._cum_opp_wins     = 0
        self._cum_agent_types  = defaultdict(int)
        self._cum_opp_types    = defaultdict(int)
        self._cum_total_reward = 0.0
        self._cum_win_scores   = []
        self._cum_pong_taken   = 0
        self._cum_pong_skipped = 0
        self._cum_kong_taken   = 0
        self._cum_win_declared = 0
        self._cum_win_skipped  = 0

    def _reset_counters(self):
        self.episodes      = 0
        self.liuju         = 0
        self.agent_wins    = 0
        self.opp_wins      = 0
        self.agent_types   = defaultdict(int)   # 智能体胡牌牌型
        self.opp_types     = defaultdict(int)   # 对手胡牌牌型
        self.total_reward  = 0.0
        self.win_scores    = []                 # 胡牌底分列表
        # 策略行为（从 info 里取）
        self.pong_taken    = 0
        self.pong_skipped  = 0
        self.kong_taken    = 0
        self.win_declared  = 0
        self.win_skipped   = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue

            self.episodes        += 1
            self._total_episodes += 1
            r = float(ep["r"])
            self.total_reward        += r
            self._cum_total_reward   += r

            winner     = info.get("winner")
            win_result = info.get("win_result")

            if winner is None:
                self.liuju           += 1
                self._cum_liuju      += 1
            elif winner == 0:
                self.agent_wins          += 1
                self._cum_agent_wins     += 1
                if win_result and hasattr(win_result, "win_types"):
                    key = _type_key(win_result.win_types)
                    self.agent_types[key]      += 1
                    self._cum_agent_types[key] += 1
                    score = float(win_result.base_score)
                    self.win_scores.append(score)
                    self._cum_win_scores.append(score)
            else:
                self.opp_wins        += 1
                self._cum_opp_wins   += 1
                if win_result and hasattr(win_result, "win_types"):
                    key = _type_key(win_result.win_types)
                    self.opp_types[key]      += 1
                    self._cum_opp_types[key] += 1

            # 策略行为统计
            pt = int(info.get("pong_taken",   0))
            ps = int(info.get("pong_skipped", 0))
            kt = int(info.get("kong_taken",   0))
            wd = int(info.get("win_declared", 0))
            self.pong_taken   += pt;  self._cum_pong_taken   += pt
            self.pong_skipped += ps;  self._cum_pong_skipped += ps
            self.kong_taken   += kt;  self._cum_kong_taken   += kt
            ws = int(info.get("win_skipped", 0))
            self.win_declared += wd;  self._cum_win_declared += wd
            self.win_skipped  += ws;  self._cum_win_skipped  += ws

        # 每 log_episodes 局输出一次
        if self.episodes > 0 and self.episodes - self._last_log_ep >= self.log_episodes:
            self._print_summary()
            self._last_log_ep = self.episodes

        return True

    def _print_summary(self):
        n   = self.episodes
        avg = self.total_reward / n

        bar = "=" * 56
        print(f"\n{bar}")
        print(f"  对局小结  [累计 {self._total_episodes:,} 局 / 训练 {self.num_timesteps:,} 步]")
        print(bar)

        # ── 平均得分 ──
        print(f"  平均得分:    {avg:+.2f}")

        # ── 终局结果 ──
        print(f"\n  终局结果（本段 {n:,} 局）:")
        print(f"    流局:         {self.liuju:5d} 局  {self.liuju/n*100:5.1f}%")
        print(f"    智能体胡牌:   {self.agent_wins:5d} 局  {self.agent_wins/n*100:5.1f}%")
        print(f"    对手胡牌:     {self.opp_wins:5d} 局  {self.opp_wins/n*100:5.1f}%")

        # ── 智能体牌型 ──
        if self.agent_types:
            print(f"\n  智能体胡牌牌型:")
            for t, c in sorted(self.agent_types.items(), key=lambda x: -x[1]):
                zh = WIN_TYPE_ZH.get(t, t)
                print(f"    {zh:12s}: {c:4d} 局  {c/n*100:4.1f}%")
            if self.win_scores:
                import numpy as np
                print(f"    平均底分: {np.mean(self.win_scores):.1f}  "
                      f"最高: {max(self.win_scores):.0f}")

        # ── 对手牌型 ──
        if self.opp_types:
            print(f"\n  对手胡牌牌型:")
            for t, c in sorted(self.opp_types.items(), key=lambda x: -x[1]):
                zh = WIN_TYPE_ZH.get(t, t)
                print(f"    {zh:12s}: {c:4d} 局  {c/n*100:4.1f}%")

        # ── 策略行为 ──
        total_pong_opp = self.pong_taken + self.pong_skipped
        pong_rate = self.pong_taken / total_pong_opp * 100 if total_pong_opp > 0 else 0
        print(f"\n  策略行为:")
        print(f"    碰牌率:  {pong_rate:5.1f}%  "
              f"（碰 {self.pong_taken} / 放弃 {self.pong_skipped}）")
        print(f"    杠牌次数: {self.kong_taken} 次  "
              f"（平均 {self.kong_taken/n:.2f} 次/局）")
        print(f"    宣胡次数: {self.win_declared} 次  "
              f"（成功率 {self.agent_wins/max(1,self.win_declared)*100:.0f}%）")
        print(f"    能胡不胡: {self.win_skipped} 次  "
              f"（每局 {self.win_skipped/n:.2f} 次）")

        # ── 累计汇总 ──
        N = self._total_episodes
        if N > 0:
            cum_avg  = self._cum_total_reward / N
            cum_pong_total = self._cum_pong_taken + self._cum_pong_skipped
            cum_pong_rate  = self._cum_pong_taken / cum_pong_total * 100 if cum_pong_total > 0 else 0
            print(f"\n  ── 累计汇总（共 {N:,} 局）──")
            print(f"    平均得分:     {cum_avg:+.2f}")
            print(f"    流局率:       {self._cum_liuju/N*100:5.1f}%")
            print(f"    智能体胡牌率: {self._cum_agent_wins/N*100:5.2f}%")
            print(f"    对手胡牌率:   {self._cum_opp_wins/N*100:5.2f}%")
            if self._cum_agent_types:
                print(f"    智能体牌型:")
                for t, c in sorted(self._cum_agent_types.items(), key=lambda x: -x[1]):
                    zh = WIN_TYPE_ZH.get(t, t)
                    print(f"      {zh:12s}: {c:4d} 局  {c/N*100:4.2f}%")
                if self._cum_win_scores:
                    import numpy as np
                    print(f"    累计平均底分: {np.mean(self._cum_win_scores):.1f}  "
                          f"最高: {max(self._cum_win_scores):.0f}")
            print(f"    碰牌率:       {cum_pong_rate:5.1f}%")
            print(f"    平均杠/局:    {self._cum_kong_taken/N:.3f}")
        print(bar)

        # 写入文件
        cum_pong_total = self._cum_pong_taken + self._cum_pong_skipped
        cum_pong_rate  = self._cum_pong_taken / cum_pong_total if cum_pong_total > 0 else 0
        record = {
            "total_episodes": self._total_episodes,
            "steps":          int(self.num_timesteps),
            "episodes":       n,
            "avg_reward":     round(float(avg), 3),
            "liuju_rate":     round(self.liuju / n, 4),
            "agent_win_rate": round(self.agent_wins / n, 4),
            "opp_win_rate":   round(self.opp_wins / n, 4),
            "agent_types":    {k: int(v) for k, v in self.agent_types.items()},
            "opp_types":      {k: int(v) for k, v in self.opp_types.items()},
            "avg_win_score":  round(float(sum(self.win_scores)/len(self.win_scores)), 2)
                              if self.win_scores else 0,
            "pong_rate":      round(pong_rate / 100, 4),
            "kong_per_game":  round(self.kong_taken / n, 4),
            "win_skipped_per_game": round(self.win_skipped / n, 4),
            # 累计字段
            "cum_avg_reward":     round(self._cum_total_reward / self._total_episodes, 3) if self._total_episodes else 0,
            "cum_liuju_rate":     round(self._cum_liuju / self._total_episodes, 4) if self._total_episodes else 0,
            "cum_agent_win_rate": round(self._cum_agent_wins / self._total_episodes, 4) if self._total_episodes else 0,
            "cum_opp_win_rate":   round(self._cum_opp_wins / self._total_episodes, 4) if self._total_episodes else 0,
            "cum_agent_types":    {k: int(v) for k, v in self._cum_agent_types.items()},
            "cum_avg_win_score":  round(float(sum(self._cum_win_scores) / len(self._cum_win_scores)), 2) if self._cum_win_scores else 0,
            "cum_pong_rate":      round(cum_pong_rate, 4),
            "cum_kong_per_game":  round(self._cum_kong_taken / self._total_episodes, 4) if self._total_episodes else 0,
        }
        with open(self.stats_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._reset_counters()
