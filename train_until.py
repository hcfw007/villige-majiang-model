"""
持续训练直到评估平均分 >= 目标分数。

策略：
  - 每轮 self-play 80万步，评估2000局
  - 自我博弈对手从最近 POOL_SIZE 个 checkpoint 中随机采样
  - 每5轮也对抗随机对手训练一轮，防止策略退化
"""

import os
import random
import argparse
import numpy as np
import torch

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.single_agent_env import SingleAgentMajiangEnv, make_model_policy, _random_policy
from env.stats_callback import StatsCallback

DEVICE      = "cpu"
VecEnvClass = SubprocVecEnv

CHECKPOINT_DIR  = "checkpoints"
LOG_DIR         = "logs"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model")
POOL_SIZE       = 5  # 对手池大小：从最近 N 个 checkpoint 随机采样对手

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def get_pool_opponent(pool: list):
    """从历史 checkpoint 池随机选一个对手；池为空则用当前最佳模型"""
    path = random.choice(pool) if pool else BEST_MODEL_PATH
    opp_model = MaskablePPO.load(path)
    return make_model_policy(opp_model)


def make_env(opponent_policy=None, seed=0):
    def _init():
        env = SingleAgentMajiangEnv(opponent_policy=opponent_policy)
        env = ActionMasker(env, lambda e: e.action_masks())
        env.reset(seed=seed)
        return env
    return _init


def quick_eval(model, n_games=2000) -> float:
    """快速评估：对抗随机对手，返回平均终局得分（不含中间奖励）"""
    total = 0.0
    for i in range(n_games):
        random.seed(i + 99999)
        env = SingleAgentMajiangEnv(opponent_policy=_random_policy)
        env = ActionMasker(env, lambda e: e.action_masks())
        obs, _ = env.reset(seed=i + 99999)
        done = False
        terminal_reward = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            if done:
                terminal_reward = reward  # 只取终局得分，忽略中间奖励
        total += terminal_reward
    return total / n_games


def train_round(opponent_policy, steps_per_round, n_envs, seed_offset, tag):
    """从 best_model 加载，训练一轮，保存并返回新模型"""
    envs = VecEnvClass([
        make_env(opponent_policy=opponent_policy, seed=i + seed_offset)
        for i in range(n_envs)
    ])
    envs = VecMonitor(envs)
    model = MaskablePPO.load(BEST_MODEL_PATH, env=envs)
    model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False, progress_bar=True,
                callback=StatsCallback(log_episodes=1_000, log_dir=LOG_DIR))
    envs.close()
    # 保存前先备份当前 best 为 prev（用于 vs 对比）
    import shutil
    if os.path.exists(BEST_MODEL_PATH + ".zip"):
        shutil.copy(BEST_MODEL_PATH + ".zip",
                    os.path.join(CHECKPOINT_DIR, "prev_model.zip"))
    save_path = os.path.join(CHECKPOINT_DIR, tag)
    model.save(save_path)
    model.save(BEST_MODEL_PATH)
    return model


def vs_prev_eval(n_games: int = 300) -> float:
    """
    用当前 best_model 对抗上一版本（prev_model），
    返回当前模型的平均得分（正值=当前版本更强）。
    """
    prev_path = os.path.join(CHECKPOINT_DIR, "prev_model.zip")
    if not os.path.exists(prev_path):
        return float("nan")

    current = MaskablePPO.load(BEST_MODEL_PATH)
    prev    = MaskablePPO.load(prev_path)
    opp_policy = make_model_policy(prev)

    total = 0.0
    for i in range(n_games):
        random.seed(i + 55555)
        env = SingleAgentMajiangEnv(opponent_policy=opp_policy)
        env = ActionMasker(env, lambda e: e.action_masks())
        obs, _ = env.reset(seed=i + 55555)
        done = False
        terminal_r = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = current.predict(obs, action_masks=masks, deterministic=True)
            obs, r, done, _, _ = env.step(int(action))
            if done:
                terminal_r = r  # 只取终局得分
        total += terminal_r
    return total / n_games


def main(args):
    assert os.path.exists(BEST_MODEL_PATH + ".zip"), \
        "请先运行阶段1训练：python3 train.py --steps 2000000 --n_envs 8"

    target      = args.target
    steps_round = args.steps_per_round
    n_envs      = args.n_envs

    model = MaskablePPO.load(BEST_MODEL_PATH)

    # 初始评估
    score = quick_eval(model)
    print(f"\n[初始] 平均得分(vs随机): {score:.2f}  目标: {target}")

    checkpoint_pool = []  # 历史 checkpoint 路径池（最近 POOL_SIZE 个）
    iteration = 0
    while score < target:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"第 {iteration} 轮训练  当前: {score:.2f}  目标: {target}")
        print(f"{'='*50}")

        # 每5轮插入一轮对抗随机对手，保持基础能力
        if iteration % 5 == 0:
            print("→ 对抗随机对手训练（防退化）")
            tag = f"random_round{iteration}"
            model = train_round(
                _random_policy, steps_round, n_envs,
                seed_offset=iteration * 1000,
                tag=tag,
            )
        else:
            # 自我博弈：从对手池随机采样
            opponent_policy = get_pool_opponent(checkpoint_pool)
            print(f"→ 自我博弈训练（对手池 {len(checkpoint_pool)} 个）")
            tag = f"selfplay_round{iteration}"
            model = train_round(
                opponent_policy, steps_round, n_envs,
                seed_offset=iteration * 1000,
                tag=tag,
            )

        # 更新对手池
        checkpoint_pool.append(os.path.join(CHECKPOINT_DIR, tag))
        if len(checkpoint_pool) > POOL_SIZE:
            checkpoint_pool.pop(0)

        score = quick_eval(model)
        vs_score = vs_prev_eval(n_games=300)
        vs_str = f"{vs_score:+.2f}" if not np.isnan(vs_score) else "N/A"
        print(f"\n[第{iteration}轮后] vs随机: {score:.2f}  vs上版本: {vs_str}  目标: {target}")

        if score >= target:
            break

        # 超过最大轮次则扩大每轮步数
        if iteration % 20 == 0:
            steps_round = min(steps_round * 2, 2_000_000)
            print(f"  ↑ 每轮步数提升至 {steps_round:,}")

    print(f"\n{'='*50}")
    print(f"目标达成！最终平均得分: {score:.2f}  共训练 {iteration} 轮")
    print(f"最佳模型: {BEST_MODEL_PATH}.zip")

    # 最终详细评估
    final_score = quick_eval(model, n_games=1000)
    print(f"\n[1000局最终评估] 平均得分: {final_score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",          type=float, default=8.0,     help="目标平均分")
    parser.add_argument("--steps_per_round", type=int,   default=800_000, help="每轮训练步数")
    parser.add_argument("--n_envs",          type=int,   default=8,       help="并行环境数")
    args = parser.parse_args()
    main(args)
