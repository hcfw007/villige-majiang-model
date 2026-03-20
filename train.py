"""
PPO 训练脚本（使用 MaskablePPO 支持合法动作掩码）。

训练策略：
  阶段1 - 对抗随机对手训练，快速学习基本策略
  阶段2 - 自我博弈（Self-play），对抗自身上一版本，持续迭代提升

用法：
  python train.py              # 从头训练
  python train.py --selfplay   # 开启自我博弈
  python train.py --resume     # 从 checkpoints/latest 继续训练
"""

import argparse
import os
import random
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.single_agent_env import SingleAgentMajiangEnv, make_model_policy, _random_policy
from env.stats_callback import StatsCallback

# RL 训练的瓶颈在游戏模拟（CPU），使用多进程并行效率最高
# MPS 对小批量神经网络更新收益有限，不如多进程并行模拟
DEVICE      = "cpu"
VecEnvClass = SubprocVecEnv

CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model")


def make_env(opponent_policy=None, seed=0):
    def _init():
        env = SingleAgentMajiangEnv(opponent_policy=opponent_policy)
        env = ActionMasker(env, lambda e: e.action_masks())
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    n_envs = args.n_envs
    print(f"训练设置: {n_envs} 并行环境, 对手策略: {'随机' if not args.selfplay else '自我博弈'}")

    # ── 阶段1：对抗随机对手 ──
    if not args.selfplay:
        envs = VecEnvClass([make_env(seed=i) for i in range(n_envs)])
        envs = VecMonitor(envs)

        if args.resume and os.path.exists(BEST_MODEL_PATH + ".zip"):
            print(f"从 {BEST_MODEL_PATH} 继续训练...")
            model = MaskablePPO.load(BEST_MODEL_PATH, env=envs)
        else:
            model = MaskablePPO(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                device=DEVICE,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_majiang",
        )

        from stable_baselines3.common.callbacks import CallbackList
        model.learn(
            total_timesteps=args.steps,
            callback=CallbackList([checkpoint_cb, StatsCallback(log_episodes=1_000)]),
            progress_bar=True,
        )
        model.save(BEST_MODEL_PATH)
        print(f"阶段1训练完成，模型保存至 {BEST_MODEL_PATH}")
        envs.close()

    # ── 阶段2：自我博弈 ──
    else:
        assert os.path.exists(BEST_MODEL_PATH + ".zip"), \
            "自我博弈需要先完成阶段1训练（运行不带 --selfplay 的命令）"

        print("加载阶段1模型作为初始对手...")
        opponent_model = MaskablePPO.load(BEST_MODEL_PATH)
        opponent_policy = make_model_policy(opponent_model)

        for iteration in range(args.selfplay_iters):
            print(f"\n── 自我博弈第 {iteration+1}/{args.selfplay_iters} 轮 ──")

            envs = VecEnvClass([
                make_env(opponent_policy=opponent_policy, seed=i + iteration * 100)
                for i in range(n_envs)
            ])
            envs = VecMonitor(envs)

            model = MaskablePPO.load(BEST_MODEL_PATH, env=envs)
            model.learn(
                total_timesteps=args.steps // args.selfplay_iters,
                progress_bar=True,
            )
            save_path = os.path.join(CHECKPOINT_DIR, f"selfplay_iter{iteration+1}")
            model.save(save_path)
            model.save(BEST_MODEL_PATH)  # 更新 best
            print(f"模型保存至 {save_path}")

            # 更新对手为新模型
            opponent_model = MaskablePPO.load(BEST_MODEL_PATH)
            opponent_policy = make_model_policy(opponent_model)
            envs.close()

        print("\n自我博弈训练完成！")


def evaluate(args):
    """评估模型对阵随机对手的胜率/得分"""
    assert os.path.exists(BEST_MODEL_PATH + ".zip"), "未找到模型文件"

    model = MaskablePPO.load(BEST_MODEL_PATH)
    opponent_policy = make_model_policy(model)

    results = {'win': 0, 'lose': 0, 'draw_game': 0, 'total_score': 0}
    n_games = args.eval_games

    for i in range(n_games):
        random.seed(i + 10000)
        env = SingleAgentMajiangEnv(opponent_policy=_random_policy)
        env = ActionMasker(env, lambda e: e.action_masks())
        obs, _ = env.reset(seed=i + 10000)
        done = False
        episode_reward = 0

        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            episode_reward += reward

        results['total_score'] += episode_reward
        if episode_reward > 0:
            results['win'] += 1
        elif episode_reward < 0:
            results['lose'] += 1
        else:
            results['draw_game'] += 1

    print(f"\n── 评估结果（{n_games} 局对抗随机对手）──")
    print(f"  正分局: {results['win']} ({results['win']/n_games*100:.1f}%)")
    print(f"  负分局: {results['lose']} ({results['lose']/n_games*100:.1f}%)")
    print(f"  零分局: {results['draw_game']} ({results['draw_game']/n_games*100:.1f}%)")
    print(f"  平均得分: {results['total_score']/n_games:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",          type=int, default=2_000_000, help="总训练步数")
    parser.add_argument("--n_envs",         type=int, default=8,         help="并行环境数")
    parser.add_argument("--selfplay",       action="store_true",          help="开启自我博弈")
    parser.add_argument("--selfplay_iters", type=int, default=5,          help="自我博弈迭代轮数")
    parser.add_argument("--resume",         action="store_true",          help="继续上次训练")
    parser.add_argument("--eval",           action="store_true",          help="只评估，不训练")
    parser.add_argument("--eval_games",     type=int, default=1000,       help="评估局数")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
