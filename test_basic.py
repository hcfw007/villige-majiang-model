"""基本功能验证"""

from env.majiang_env import MajiangEnv
from agents.random_agent import RandomAgent


def run_random_game(seed=None):
    import random
    if seed is not None:
        random.seed(seed)

    env = MajiangEnv(dealer=0)
    obs, info = env.reset()
    agents = [RandomAgent(i) for i in range(3)]

    print(f"万能牌: {info['wild_name']}  牌墙剩余: {info['wall_remaining']}")

    steps = 0
    done  = False
    while not done and steps < 500:
        seat   = info['current_player']
        action = agents[seat].act(obs, env)
        obs, rewards, done, info = env.step(action)
        steps += 1

    if info['winner'] is not None:
        wr = info['win_result']
        print(f"玩家 {info['winner']} 胡牌！牌型: {wr.win_types}  底分: {wr.base_score}")
        print(f"奖励: {rewards}")
    else:
        print("流局")
        print(f"奖励: {rewards}")

    print(f"共 {steps} 步")


if __name__ == '__main__':
    for i in range(5):
        print(f"\n── 第 {i+1} 局 ──")
        run_random_game(seed=i)
