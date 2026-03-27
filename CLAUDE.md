# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Train from scratch vs random opponents
python train.py --steps 2000000 --n_envs 8

# Phase 2: Self-play training
python train.py --selfplay --steps 2000000 --selfplay_iters 5

# Continuous training until target score
python train_until.py --target 8.0 --n_envs 8 --steps_per_round 800000

# Evaluate current best model
python train.py --eval --eval_games 1000

# Run sanity tests
python test_basic.py
python test_hand.py

# Analyze trained model strategy
python inspect_strategy.py
```

## Architecture

### Game Rules (Custom 3-Player Variant)
- 136 tiles (万/筒/条/字), no flowers, no chow
- One wild tile type (癞子) determined per game via dice roll
- **Self-draw only** — players cannot claim discards to win; only pong and kong on discards
- Win threshold: ≥10 base score, or ≥20 if the player ever held ≥2 wild tiles
- Win types: standard (4 melds+pair), 清一色, 一条龙, 七对, 三调一, 字一色, 十三幺, 开会 (4 wild tiles, 40 pts)

### Environment Layers
- **`env/game.py`** — Core game state (`GameState`, `PlayerState`, `Meld`), game phases (`draw` → `action` → `respond` → `end`), and action executors
- **`env/majiang_env.py`** — Gymnasium multi-agent env. Builds 382-dim observations and 39-action masked space. Dark kongs (暗杠) are hidden from opponents.
- **`env/single_agent_env.py`** — Single-agent RL wrapper. Seat 0 = RL agent; seats 1 & 2 = `opponent_policy` callable. Handles reward shaping and episode termination.
- **`env/hand.py`** — Win detection: `evaluate_hand()` returns `WinResult(win_types, base_score)`
- **`env/shanten.py`** — `calc_shanten()` for distance-to-win, used in intermediate rewards
- **`env/scorer.py`** — Settlement and kong score calculations
- **`env/stats_callback.py`** — `StatsCallback` logs per-episode win types, behavioral stats (pong/kong/win rates) to `logs/`

### Observation Space (382-dim)
| Range | Content |
|-------|---------|
| 0–33 | Self hand tile counts |
| 34–135 | Open melds (self, left, right) |
| 136–237 | Discard piles (self, left, right) |
| 238–271 | All visible tiles |
| 272–305 | Wild tile one-hot |
| 306–317 | Game state features (dealer, remaining tiles, kongs, shanten, etc.) |
| 318–381 | Last discard one-hot + suit presence bits |

### Action Space (39 actions)
- `0–33`: Discard tile type
- `34`: Self kong, `35`: Pong, `36`: Kong from discard, `37`: Declare win, `38`: Pass

### Reward Shaping
- **Terminal:** `settlement_delta` if win/opponent wins; `settlement_delta - LIUJU_PENALTY(3.0)` if draw
- **Intermediate:** Shanten improvement rewards (`SHANTEN_REWARD_SCALE=0.4`), tenpai bonus (`TENPAI_BONUS=3.0`), kong reward (`KONG_REWARD=2.0`), jiakang-while-tenpai bonus (`JIAKANG_TENPAI_BONUS=4.0`)
- Path viability filter: shanten rewards only apply when the hand is pursuing a valid scoring path (est. score ≥ threshold)

### Training
- **Algorithm:** MaskablePPO (`sb3_contrib`), MlpPolicy `[512, 512, 256]`, CPU-only (simulation is the bottleneck)
- **Parallelism:** `SubprocVecEnv` with 8 workers
- **Self-play (train_until.py):** Maintains pool of last 5 checkpoints; every 5th round trains vs random to prevent degeneration
- **Checkpoints:** `checkpoints/best_model.zip` is the canonical model used for eval and self-play

### Key Files
```
env/
  tile.py            # Tile encoding (34 types), suit/name utilities
  game.py            # Game state and action logic
  majiang_env.py     # Gymnasium env, observation builder, action masking
  single_agent_env.py# RL wrapper, reward shaping, opponent integration
  hand.py            # Win hand detection
  shanten.py         # Shanten calculation
  scorer.py          # Score/payment calculation
  stats_callback.py  # Training statistics logging
agents/
  random_agent.py    # Baseline random opponent
train.py             # Single-phase training + evaluation
train_until.py       # Continuous self-play training loop
checkpoints/         # Saved model files (gitignored)
logs/                # Training logs and game stats
```
