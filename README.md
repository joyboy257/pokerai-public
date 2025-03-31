# PokerAI: Reinforcement Learning for Texas Hold'em

This project implements a reinforcement learning-based agent ("Abel") trained to play No-Limit Texas Hold’em Poker using self-play and curriculum learning.

## Features

- RL agent using Q-learning updates
- Scripted opponent ("Kane") for benchmarking
- Self-play training framework
- Performance metrics tracking
- Evaluation visualizations and logs

## Project Structure

| File/Folder           | Description                          |
|-----------------------|--------------------------------------|
| `ai/`                 | Reinforcement learning logic         |
| `game/`               | Poker environment and rules          |
| `scripts/`            | Training and evaluation scripts      |
| `main.py`             | Main entry point for training        |
| `requirements.txt`    | Python dependencies                  |

## Quick Start

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training
python main.py
