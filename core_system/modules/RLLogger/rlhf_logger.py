"""
# Logging System for RL Training in future versions of Kumora
# This is a RL-compatible logging system for Kumora
# We use this in app.py to track state-action-reward tuples  fir future reinforcement learning (RL) training.
# It logs user inputs, Kumora's replies, and the emotional rewards for each interaction.
# The logged data can be used to train RL models to improve Kumora's emotional intelligence and response quality.
# The logging is done in JSON Lines format for easy parsing and future analysis.
"""

import json
import os
from datetime import datetime

LOG_FILE = "kumora_rl_log.jsonl"  # JSON Lines format (one log per line)

def log_interaction(state, action, reward, reply_emotion=None, rlaif_reward=None):
    """
    Logs a single (state, action, reward) interaction for later RL training.
    `state`: User input, emotion, previous emotion, etc.
    `action`: Kumora's reply.
    `reward`: Numerical reward (e.g., emotion improvement).
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "state": state,
        "action": action,
        "reply_emotion": reply_emotion,
        "reward": reward,
        "rlaif_reward": rlaif_reward
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
