# reward_goemotion.py

# GoEmotions-based valence reward strategy for Kumora

def get_emotional_reward(prev_emotion, current_emotion, confidence=1.0):
    """
    Compute reward based on emotional transition between two GoEmotions labels.
    Optionally scale reward by confidence (0.0 to 1.0).
    """
    valence_scale = {
        # Strong negative (-2)
        "anger": -2, "disgust": -2, "grief": -2, "sadness": -2, "fear": -2, "remorse": -2,

        # Mild negative (-1)
        "annoyance": -1, "disapproval": -1, "disappointment": -1,
        "embarrassment": -1, "confusion": -1, "nervousness": -1,

        # Neutral (0)
        "neutral": 0, "curiosity": 0, "realization": 0, "surprise": 0,

        # Mild positive (+1)
        "admiration": 1, "amusement": 1, "excitement": 1, "pride": 1,
        "approval": 1, "relief": 1,

        # Strong positive (+2)
        "joy": 2, "love": 2, "gratitude": 2, "caring": 2, "optimism": 2, "desire": 2
    }

    prev_val = valence_scale.get(prev_emotion, 0)
    curr_val = valence_scale.get(current_emotion, 0)
    delta = curr_val - prev_val

    # Reward logic
    if delta >= 2:
        reward = 1.0
    elif delta == 1:
        reward = 0.5
    elif delta == 0:
        reward = -0.1
    else:
        reward = -0.5

    return reward * confidence

# Example test cases for real-world transitions in Kumora
if __name__ == '__main__':
    test_data = [
        ("sadness", "relief", 0.8),
        ("confusion", "joy", 0.6),
        ("fear", "caring", 0.9),
        ("anger", "neutral", 0.7),
        ("neutral", "love", 1.0),
        ("disappointment", "joy", 0.5),
        ("joy", "anger", 1.0),
        ("grief", "gratitude", 0.9),
        ("remorse", "approval", 0.7),
        ("surprise", "disapproval", 0.6)
    ]

    print("\nTesting emotional reward transitions:\n")
    for prev, curr, conf in test_data:
        r = get_emotional_reward(prev, curr, conf)
        print(f"{prev} → {curr} (confidence {conf}) → reward = {r:.2f}")
