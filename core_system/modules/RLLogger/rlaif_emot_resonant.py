def is_response_emotionally_resonant(user_emotion, response_emotion):
    # Example soothing/matching map
    soothing_pairs = {
        "anxious": ["calm", "hopeful"],
        "sad": ["supportive", "hopeful", "gentle"],
        "angry": ["understood", "heard", "balanced"],
        "lonely": ["connected", "seen"],
        "tired": ["restful", "gentle"]
    }

    dominant_user = user_emotion["dominant"]
    dominant_response = response_emotion["dominant"]

    # Match or soothe
    return (
        dominant_response == dominant_user or
        dominant_response in soothing_pairs.get(dominant_user, [])
    )
