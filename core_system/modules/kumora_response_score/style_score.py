from prompts.kumora_emotion_responses import kumora_emotion_responses
import re

def kumora_style_score(text: str) -> float:
    score = 0.0

    # Penalize robotic phrases
    if re.search(r"\bAs an AI\b|\bI am a language model\b", text):
        score -= 2.0

    # Reward emotional validation
    if re.search(r"\bThat sounds (heavy|painful|hard|tiring)\b", text):
        score += 1.5

    # Reward gentle tone
    if "..." in text:
        score += 0.5

    # Penalize advice
    if re.search(r"\byou should\b|\btry to\b|\byou need to\b", text):
        score -= 1.0

    # Reward reflection
    if re.search(r"\bMaybe\b|\bIt makes sense\b|\bWhat if\b", text):
        score += 0.8

    # Normalize
    return max(score, 0.0)

def select_kumora_response(responses):
    """Select the most emotionally aligned Kumora-style response."""
    if not responses:
        return "I'm here with you, gently."

    # Score each response using Kumora-style
    print("Scoring responses for Kumora style...",)
    best_response = max(responses, key=lambda x: kumora_style_score(x[0]))
    
    return best_response[0]

def select_kumora_emotion_response(emotion: str) -> str:
    """Select a response based on the given emotion."""
    if emotion in kumora_emotion_responses:
        return kumora_emotion_responses[emotion]
    else:
        return "I’m here with you, no matter what you’re feeling."  # Default response