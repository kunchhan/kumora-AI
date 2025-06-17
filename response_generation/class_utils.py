from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
# ==================== Enums and Data Classes ====================

class SupportType(Enum):
    """Types of emotional support"""
    NEUTRAL = "neutral"
    CONVERSATIONAL_INQUIRY = "conversational_inquiry"
    CRISIS = "crisis"
    VALIDATION = "validation"
    GROWTH = "growth"
    GENERAL = "general"
    CELEBRATION = "celebration"
    PROBLEM_SOLVING = "problem_solving"


class EmpathyLevel(Enum):
    """Controls the depth of emotional reflection in the AI's response."""
    LOW = 1         # Factual, minimal emotional engagement
    MEDIUM = 2      # Balanced emotional acknowledgment
    HIGH = 3        # Deep emotional engagement
    ADAPTIVE = 4    # Adjusts based on user's state


class ResponseStyle(Enum):
    """Defines the overall tone and persona of Kumora's responses."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    WARM = "warm"
    GENTLE = "gentle"
    ENCOURAGING = "encouraging"
    REFLECTIVE = "reflective"


@dataclass
class EmotionalContext:
    """
    Analyzes and categorizes emotional data from the classifier 
    to inform downstream prompt selection.
    """
    primary_emotion: str
    detected_emotions: List[str]
    intensity: float
    valence: str
    confidence: float = 0.8
    triggers: List[str] = field(default_factory=list)
    
    def get_emotion_category(self) -> str:
        """Categorize emotions for prompt selection"""

        # High-distress emotions that may indicate a need for more careful, in-depth support.
        negative_high = [
            'Feeling overwhelmed',
            'Loneliness or Isolation',
            'Low self-esteem',
            'Sensitivity to rejection'
        ]

        # Actionable negative emotions that call for validation and support.
        negative_medium = [
            'Anger or frustration',
            'Anxiety',
            'Emotional sensitivity',
            'Irritability',
            'Mood swings',
            'Physical discomfort',
            'Restlessness',
            'Sadness',
            'Tearfulness'
        ]

        # All positive emotions that can be reinforced to encourage growth and well-being.
        positive = [
            'Attractiveness',
            'Clarity',
            'Confidence',
            'Empowerment',
            'Feeling in control',
            'High energy',
            'Hopefulness',
            'Improved mood',
            'Motivation',
            'Optimism',
            'Productivity',
            'Renewed energy',
            'Sexual drive',
            'Sociability'
        ]

        # --- Categorization Logic ---
        # We check all detected emotions, not just the primary, for a robust safety net.
        # Rule 1: Check for high-negative emotions, especially with high intensity.
        if self.primary_emotion in negative_high and self.intensity > 0.7:
            return "crisis"
        # Rule 2: Check if a high-negative emotion is detected but intensity is lower, treat as 'support'
        elif self.primary_emotion in negative_high:
            return "support"
        # Rule 3: Check if the primary emotion falls into the medium-negative category.
        elif self.primary_emotion in negative_medium:
            return "support"
        # Rule 4: Check if the primary emotion is positive.
        elif self.primary_emotion in positive:
            return "growth"
        # Rule 5: Fallback for any emotions that might not be in the lists or for mixed signals.
        else:
            return "general"


@dataclass
class UserContext:
    """User context from context management system"""
    user_id: str
    active_goals: List[Dict] = field(default_factory=list)
    recent_topics: List[str] = field(default_factory=list)
    emotional_trajectory: str = "stable"
    effective_strategies: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    session_number: int = 1
    
    def get_relationship_depth(self) -> str:
        """Determine relationship depth based on session history"""
        if self.session_number <= 1:
            return "new"
        elif self.session_number <= 5:
            return "developing"
        elif self.session_number <= 20:
            return "established"
        else:
            return "deep"


@dataclass
class PromptConfig:
    """
    Configuration dataclass for generating prompts sent to the Large Language Model.
    
    This class centralizes all parameters that control the LLM's response generation,
    making it easy to fine-tune Kumora's behaviour, tone, and safety guardrails from 
    a single, well-documented source.
    """
    # --- Generation Control Parameters ---
    max_tokens: int = 200
    """
    The maximum number of tokens (words/sub-words) for the generated response.
    Purpose: Controls response length to keep answers concise and focused, preventing user fatigue.
    Tuninc for Kumora: A value between 150-250 is ideal to be supportive without being overly verbose.
    """

    temperature: float = 0.7
    """
    Controls the randomness and "creativity" of the LLM's output.
    Range: 0.0 (deterministic) to 2.0 (highly random).
    Purpose: Higher values (e.g., > 0.8) make the output more creative but also more unpredictable
    and potentially less coherent. Lower values (e.g., < 0.3) make it more deterministic and focused.
    Tuning for Kumora: A value between 0.5-0.8 offers a good balance It's creative enough to
    sound human and not robotic, but conservative enough to avoid generating
    potentially nonsensical or inappropriate content, which is critical for
    a mental wellness application.
    """

     # --- Prompting Technique Parameters ---
    include_examples: bool = True
    """
    Determines whether to include few-shot examples in the system prompt.
    Purpose: This is a powerful technique to "show" the LLM the desired
    response style, tone, and format. Providing 2-3 high-quality
    examples of (User Input -> Ideal Kumora Response), it is a powerful 
    technique (few-shot prompting) to guide the model towards the desired 
    tone, format, and empathetic style, leading to more consistent and 
    high-quality outputs. For Kumora, this is essential.
    """

    use_chain_of_thought: bool = True
    """
    Enables Chain-of-Thought (CoT) prompting, instructing the model to "think step-by-step".
    Purpose: For complex user situations involving multiple conflicting emotions,
    CoT encourages the model to first analyze the user's state internally before 
    constructing the final empathetic response. This can lead to more thoughtful and 
    relevant replies, though it may slightly increase latency.
    Example internal monologue:
    1. Identify the user's primary emotion (e.g., sadness, loneliness).
    2. Validate this feeling without judgment.
    3. Formulate a gentle, open-ended question to encourage reflection.
    This structured thinking leads to more thoughtful and relevant responses.
    """

    # --- Content and Persona Parameters ---
    safety_level: str = "high"
    """
    An application-level setting to enforce safety guardrails.
    Purpose: This is not a direct LLM parameter but a flag for your own logic.
    This is a critical and responsible setting for an app like Kumora.
    A 'high' setting should trigger the strictest rules:
    - Using a system prompt with very strong prohibitions (no medical advice, etc.).
    - Filtering RAG results to only include professionally vetted content.
    - Potentially running the final response through a safety-checking module.
    For Kumora, this should always be 'high' to block potentially harmful,
    unethical, or inappropriate content from being generated.
    """

    response_style: ResponseStyle = ResponseStyle.WARM
    """
    Specifies the desired persona for the response, using the ResponseStyle enum.
    Purpose: This allows for programmatic control over Kumora's "voice". While WARM 
    is the default, you could potentially adapt the style based on the user's 
    long-term preferences or specific conversational contexts in the future.
    This Enum value can be directly injected into the system prompt,
    e.g., "You are Kumora. Your response style is {response_style.value}."
    This makes the persona an explicit, configurable part of the prompt.
    """

    empathy_level: EmpathyLevel = EmpathyLevel.HIGH
    """
    Determines the depth of emotional validation in the response, using the EmpathyLevel enum.
    Purpose: This directly controls how explicitly Kumora should engage with emotions.
    Similar to `response_style`, this guides the depth of emotional reflection in the response. 
    A 'HIGH' level instructs the model to prioritize active listening techniques like paraphrasing 
    and emotional validation rather than just surface-level acknowledgement.
    """
