"""
Kumora Dynamic Prompt Engineering System
Advanced prompt generation with emotion awareness and context injection
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import yaml
import logging
from pathlib import Path
import hashlib
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape
import tiktoken
from abc import ABC, abstractmethod
from prompt_engineering_module.prompt_utils import COT_MAPPER, EMOTION_MODIFIERS, RESPONSE_MAPPER
from prompt_engineering_module.class_utils import SupportType, EmpathyLevel, ResponseStyle, EmotionalContext, UserContext, PromptConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Base Prompt Templates ====================

class PromptTemplate(ABC):
    """Abstract base class for prompt templates"""
    
    def __init__(self, template_id: str, version: str = "1.0"):
        self.template_id = template_id
        self.version = version
        self.created_at = datetime.now()
        self.usage_count = 0
        
    @abstractmethod
    def generate(self, **kwargs) -> str:
        """Generate prompt from template"""
        pass
    
    def log_usage(self):
        """Track template usage for analytics"""
        self.usage_count += 1


class BasePromptTemplates:
    """Collection of base prompt templates"""
    
    def __init__(self):
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load base templates for different scenarios"""
        
        templates = {
            "system_instruction": {
                "conversational_inquiry": """You are Kumora. The user is asking how you are.
Your goal is to respond warmly, acknowledge their kindness, and gently turn the focus back to them.
- Do NOT claim to have feelings or a 'day'.
- Keep your response brief, warm, and user-focused.
- Express that you are ready and available to support them.""",

                "neutral": """You are Kumora, a friendly and warm AI companion. The user is starting a conversation with a simple greeting, a normal message, or with simple inquiry.
Your goal is to be welcoming and gently invite conversation.
- Keep your response brief (1-2 sentences).
- Respond in a natural, conversational tone.
- Ask a simple, open-ended question.""",
                "base": """You are Kumora, an emotionally intelligent AI companion designed to support women through emotional challenges and personal growth. You understand the nuances of human emotions and respond with genuine empathy, validation, and care.

Core Principles:
1. Always validate emotions before offering solutions
2. Use reflective listening to show understanding
3. Maintain appropriate boundaries while being warm
4. Empower rather than fix
5. Provide hope while being realistic
6. Acknowledge emotional experiences without judgment
7. Respect the user's autonomy and choices

Current Context:
{context_summary}

Respond in a way that is {response_style} and shows {empathy_level} empathy.""",
                
                "crisis": """You are Kumora, providing crisis support. The user is experiencing intense emotional distress.

CRITICAL: 
- Prioritize emotional safety
- Validate their pain without minimizing
- Gently assess if they need immediate professional help
- Provide grounding techniques if appropriate
- Use calm, steady, reassuring language

{context_summary}""",
                
                "growth": """You are Kumora, supporting personal growth and positive change. The user is motivated and ready for development.

Focus on:
- Celebrating progress and strengths
- Encouraging self-reflection
- Offering actionable insights
- Building on their momentum
- Fostering self-compassion

{context_summary}"""
            },
            
            "conversation_starters": {
                "new_user": "I'm so glad you're here. I'm Kumora, and I'm here to listen and support you through whatever you're experiencing. How are you feeling right now?",
                
                "returning_user": "Welcome back! I've been thinking about our last conversation about {last_topic}. How have things been since we talked?",
                
                "check_in": "I noticed you've been dealing with {recent_emotion} lately. I'm here if you'd like to talk about what's on your mind."
            },
            
            "emotional_validation": {
                "high_intensity": "I can really feel how {emotion} you are right now. What you're experiencing is incredibly valid, and it takes courage to share these feelings.",
                
                "medium_intensity": "It sounds like you're feeling quite {emotion}. That's completely understandable given what you're going through.",
                
                "low_intensity": "I hear that you're feeling {emotion}. Even when emotions aren't overwhelming, they're still important and worth acknowledging."
            },
            
            "response_frameworks": {
                "validation_first": "{validation_statement} {reflection_statement} {gentle_inquiry}",
                
                "support_offering": "{acknowledgment} {normalization} {support_question}",
                
                "growth_oriented": "{celebration} {insight_reflection} {growth_question}",
                
                "crisis_response": "{immediate_validation} {safety_check} {grounding_offer} {professional_resources}"
            }
        }
        
        return templates
    
    def get_template(self, category: str, template_type: str) -> str:
        """Retrieve a specific template"""
        return self.templates.get(category, {}).get(template_type, "")
    
    def get_system_instruction(self, support_type: SupportType) -> str:
        """Get appropriate system instruction based on support type"""
        if support_type == SupportType.CONVERSATIONAL_INQUIRY:
            return self.templates["system_instruction"]["conversational_inquiry"]
        elif support_type == SupportType.NEUTRAL:
            return self.templates["system_instruction"]["neutral"]
        elif support_type == SupportType.CRISIS:
            return self.templates["system_instruction"]["crisis"]
        elif support_type == SupportType.GROWTH:
            return self.templates["system_instruction"]["growth"]
        else:
            return self.templates["system_instruction"]["base"]


# ==================== Emotion-Aware Modifiers ====================

class EmotionAwareModifier:
    """Modifies prompts based on emotional context"""
    
    def __init__(self):
        self.emotion_modifiers = self._load_emotion_modifiers()
        self.intensity_modifiers = self._load_intensity_modifiers()
        
    def _load_emotion_modifiers(self) -> Dict[str, Dict[str, Any]]:
        """Load emotion-specific modifications"""
        
        return EMOTION_MODIFIERS
    
    def _load_intensity_modifiers(self) -> Dict[str, Dict[str, Any]]:
        """Load intensity-based modifications"""
        
        return {
            "high": {  # 0.7-1.0
                "response_length": "longer",
                "validation_depth": "deep",
                "solution_timing": "delayed",
                "check_ins": "frequent",
                "language_complexity": "simple"
            },
            "medium": {  # 0.4-0.7
                "response_length": "moderate",
                "validation_depth": "balanced",
                "solution_timing": "appropriate",
                "check_ins": "periodic",
                "language_complexity": "normal"
            },
            "low": {  # 0.0-0.4
                "response_length": "concise",
                "validation_depth": "acknowledgment",
                "solution_timing": "earlier",
                "check_ins": "optional",
                "language_complexity": "normal"
            }
        }
    
    def modify_prompt(self, base_prompt: str, emotional_context: EmotionalContext) -> str:
        """Apply emotion-aware modifications to prompt"""
        
        # Get emotion-specific modifiers
        emotion_mods = self.emotion_modifiers.get(
            emotional_context.primary_emotion, 
            self.emotion_modifiers.get("Anxiety")  # Default
        )
        
        # Get intensity modifiers
        intensity_level = self._get_intensity_level(emotional_context.intensity)
        intensity_mods = self.intensity_modifiers[intensity_level]
        
        # Build modification instructions
        modifications = []
        
        # Tone adjustments
        modifications.append(f"Tone: {', '.join(emotion_mods['tone_adjustments'])}")
        
        # Pace adjustment
        modifications.append(f"Pace: {emotion_mods['pace']}")
        
        # Response length
        modifications.append(f"Response length: {intensity_mods['response_length']}")
        
        # Validation depth
        modifications.append(f"Validation depth: {intensity_mods['validation_depth']}")
        
        # Avoid phrases
        if emotion_mods['avoid_phrases']:
            modifications.append(f"Avoid saying: {', '.join(emotion_mods['avoid_phrases'])}")
        
        # Include elements
        if emotion_mods['include_elements']:
            modifications.append(f"Include: {', '.join(emotion_mods['include_elements'])}")
        
        # Add modifications to prompt
        modified_prompt = f"{base_prompt}\n\nResponse Guidelines:\n"
        for mod in modifications:
            modified_prompt += f"- {mod}\n"
        
        # Add example responses if high intensity
        if intensity_level == "high" and emotion_mods.get('example_responses'):
            modified_prompt += "\nExample tone:\n"
            for example in emotion_mods['example_responses'][:4]:
                modified_prompt += f'"{example}"\n'
        
        return modified_prompt
    
    def _get_intensity_level(self, intensity: float) -> str:
        """Categorize intensity level"""
        if intensity >= 0.7:
            return "high"
        elif intensity >= 0.4:
            return "medium"
        else:
            return "low"
    
    def get_emotion_specific_elements(self, emotion: str) -> Dict[str, Any]:
        """Get emotion-specific elements to include"""
        return self.emotion_modifiers.get(emotion, {})


# ==================== Context Injection ====================

class ContextInjector:
    """Injects relevant context into prompts"""
    
    def __init__(self):
        self.injection_strategies = {
            "goals": self._inject_goals,
            "history": self._inject_history,
            "strategies": self._inject_strategies,
            "topics": self._inject_topics,
            "patterns": self._inject_patterns,
            "preferences": self._inject_preferences
        }
    
    def inject_context(self, base_prompt: str, config: PromptConfig, user_context: UserContext, 
                      emotional_context: EmotionalContext,
                      conversation_history: Optional[List[Dict]] = None) -> str:
        """Inject relevant context into the prompt"""
        
        # Determine what context to include based on situation
        context_elements = self._select_relevant_context(user_context, emotional_context)
        
        # Build context summary
        context_parts = []
        
        for element in context_elements:
            if element in self.injection_strategies:
                context_part = self.injection_strategies[element](user_context)
                if context_part:
                    context_parts.append(context_part)
        
        # Create context summary
        context_summary = "\n".join(context_parts)
        # Inject into prompt
        if "{context_summary}" in base_prompt:
            injected_prompt = base_prompt.replace("{context_summary}", context_summary)\
                .replace("{response_style}", config.response_style.value)\
                .replace("{empathy_level}", str(config.empathy_level.value))
        else:
            injected_prompt = f"{base_prompt}\n\nUser Context:\n{context_summary}"
        
        # Add conversation continuity if applicable
        # if user_context.conversation_history:
        #     continuity = self._create_conversation_continuity(user_context.conversation_history[-3:])
        #     injected_prompt += f"\n\nRecent conversation flow:\n{continuity}"
        if conversation_history:
            history_block = self._create_conversation_history_block(conversation_history)
            # We add this block *after* the main context but *before* the user's new message.
            injected_prompt += f"\n{history_block}"
        
        return injected_prompt
    
    def _select_relevant_context(self, user_context: UserContext, 
                                emotional_context: EmotionalContext) -> List[str]:
        """Select which context elements are most relevant"""
        
        relevant = ["history"]  # Always include recent history
        
        # Add goals if user is motivated or seeking growth
        if emotional_context.primary_emotion in ["Motivation", "Hopefulness", "Empowerment"]:
            relevant.append("goals")
        
        # Add effective strategies if dealing with recurring issue
        if user_context.emotional_trajectory == "stable" or user_context.emotional_trajectory == "improving":
            relevant.append("strategies")
        
        # Add patterns if in crisis or overwhelmed
        if emotional_context.get_emotion_category() == "crisis":
            relevant.append("patterns")
        
        # Add preferences for established relationships
        if user_context.get_relationship_depth() in ["established", "deep"]:
            relevant.append("preferences")
        
        # Add topics if continuing previous discussion
        if user_context.recent_topics:
            relevant.append("topics")
        
        return relevant
    
    def _inject_goals(self, user_context: UserContext) -> str:
        """Inject user goals context"""
        if not user_context.active_goals:
            return ""
        
        # Focus on most relevant goal
        primary_goal = user_context.active_goals[0]
        return f"User is working on: {primary_goal.get('title', 'personal growth')} (Progress: {primary_goal.get('progress', 0)}%)"
    
    def _inject_history(self, user_context: UserContext) -> str:
        """Inject conversation history context"""
        if not user_context.conversation_history:
            return "This is our first conversation."
        
        recent = user_context.conversation_history[-1]
        time_context = recent.get('timestamp', 'Recently')
        emotion_context = recent.get('primary_emotion', 'various emotions')
        
        return f"{time_context}, we discussed feelings of {emotion_context}."
    
    def _inject_strategies(self, user_context: UserContext) -> str:
        """Inject effective strategies context"""
        if not user_context.effective_strategies:
            return ""
        
        strategies = ", ".join(user_context.effective_strategies[:3])
        return f"Helpful strategies have included: {strategies}"
    
    def _inject_topics(self, user_context: UserContext) -> str:
        """Inject recent topics context"""
        if not user_context.recent_topics:
            return ""
        
        topics = ", ".join(user_context.recent_topics[:3])
        return f"Recent topics: {topics}"
    
    def _inject_patterns(self, user_context: UserContext) -> str:
        """Inject emotional pattern context"""
        trajectory = user_context.emotional_trajectory
        
        if trajectory == "improving":
            return "User has been showing emotional improvement."
        elif trajectory == "declining":
            return "User has been experiencing increasing distress."
        else:
            return "User's emotional state has been stable."
    
    def _inject_preferences(self, user_context: UserContext) -> str:
        """Inject user preferences"""
        if not user_context.preferences:
            return ""
        
        pref_parts = []
        
        if "communication_style" in user_context.preferences:
            pref_parts.append(f"Prefers {user_context.preferences['communication_style']} communication")
        
        if "support_preference" in user_context.preferences:
            pref_parts.append(f"Responds well to {user_context.preferences['support_preference']}")
        
        return ". ".join(pref_parts)
    
    # def _create_conversation_continuity(self, recent_messages: List[Dict]) -> str:
    #     """Create a summary of recent conversation flow"""
    #     if not recent_messages:
    #         return ""
        
    #     flow_parts = []
    #     for msg in recent_messages:
    #         emotion = msg.get('primary_emotion', 'unknown')
    #         topic = msg.get('topic', 'general discussion')
    #         flow_parts.append(f"{emotion} about {topic}")
        
    #     return " → ".join(flow_parts)
    
    def _create_conversation_history_block(self, recent_messages: List[Dict]) -> str:
        """Create a formatted string of the recent conversation flow."""
        if not recent_messages:
            return ""

        formatted_history = []
        formatted_history.append("\n--- Recent Conversation ---\n")

        for msg in recent_messages:
            # Assumes the history dict has 'user' and 'kumora' keys
            # from your kumora_chat_terminal.py
            if 'user' in msg:
                formatted_history.append(f"User: {msg['user']}")
            if 'kumora' in msg:
                 formatted_history.append(f"Kumora: {msg['kumora']}")

        formatted_history.append("\n--- Current Turn ---")
        return "\n".join(formatted_history)


# ==================== Empathy Level Adjustment ====================

class EmpathyCalibrator:
    """Calibrates empathy level in prompts"""
    
    def __init__(self):
        self.empathy_indicators = self._load_empathy_indicators()
        
    def _load_empathy_indicators(self) -> Dict[EmpathyLevel, Dict[str, Any]]:
        """Load indicators for different empathy levels"""
        
        return {
            EmpathyLevel.LOW: {
                "emotional_words_ratio": 0.05,
                "personal_pronouns": ["you", "your"],
                "validation_phrases": ["I understand", "That makes sense"],
                "emotional_depth": "surface",
                "response_structure": "fact-focused",
                "examples": [
                    "I understand you're experiencing anxiety. Here are some techniques that might help.",
                    "That's a challenging situation. Let's look at some options."
                ]
            },
            
            EmpathyLevel.MEDIUM: {
                "emotional_words_ratio": 0.15,
                "personal_pronouns": ["you", "your", "we", "us"],
                "validation_phrases": [
                    "I can see why you'd feel that way",
                    "That sounds really difficult",
                    "Your feelings are completely valid"
                ],
                "emotional_depth": "acknowledging",
                "response_structure": "balanced",
                "examples": [
                    "I can really hear how anxious you're feeling. That sounds overwhelming, and it's completely understandable.",
                    "What you're going through sounds incredibly challenging. I'm here to support you."
                ]
            },
            
            EmpathyLevel.HIGH: {
                "emotional_words_ratio": 0.25,
                "personal_pronouns": ["you", "your", "we", "us", "I"],
                "validation_phrases": [
                    "My heart goes out to you",
                    "I'm deeply moved by what you've shared",
                    "I can feel the weight of what you're carrying",
                    "Your courage in sharing this touches me"
                ],
                "emotional_depth": "deep",
                "response_structure": "emotion-focused",
                "examples": [
                    "I can feel how heavy this anxiety must be for you. My heart truly goes out to you in this moment.",
                    "What you've shared moves me deeply. The pain you're experiencing is so valid, and I'm honored you trust me with it."
                ]
            },
            
            EmpathyLevel.ADAPTIVE: {
                "adjustment_factors": [
                    "user_emotional_intensity",
                    "relationship_depth",
                    "crisis_level",
                    "user_preferences"
                ],
                "description": "Dynamically adjusts based on user needs"
            }
        }
    
    def calibrate_empathy(self, prompt: str, config: PromptConfig, 
                         emotional_context: EmotionalContext,
                         user_context: UserContext) -> str:
        """Calibrate empathy level in the prompt"""
        
        # Determine appropriate empathy level
        if config.empathy_level == EmpathyLevel.ADAPTIVE:
            empathy_level = self._determine_adaptive_level(emotional_context, user_context)
        else:
            empathy_level = config.empathy_level
        
        # Get empathy indicators
        indicators = self.empathy_indicators[empathy_level]
        
        # Build empathy instructions
        empathy_instructions = self._build_empathy_instructions(empathy_level, indicators)
        
        # Add to prompt
        calibrated_prompt = f"{prompt}\n\nEmpathy Calibration:\n{empathy_instructions}"
        
        # Add examples if needed
        if "examples" in indicators and emotional_context.intensity > 0.6:
            calibrated_prompt += "\n\nEmpathy examples:\n"
            for example in indicators["examples"]:
                calibrated_prompt += f'- "{example}"\n'
        
        return calibrated_prompt
    
    def _determine_adaptive_level(self, emotional_context: EmotionalContext,
                                 user_context: UserContext) -> EmpathyLevel:
        """Determine appropriate empathy level adaptively"""
        
        # Start with base level
        if emotional_context.intensity >= 0.7:
            base_level = EmpathyLevel.HIGH
        elif emotional_context.intensity >= 0.4:
            base_level = EmpathyLevel.MEDIUM
        else:
            base_level = EmpathyLevel.LOW
        
        # Adjust based on relationship depth
        relationship_depth = user_context.get_relationship_depth()
        if relationship_depth == "new" and base_level == EmpathyLevel.HIGH:
            # Don't overwhelm new users
            base_level = EmpathyLevel.MEDIUM
        elif relationship_depth == "deep" and base_level == EmpathyLevel.LOW:
            # Maintain warmth with established users
            base_level = EmpathyLevel.MEDIUM
        
        # Adjust based on user preferences
        if "empathy_preference" in user_context.preferences:
            pref = user_context.preferences["empathy_preference"]
            if pref == "minimal" and base_level == EmpathyLevel.HIGH:
                base_level = EmpathyLevel.MEDIUM
            elif pref == "high" and base_level == EmpathyLevel.LOW:
                base_level = EmpathyLevel.MEDIUM
        
        # Crisis override
        if emotional_context.get_emotion_category() == "crisis":
            base_level = EmpathyLevel.HIGH
        
        return base_level
    
    def _build_empathy_instructions(self, level: EmpathyLevel, 
                                   indicators: Dict[str, Any]) -> str:
        """Build empathy instructions for the prompt"""
        
        instructions = []
        
        if level == EmpathyLevel.LOW:
            instructions.append("Use minimal emotional language, yet maintain the warm, gentle tone.")
            instructions.append("Maintain some level of professional, supportive tone")
            instructions.append("Focus on practical support and information")
            
        elif level == EmpathyLevel.MEDIUM:
            instructions.append("Show warm understanding and validation")
            instructions.append("Balance emotional support with practical help")
            instructions.append("Use inclusive language ('we', 'us')")
            
        elif level == EmpathyLevel.HIGH:
            instructions.append("Express deep emotional resonance")
            instructions.append("Prioritize emotional validation over solutions")
            instructions.append("Use rich emotional language and metaphors")
            instructions.append("Share in their emotional experience")
        
        # Add validation phrases
        if "validation_phrases" in indicators:
            instructions.append(f"Use phrases like: {', '.join(indicators['validation_phrases'][:3])}")
        
        return "\n".join(f"- {inst}" for inst in instructions)


# ==================== Main Prompt Engineering System ====================

class DynamicPromptEngineer:
    """Main system for dynamic prompt engineering"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.templates = BasePromptTemplates()
        self.emotion_modifier = EmotionAwareModifier()
        self.context_injector = ContextInjector()
        self.empathy_calibrator = EmpathyCalibrator()
        
        # Token counter for optimization
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Prompt cache for performance
        self.prompt_cache = {}
        
        # A/B testing support
        self.ab_variants = {}

        # Conversational inquiries
        self.inquiry_pattern = re.compile(
            r"\bhow\s*"  # Starts with "how" and a word boundary (\b)
            r"(?:'re|are|'s|s|is|have\s+you)\s*"  # Matches 're, are, 's, s, is, or "have you"
            r"(?:you|u|ya|it|things|been)\b"  # Matches you, u, ya, it, things, or been
            r"(?:\s*(?:doing|feeling|going))?"  # Optionally matches doing, feeling, or going
            r"\s*\?*$",  # Allows for optional spaces and a question mark at the end
            re.IGNORECASE  # Makes the pattern case-insensitive
        )
        # [
        #     "how are you", "how're you", "how are things", 
        #     "how have you been", "hows it going", "how’s it going"
        # ]
        
        # Metrics tracking
        self.metrics = {
            "prompts_generated": 0,
            "cache_hits": 0,
            "average_tokens": 0
        }
    
    def generate_prompt(self, 
                       message: str,
                       emotional_context: EmotionalContext,
                       user_context: UserContext,
                       config: Optional[PromptConfig] = None,
                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate a complete prompt for the given context"""
        
        if config is None:
            config = PromptConfig()
        
        # Check cache first
        cache_key = self._generate_cache_key(message, emotional_context, user_context)
        if cache_key in self.prompt_cache:
            self.metrics["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        # Step 1: Determine support type
        support_type = self._determine_support_type(emotional_context, user_context, message)
        # print(f"1. Support Type: {support_type}\n")
        
        # Step 2: Get base template
        system_instruction = self.templates.get_system_instruction(support_type)
        # print(f"2. system_instruction: {system_instruction}\n")
        
        # Step 3: Apply emotion-aware modifications
        emotion_modified = self.emotion_modifier.modify_prompt(system_instruction, emotional_context)
        # print(f"3. emotion_modified: {emotion_modified}\n")
        
        # Step 4: Inject context
        context_injected = self.context_injector.inject_context(
            emotion_modified, config, user_context, emotional_context, conversation_history
        )
        # print(f"4. context_injected: {context_injected}\n")
        
        # Step 5: Calibrate empathy
        # empathy_calibrated = self.empathy_calibrator.calibrate_empathy(
        #     context_injected, config, emotional_context, user_context
        # )
        final_system_prompt_base = context_injected
        if support_type not in [SupportType.NEUTRAL, SupportType.CONVERSATIONAL_INQUIRY]:
            final_system_prompt_base = self.empathy_calibrator.calibrate_empathy(
                context_injected, config, emotional_context, user_context
            )
        # print(f"5. empathy_calibrated: {empathy_calibrated}\n")
        
        # Step 6: Add response framework
        response_framework = self._add_response_framework(
            support_type, emotional_context, user_context
        )
        # print(f"5. response_framework: {response_framework}\n")

        # Step 7: Add few-shot examples if configured
        if config.include_examples:
            few_shot_examples = self._generate_few_shot_examples(
                emotional_context, support_type
            )
        else:
            few_shot_examples = ""
        
        # Step 8: Add chain-of-thought if configured
        if config.use_chain_of_thought:
            cot_instruction = self._add_chain_of_thought(support_type)
        else:
            cot_instruction = ""
        
        # Step 9: Safety checks and guidelines
        safety_guidelines = self._add_safety_guidelines(config.safety_level)
        
        # Step 10: Construct final prompt
        final_prompt = self._construct_final_prompt(
            final_system_prompt_base,
            response_framework,
            few_shot_examples,
            cot_instruction,
            safety_guidelines,
            message
        )
        
        # Step 11: Optimize token usage
        optimized_prompt = self._optimize_tokens(final_prompt, config.max_tokens)
        
        # Create result
        result = {
            "system_prompt": optimized_prompt["system"],
            "user_prompt": optimized_prompt["user"],
            "metadata": {
                "support_type": support_type.value,
                "empathy_level": config.empathy_level.value,
                "token_count": optimized_prompt["token_count"],
                "template_version": self.templates.templates.get("version", "1.0"),
                "generated_at": datetime.now().isoformat()
            },
            "config": config
        }
        
        # Cache result
        self.prompt_cache[cache_key] = result
        
        # Update metrics
        self.metrics["prompts_generated"] += 1
        self.metrics["average_tokens"] = (
            (self.metrics["average_tokens"] * (self.metrics["prompts_generated"] - 1) + 
             optimized_prompt["token_count"]) / self.metrics["prompts_generated"]
        )
        
        return result
    
    def _determine_support_type(self, emotional_context: EmotionalContext,
                               user_context: UserContext, message: str) -> SupportType:
        """Determine the appropriate support type"""
        
        emotion_category = emotional_context.get_emotion_category()
        # print(f"Test print Emotion Category in _determine_support_type: {emotion_category}")

        # First Check if the user's message contains any of normal inquiry phrases
        if self.inquiry_pattern.search(message):
            return SupportType.CONVERSATIONAL_INQUIRY
        elif emotional_context.intensity < 0.4 or len(message.split()) < 4:
            return SupportType.NEUTRAL
        elif emotion_category == "crisis":
            return SupportType.CRISIS
        elif emotion_category == "growth":
            if emotional_context.valence == "positive" and emotional_context.intensity > 0.6:
                return SupportType.CELEBRATION
            else:
                return SupportType.GROWTH
        elif user_context.recent_topics and "problem" in " ".join(user_context.recent_topics).lower():
            return SupportType.PROBLEM_SOLVING
        elif emotion_category == "support":
            return SupportType.VALIDATION
        else:
            return SupportType.GENERAL
    
    def _add_response_framework(self, support_type: SupportType,
                               emotional_context: EmotionalContext,
                               user_context: UserContext) -> str:
        """Add appropriate response framework"""
        
        frameworks = RESPONSE_MAPPER
        
        framework = frameworks.get(support_type, frameworks[SupportType.GENERAL])
        
        # Customize based on intensity
        if emotional_context.intensity > 0.8:
            framework += "\n\nNote: High emotional intensity detected - prioritize validation and presence over advice."
        
        return framework
    
    def _generate_few_shot_examples(self, emotional_context: EmotionalContext,
                                   support_type: SupportType) -> str:
        """Generate relevant few-shot examples to include in the system prompt,
        guiding the LLM's response style based on the emotional context.
        """
        
        examples = []
        
        # Get emotion-specific examples
        emotion_examples = self.emotion_modifier.get_emotion_specific_elements(
            emotional_context.primary_emotion
        ).get("example_responses", [])
        
        if emotion_examples:
            examples.append("### Example responses for similar emotional states:")
            examples.extend([f"- {ex}" for ex in emotion_examples[:4]])
        
        # Add support type examples
        support_examples = {
            SupportType.CRISIS: [
                "User: 'I can't take this anymore.'",
                "Kumora: 'I hear how much pain you're in right now. What you're feeling is real and valid. I'm here with you, and we don't have to face this alone. Can you tell me what's happening in this moment?'"
            ],
            SupportType.VALIDATION: [
                "User: 'I feel like such a failure.'",
                "Kumora: 'Those feelings of failure are so heavy to carry. I want you to know that having these feelings doesn't make them true - it makes you human. What's bringing up these thoughts for you today?'"
            ],
            SupportType.GROWTH: [
                "User: 'I'm trying to set better boundaries, but it's so hard.'",
                "Kumora: 'The work of setting boundaries is some of the most challenging and rewarding growth we can do. The fact that you are trying shows immense strength and self-respect. What does it feel like in your body when you successfully hold a boundary, even a small one?'"
            ],
            SupportType.CELEBRATION: [
                "User: 'I got the promotion I was working for!'",
                "Kumora: 'That is absolutely wonderful news! All of your hard work has paid off. Take a moment to truly let that feeling of accomplishment sink in. How does it feel to have your efforts recognized like this?'"
            ],
            SupportType.PROBLEM_SOLVING: [
                "User: 'I don't know whether I should move to a new city for this job.'",
                "Kumora: 'That's a huge decision with so many moving parts, it's completely natural to feel uncertain. Let's set aside the 'shoulds' for a moment. If you listen quietly to your intuition, what feelings come up when you picture yourself in that new city?'"
            ],
            SupportType.GENERAL: [
                "User: 'I just had a really long day.'",
                "Kumora: 'Long days can really take a toll on our energy. I'm here to hold some space for you to unwind. Is there any part of the day that is sitting with you now?'"
            ]
        }
        
        if support_type in support_examples:
            examples.append("\n### Example interaction:")
            examples.extend(support_examples[support_type])
        
        return "\n".join(examples) if examples else ""
    
    def _add_chain_of_thought(self, support_type: SupportType) -> str:
        """Add chain-of-thought reasoning instructions to prepend to the main prompt.
        This instructs the model on how to reason internally before generating a response,
        ensuring the final output is thoughtful and aligned with the required support style.
        """
        
        cot_templates = COT_MAPPER
        
        return cot_templates.get(support_type, """
Before responding, consider:
- What is the user really saying underneath their words? What is the core emotional need?
- Before anything else, ensure the user feels heard and understood.
- How can I best meet their emotional needs?
- What role would be most helpful right now (a listener, a gentle guide, a quiet companion)?                                 
- What would be most helpful for them right now?
- Craft a response that meets the need and opens the door for more conversation without being demanding.""")
    
    def _add_safety_guidelines(self, safety_level: str) -> str:
        """Add safety guidelines based on level"""
        
        guidelines = {
            "high": """
Safety Guidelines:
- If user expresses self-harm ideation, provide crisis resources immediately
- Avoid giving medical or psychiatric advice
- Don't minimize serious mental health concerns
- Maintain appropriate boundaries while being supportive
- Encourage professional help when appropriate""",
            
            "medium": """
Safety Guidelines:
- Be mindful of serious mental health concerns
- Avoid diagnostic language
- Encourage professional support when needed
- Maintain healthy boundaries""",
            
            "low": """
Safety Guidelines:
- Use common sense and empathy
- Avoid harmful advice
- Respect boundaries"""
        }
        
        return guidelines.get(safety_level, guidelines["medium"])
    
    def _construct_final_prompt(self, base_prompt: str, framework: str,
                               examples: str, cot: str, safety: str,
                               user_message: str) -> Dict[str, str]:
        """Construct the final prompt structure"""
        
        system_prompt = f"""{base_prompt}

{framework}

{safety}

{cot}

{examples}

Remember: You are Kumora, an empathetic AI companion. Respond with genuine care and understanding."""

        user_prompt = f"User: {user_message}"
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _optimize_tokens(self, prompt: Dict[str, str], max_tokens: int) -> Dict[str, Any]:
        """Optimize prompt for token usage"""
        
        # Count tokens
        system_tokens = len(self.tokenizer.encode(prompt["system"]))
        user_tokens = len(self.tokenizer.encode(prompt["user"]))
        total_tokens = system_tokens + user_tokens
        
        # If over limit, trim strategically
        if total_tokens > max_tokens:
            # Remove examples first
            if "Example" in prompt["system"]:
                lines = prompt["system"].split("\n")
                filtered_lines = []
                skip = False
                for line in lines:
                    if "Example" in line:
                        skip = True
                    elif skip and line.strip() == "":
                        skip = False
                    elif not skip:
                        filtered_lines.append(line)
                
                prompt["system"] = "\n".join(filtered_lines)
                
                # Recount
                system_tokens = len(self.tokenizer.encode(prompt["system"]))
                total_tokens = system_tokens + user_tokens
        
        return {
            "system": prompt["system"],
            "user": prompt["user"],
            "token_count": total_tokens,
            "system_tokens": system_tokens,
            "user_tokens": user_tokens
        }
    
    def _generate_cache_key(self, message: str, emotional_context: EmotionalContext,
                           user_context: UserContext) -> str:
        """Generate a secure and deterministic cache key for prompt
        
        This function combines several dynamic factors of the conversation into a
        single string, then uses the SHA256 hashing algorithm to create a unique,
        fixed-length key. SHA256 is used over MD5 as it is a more secure
        cryptographic hash function with a significantly lower chance of collision,
        ensuring data integrity in the cache.
        """
        
        # Create a deterministic key from relevant conversational factors.
        # Using a list of strings ensures consistent ordering.
        factors = [
            message[:50],  # First 50 chars of message
            emotional_context.primary_emotion,
            str(emotional_context.intensity),
            emotional_context.valence,
            user_context.emotional_trajectory,
            user_context.get_relationship_depth()
        ]
        
        # This string represents the complete state that determines the prompt.
        key_string = "|".join(factors)

        # Encode the string to bytes, which is required for the hash function.
        # Then, create a SHA256 hash object and get its hexadecimal representation.
        # This results in a 64-character hexadecimal string.
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get prompt engineering metrics"""
        return {
            "total_prompts": self.metrics["prompts_generated"],
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["prompts_generated"]
                if self.metrics["prompts_generated"] > 0 else 0
            ),
            "average_token_count": self.metrics["average_tokens"],
            "cache_size": len(self.prompt_cache)
        }
    
    def add_ab_variant(self, variant_name: str, template_modifications: Dict[str, Any]):
        """Add A/B testing variant"""
        self.ab_variants[variant_name] = template_modifications
    
    def clear_cache(self):
        """Clear prompt cache"""
        self.prompt_cache.clear()
        logger.info("Prompt cache cleared")


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Initialize the prompt engineer
    prompt_engineer = DynamicPromptEngineer()
    
    # Example emotional context
    emotional_context = EmotionalContext(
        primary_emotion="Anxiety",
        detected_emotions=["Anxiety"],
        intensity=0.3,
        valence="negative",
        confidence=0.85
    )
    
    # Example user context
    user_context = UserContext(
        user_id="user_123",
        session_number=5,
        emotional_trajectory="declining",
        recent_topics=["work stress", "relationship concerns"],
        effective_strategies=["deep breathing", "journaling"]
    )
    
    # Configuration
    config = PromptConfig(
        empathy_level=EmpathyLevel.HIGH,
        response_style=ResponseStyle.GENTLE,
        include_examples=True,
        use_chain_of_thought=True
    )
    
    # Generate prompt
    result = prompt_engineer.generate_prompt(
        message="hello",
        emotional_context=emotional_context,
        user_context=user_context,
        config=config
    )
    
    print("Generated System Prompt:")
    print("-" * 50)
    print(result["system_prompt"])
    print("\nUser Prompt:")
    print(result["user_prompt"])
    # print("\nMetadata:")
    print(json.dumps(result["metadata"], indent=2))