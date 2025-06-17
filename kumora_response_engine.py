"""
Kumora Response Generation Engine
Using Llama 3.2 3B (local) for empathetic responses and GPT-3.5 (API) as fallback
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import openai
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiohttp
import time
import logging
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import psutil
import GPUtil

# Import your existing components
from emotion_intelligence_system.emotion_classifier import *
from context_management.context_management_system import *
from context_management.kumora_context import *
from prompt_engineering_module.prompt_engineering_system import *
from prompt_engineering_module.class_utils import *
from prompt_engineering_module.prompt_utils import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_TOKEN")
login(token=access_token, add_to_git_credential=False)

@dataclass
class ModelConfig:
    """Configuration for model setup"""

    # Llama 3.2 3B configuration
    llama_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    llama_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    llama_torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    llama_max_memory: Dict[int, str] = field(default_factory=lambda: {0: "3GB"})
    llama_load_in_8bit: bool = False  # Set True if memory constrained
    llama_load_in_4bit: bool = False  # Set True for even more memory savings

    # Gemini 2.5 configuration
    gemini_model_name: str = "gemini-1.5-flash-latest"
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # GPT-4.1-mini configuration
    gpt_model: str = "gpt-4.1-mini"
    gpt_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gpt_temperature: float = 0.7
    gpt_max_tokens: int = 200

    # Response generation settings
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # System settings
    response_timeout: float = 30.0
    use_streaming: bool = True
    cache_responses: bool = True
    max_cache_size: int = 1000


@dataclass
class GenerationMetrics:
    """Metrics for response generation"""
    model_used: str
    generation_time: float
    token_count: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool = False
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None


# ==================== Base Response Generator ====================

class BaseResponseGenerator(ABC):
    """Abstract base class for response generators"""

    @abstractmethod
    async def generate(self, prompt: str, config: ModelConfig) -> Tuple[str, GenerationMetrics]:
        """Generate response from prompt"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if generator is healthy"""
        pass

    def extract_response(self, full_text: str, prompt: str) -> str:
        """Extract only the generated response from full text"""
        # Remove prompt from response
        if prompt in full_text:
            response = full_text.replace(prompt, "").strip()
        else:
            response = full_text.strip()

        # Clean up any remaining formatting
        response = response.replace("<|assistant|>", "").strip()
        response = response.replace("</s>", "").strip()

        return response


# ==================== Llama 3.2 Generator ====================

class Llama32Generator(BaseResponseGenerator):
    """Local Llama 3.2 3B generator for empathetic responses"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self._initialized = False

    def initialize(self):
        """Initialize Llama model and tokenizer"""
        if self._initialized:
            return

        logger.info("Initializing Llama 3.2 3B model...")

        try:
            # Check available memory
            if self.config.llama_device == "cuda":
                gpu = GPUtil.getGPUs()[0]
                logger.info(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llama_model_name,
                trust_remote_code=True
            )

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate settings
            if self.config.llama_load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.llama_torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.llama_load_in_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.config.llama_torch_dtype
                )
            else:
                quantization_config = None

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llama_model_name,
                torch_dtype=self.config.llama_torch_dtype,
                device_map="auto" if self.config.llama_device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                max_memory=self.config.llama_max_memory if self.config.llama_device == "cuda" else None
            )

            if self.config.llama_device == "cpu":
                self.model = self.model.to(self.config.llama_device)

            # Set to evaluation mode
            self.model.eval()

            self._initialized = True
            logger.info("Llama 3.2 3B model initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            raise

    async def generate(self, prompt: str, config: ModelConfig) -> Tuple[str, GenerationMetrics]:
        """Generate response using Llama 3.2"""
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        try:
            # Format prompt for Llama 3.2 Instruct
            formatted_prompt = self._format_prompt(prompt)

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt"
                # truncation=True,
                # max_length=2048 - config.max_new_tokens
            ).to(self.config.llama_device)

            # if self.config.llama_device == "cuda":
            #     inputs = {k: v.to(self.config.llama_device) for k, v in inputs.items()}

            prompt_tokens = inputs['input_ids'].shape[1]

             # Use TextIteratorStreamer for true streaming
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Run generation in a separate thread so it doesn't block the event loop
            generation_kwargs = dict(
                inputs,
                streamer=self.streamer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Generate
            # Using asyncio.to_thread for the blocking model call
            await asyncio.to_thread(self.model.generate, **generation_kwargs)

            # with torch.no_grad():
            #     if config.use_streaming:
            #         # Streaming generation
            #         response = await self._generate_streaming(inputs, config)
            #     else:
            #         # Standard generation
            #         outputs = self.model.generate(
            #             **inputs,
            #             max_new_tokens=config.max_new_tokens,
            #             temperature=config.temperature,
            #             top_p=config.top_p,
            #             do_sample=config.do_sample,
            #             repetition_penalty=config.repetition_penalty,
            #             pad_token_id=self.tokenizer.pad_token_id,
            #             eos_token_id=self.tokenizer.eos_token_id
            #         )

            #         # Decode
            #         full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            #         response = self.extract_response(full_response, formatted_prompt)
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

            # Calculate metrics
            completion_tokens = len(self.tokenizer.encode(response))
            generation_time = time.time() - start_time

            metrics = GenerationMetrics(
                model_used="llama-3.2-3b",
                generation_time=generation_time,
                token_count=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )

            return response, metrics

        except Exception as e:
            logger.error(f"Error generating with Llama 3.2: {e}")
            raise

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for Llama 3.2 Instruct"""
        # Llama 3.2 Instruct format
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # async def _generate_streaming(self, inputs: Dict, config: ModelConfig) -> str:
    #     """Generate response with streaming"""
    #     # For now, using standard generation
    #     # Streaming can be implemented with TextIteratorStreamer
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_new_tokens=config.max_new_tokens,
    #         temperature=config.temperature,
    #         top_p=config.top_p,
    #         do_sample=config.do_sample,
    #         repetition_penalty=config.repetition_penalty,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id
    #     )

    #     full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return self.extract_response(full_response, inputs['input_ids'])

    async def health_check(self) -> bool:
        """Check if Llama generator is healthy"""
        try:
            if not self._initialized:
                self.initialize()

            # Simple generation test
            test_prompt = "Hello, how are you?"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            if self.config.llama_device == "cuda":
                inputs = {k: v.to(self.config.llama_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

            return True

        except Exception as e:
            logger.error(f"Llama health check failed: {e}")
            return False
        
# ==================== Gemini Generator ====================

class GeminiGenerator(BaseResponseGenerator):
    """Gemini API generator for fast, high-quality responses"""

    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            generation_config = genai.GenerationConfig(
                max_output_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            self.model = genai.GenerativeModel(
                model_name=self.config.gemini_model_name,
                generation_config=generation_config
            )
            logger.info("Gemini Generator initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Generator: {e}")
            raise

    async def generate(self, prompt: str, config: ModelConfig) -> Tuple[str, GenerationMetrics]:
        """Generate response using the Gemini API"""
        start_time = time.time()
        try:
            # Use asyncio.to_thread to run the synchronous SDK call in a non-blocking way
            response = await asyncio.to_thread(self.model.generate_content, prompt)

            response_text = response.text
            generation_time = time.time() - start_time

            # Calculate token counts
            prompt_tokens = self.model.count_tokens(prompt).total_tokens
            completion_tokens = self.model.count_tokens(response_text).total_tokens

            metrics = GenerationMetrics(
                model_used=self.config.gemini_model_name,
                generation_time=generation_time,
                token_count=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            return response_text, metrics

        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if the Gemini API is accessible"""
        try:
            # A lightweight check
            await asyncio.to_thread(self.model.generate_content, "Hello", generation_config={"max_output_tokens": 5})
            return True
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False


# ==================== GPT-3.5 Fallback Generator ====================

class GPTGenerator(BaseResponseGenerator):
    """GPT-4.1 API generator as fallback"""

    def __init__(self, config: ModelConfig):
        self.config = config
        openai.api_key = config.gpt_api_key

    async def generate(self, prompt: str, config: ModelConfig) -> Tuple[str, GenerationMetrics]:
        """Generate response using GPT-4.1"""
        start_time = time.time()

        try:
            # Call OpenAI API
            response = await self._call_openai_api(prompt, config)

            # Extract response text
            response_text = response.choices[0].message.content

            # Calculate metrics
            generation_time = time.time() - start_time

            metrics = GenerationMetrics(
                model_used=config.gpt_model,
                generation_time=generation_time,
                token_count=response.usage.total_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                fallback_triggered=True
            )

            return response_text, metrics

        except Exception as e:
            logger.error(f"Error generating with GPT-3.5: {e}")
            raise

    async def _call_openai_api(self, prompt: str, config: ModelConfig) -> Dict:
        """Call OpenAI API with retry logic"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model=config.gpt_model,
                    messages=[
                        {"role": "system", "content": "You are a highly empathetic, emotionally intelligent companion. Respond reflectively and with emotional presence."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.gpt_temperature,
                    max_tokens=config.gpt_max_tokens,
                    top_p=config.top_p
                    
                )
                return response
            
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise
            except (openai.APIError, openai.APIConnectionError, openai.Timeout) as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def health_check(self) -> bool:
        """Check if GPT-3.5 API is accessible"""
        try:
            response = openai.chat.completions.create(
                model=self.config.gpt_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"GPT-4.1-mini health check failed: {e}")
            return False


# ==================== Response Cache ====================

class ResponseCache:
    """Simple response cache for performance"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get(self, key: str) -> Optional[str]:
        """Get response from cache"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, key: str, value: str):
        """Set response in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]

        self.cache[key] = value
        self.access_count[key] = 1

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()


# ==================== Main Response Engine ====================

class KumoraResponseEngine:
    """
    Main response generation engine for Kumora
    Integrates emotion classification, prompt engineering, and model generation
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        # Initialize components
        logger.info("Initializing Kumora Response Engine...")

        # Existing components
        self.emotion_classifier = EmotionIntelligenceModule("kumora_emotion_model_final")
        self.context_manager = get_context_manager()  # This will use Redis if available, otherwise in-memory
        self.prompt_engineer = DynamicPromptEngineer()

        # Response generators
        # self.llama_generator = Llama32Generator(self.config)
        self.primary_generator = GeminiGenerator(self.config) # Add Gemini as the primary
        self.gpt_generator = GPTGenerator(self.config)

        # Response cache
        self.cache = ResponseCache(self.config.max_cache_size) if self.config.cache_responses else None

        # Metrics
        self.total_requests = 0
        # self.llama_success = 0
        self.gemini_success = 0
        self.fallback_count = 0

        logger.info("Kumora Response Engine initialized!")

    async def generate_response(self,
                              user_message: str,
                              user_id: str,
                              session_id: str,
                              use_fallback: bool = False,
                              conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate empathetic response for user message

        Args:
            user_message: User's input message
            user_id: User identifier
            session_id: Session identifier
            use_fallback: Force use of fallback model

        Returns:
            Dict containing response and metadata
        """
        self.total_requests += 1
        start_time = time.time()

        try:
            # Step 1: Analyze emotions
            logger.info("Analyzing emotions...")
            emotion_analysis = self.emotion_classifier.analyze_emotions(user_message)

            # Step 2: Get user context
            logger.info("Retrieving user context...")
            user_context = self._get_user_context(user_id, session_id)

            # Step 3: Create emotional context
            emotional_context = EmotionalContext(
                primary_emotion=emotion_analysis['primary_emotion'],
                detected_emotions=emotion_analysis['detected_emotions'],
                intensity=emotion_analysis['emotional_intensity'],
                valence=emotion_analysis['emotional_valence'],
                confidence=emotion_analysis.get('confidence', 0.8)
            )

            # Step 4: Determine prompt configuration
            prompt_config = self._determine_prompt_config(emotional_context, user_context)

            # Step 5: Generate dynamic prompt
            logger.info("Generating dynamic prompt...")
            prompt_result = self.prompt_engineer.generate_prompt(
                message=user_message,
                emotional_context=emotional_context,
                user_context=user_context,
                config=prompt_config,
                conversation_history=conversation_history 
            )

            # Step 6: Check cache
            if self.cache and not use_fallback:
                cache_key = self._generate_cache_key(user_message, emotional_context)
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    logger.info("Cache hit!")
                    return {
                        'response': cached_response,
                        'metadata': {
                            'cached': True,
                            'emotion_analysis': emotion_analysis,
                            'generation_time': 0.0
                        }
                    }

            # Step 7: Generate response
            response_text, metrics = await self._generate_with_model(
                prompt_result['system_prompt'],
                use_fallback
            )

            # Step 8: Post-process response
            final_response = self._post_process_response(
                response_text,
                emotional_context,
                user_context
            )

            # Step 9: Update context
            self._update_context(user_id, session_id, emotional_context, final_response)

            # Step 10: Cache if successful
            if self.cache and not use_fallback and metrics.model_used == "llama-3.2-3b":
                self.cache.set(cache_key, final_response)

            # Prepare result
            total_time = time.time() - start_time

            return {
                'response': final_response,
                'metadata': {
                    'emotion_analysis': emotion_analysis,
                    'support_type': prompt_result['metadata']['support_type'],
                    'model_used': metrics.model_used,
                    'generation_time': metrics.generation_time,
                    'total_time': total_time,
                    'cached': False,
                    'fallback_triggered': metrics.fallback_triggered,
                    'prompt_tokens': metrics.prompt_tokens,
                    'completion_tokens': metrics.completion_tokens
                }
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")

            # Emergency fallback
            return {
                'response': "I hear you, and I want to help. Could you tell me a bit more about what you're experiencing?",
                'metadata': {
                    'error': str(e),
                    'fallback': 'emergency'
                }
            }

    async def _generate_with_model(self, prompt: str, use_fallback: bool) -> Tuple[str, GenerationMetrics]:
        """Generate response using appropriate model"""

        # Try Llama first (unless fallback requested)
        # if not use_fallback:
        #     try:
        #         logger.info("Generating with Llama 3.2...")
        #         response, metrics = await asyncio.wait_for(
        #             self.llama_generator.generate(prompt, self.config),
        #             timeout=self.config.response_timeout
        #         )
        #         self.llama_success += 1
        #         return response, metrics

        #     except asyncio.TimeoutError:
        #         logger.warning("Llama generation timed out, using fallback...")
        #         metrics_fallback_reason = "timeout"
        #     except Exception as e:
        #         logger.warning(f"Llama generation failed: {e}, using fallback...")
        #         metrics_fallback_reason = str(e)
        # else:
        #     metrics_fallback_reason = "requested"
        # Try Gemini first (unless fallback requested)
        if not use_fallback:
            try:
                logger.info("Generating with Gemini...") # Updated log
                response, metrics = await asyncio.wait_for(
                    self.primary_generator.generate(prompt, self.config), # Use primary_generator
                    timeout=self.config.response_timeout
                )
                self.gemini_success += 1 # Use renamed metric
                return response, metrics

            except asyncio.TimeoutError:
                logger.warning("Gemini generation timed out, using fallback...")
                metrics_fallback_reason = "timeout"
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}, using fallback...")
                metrics_fallback_reason = str(e)
        else:
            metrics_fallback_reason = "requested"

        # Use GPT-4.1 fallback
        self.fallback_count += 1
        logger.info("Generating with GPT-4.1-mini fallback...")

        try:
            response, metrics = await self.gpt_generator.generate(prompt, self.config)
            metrics.fallback_reason = metrics_fallback_reason
            return response, metrics
        except Exception as e:
            logger.error(f"Fallback generation also failed: {e}")
            raise

    def _get_user_context(self, user_id: str, session_id: str) -> UserContext:
        """Get or create user context"""
        try:
            # Get comprehensive context from context manager
            context = self.context_manager.get_comprehensive_context(user_id, session_id)

            # Convert to UserContext object
            return UserContext(
                user_id=user_id,
                active_goals=context.get('active_goals', []),
                recent_topics=[topic['topic'] for topic in context.get('session', {}).get('topics', [])],
                emotional_trajectory=context.get('emotional_trajectory', 'stable'),
                effective_strategies=[s['name'] for s in context.get('recommended_coping', [])],
                preferences=context.get('personalization', {}).get('preferences', {}),
                session_number=context.get('user_profile', {}).get('total_sessions', 1)
            )
        except:
            # Return minimal context if error
            return UserContext(user_id=user_id)

    def _determine_prompt_config(self, emotional_context: EmotionalContext,
                               user_context: UserContext) -> PromptConfig:
        """
        Determine optimal prompt configuration based on a holistic view of the
        emotional and user context.
        """
        # Base configuration
        config = PromptConfig()
        
        # 1. Determine Empathy Level
        # This logic considers intensity, but also the user relationship depth.
        relationship_depth = user_context.get_relationship_depth()
        if emotional_context.intensity > 0.8:
            config.empathy_level = EmpathyLevel.HIGH
        elif emotional_context.intensity > 0.5 and relationship_depth != "new":
            config.empathy_level = EmpathyLevel.MEDIUM
        else:
            # For new users or low intensity, let the calibrator decide.
            config.empathy_level = EmpathyLevel.ADAPTIVE
            
        # Override with user preference if it exists
        if user_context.preferences.get("empathy_preference"):
            pref = user_context.preferences["empathy_preference"]
            if pref == "high": config.empathy_level = EmpathyLevel.HIGH
            elif pref == "medium": config.empathy_level = EmpathyLevel.MEDIUM
            elif pref == "low": config.empathy_level = EmpathyLevel.LOW

        # 2. Determine Response Style
        # This logic is now more dynamic, pulling from emotion modifiers and trajectory.
        primary_emotion = emotional_context.primary_emotion
        emotion_mods = EMOTION_MODIFIERS.get(primary_emotion, {})
        tone_adjustments = emotion_mods.get('tone_adjustments', [])

        if 'encouraging' in tone_adjustments or 'upbeat' in tone_adjustments:
            config.response_style = ResponseStyle.ENCOURAGING
        elif 'calm' in tone_adjustments or 'reassuring' in tone_adjustments or 'grounding' in tone_adjustments:
            config.response_style = ResponseStyle.GENTLE
        elif user_context.emotional_trajectory == 'declining':
            config.response_style = ResponseStyle.GENTLE
        elif user_context.emotional_trajectory == 'improving':
            config.response_style = ResponseStyle.ENCOURAGING
        else:
            # Default to a warm and inviting style.
            config.response_style = ResponseStyle.WARM

        # 3. Determine when to use Chain-of-Thought (CoT)
        # CoT is used for complex, sensitive, or critical situations.
        use_cot = any([
            emotional_context.get_emotion_category() == "crisis",
            emotional_context.intensity > 0.8, # Very high intensity requires careful thought
            user_context.emotional_trajectory == 'declining', # User is struggling, needs thoughtful response
            len(emotional_context.detected_emotions) > 3, # Emotionally complex situation
        ])
        config.use_chain_of_thought = use_cot
        
        # 4. Determine when to include few-shot examples
        # Examples help guide the model in nuanced or high-stakes interactions.
        include_ex = any([
            emotional_context.intensity > 0.7,
            emotional_context.get_emotion_category() in ["crisis", "growth"],
            relationship_depth == "deep" # Guide the model on personalized tone for established users
        ])
        config.include_examples = include_ex
        
        return config

    def _post_process_response(self, response: str,
                             emotional_context: EmotionalContext,
                             user_context: UserContext) -> str:
        """Post-process generated response"""

        # Ensure response ends properly
        if response and not response[-1] in '.!?':
            response += '.'

        # Add user name if preferred and appropriate
        if user_context.preferences.get('use_name') and user_context.preferences.get('name'):
            name = user_context.preferences['name']
            # Add name at beginning for high-intensity emotions
            if emotional_context.intensity > 0.7 and not name.lower() in response.lower():
                response = f"{name}, {response[0].lower()}{response[1:]}"

        # Ensure minimum length for validation
        if len(response.split()) < 20 and emotional_context.get_emotion_category() != "general":
            response += " I'm here to listen and support you through this."

        return response

    def _update_context(self, user_id: str, session_id: str,
                       emotional_context: EmotionalContext, response: str):
        """Update context with interaction"""
        try:
            # Update emotional state in context
            emotional_state = EmotionalState(
                primary_emotion=emotional_context.primary_emotion,
                detected_emotions=emotional_context.detected_emotions,
                intensity=emotional_context.intensity,
                valence=emotional_context.valence,
                confidence=emotional_context.confidence
            )

            self.context_manager.session.update_emotional_state(session_id, emotional_state)

        except Exception as e:
            logger.warning(f"Failed to update context: {e}")

    def _generate_cache_key(self, message: str, emotional_context: EmotionalContext) -> str:
        """Generate cache key for response"""
        import hashlib

        key_components = [
            message[:100],  # First 100 chars
            emotional_context.primary_emotion,
            str(emotional_context.intensity),
            emotional_context.valence
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        health_status = {
            'status': 'healthy',
            'components': {},
            'metrics': {
                'total_requests': self.total_requests,
                # 'llama_success_rate': self.llama_success / max(self.total_requests, 1),
                'gemini_success_rate': self.gemini_success / max(self.total_requests, 1),
                'fallback_rate': self.fallback_count / max(self.total_requests, 1)
            }
        }

        # Check Llama
        # try:
        #     llama_healthy = await self.llama_generator.health_check()
        #     health_status['components']['llama_3.2'] = 'healthy' if llama_healthy else 'unhealthy'
        # except Exception as e:
        #     health_status['components']['llama_3.2'] = f'error: {str(e)}'
        #     health_status['status'] = 'degraded'

        # Check Gemini
        try:
            primary_healthy = await self.primary_generator.health_check()
            health_status['components']['gemini'] = 'healthy' if primary_healthy else 'unhealthy'
            if not primary_healthy:
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['gemini'] = f'error: {str(e)}'
            health_status['status'] = 'degraded'

        # Check GPT-4.1-mini
        try:
            gpt_healthy = await self.gpt_generator.health_check()
            health_status['components']['gpt_3.5'] = 'healthy' if gpt_healthy else 'unhealthy'
        except Exception as e:
            health_status['components']['gpt_3.5'] = f'error: {str(e)}'
            if health_status['status'] == 'degraded':
                health_status['status'] = 'unhealthy'

        return health_status


# ==================== Standalone Functions ====================

async def initialize_kumora_engine(config: Optional[ModelConfig] = None) -> KumoraResponseEngine:
    """Initialize Kumora response engine with all components"""

    if config is None:
        # Default configuration
        config = ModelConfig()

        # Adjust based on available resources
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            available_memory = gpu.memoryTotal - gpu.memoryUsed

            if available_memory < 4000:  # Less than 4GB available
                logger.info("Limited GPU memory detected, using 4-bit quantization")
                config.llama_load_in_4bit = True
            elif available_memory < 6000:  # Less than 6GB available
                logger.info("Moderate GPU memory detected, using 8-bit quantization")
                config.llama_load_in_8bit = True

    engine = KumoraResponseEngine(config)

    # Pre-initialize Llama model
    # logger.info("Pre-initializing Llama model...")
    # engine.llama_generator.initialize()

    # Run health check
    health = await engine.health_check()
    logger.info(f"Engine health: {json.dumps(health, indent=2)}")

    return engine


# ==================== Usage Example ====================

# if __name__ == "__main__":
#     """Example usage of Kumora Response Engine"""

#     # Initialize engine
#     engine = await initialize_kumora_engine()

#     # Test messages with different emotions
#     test_messages = [
#         {
#             'message': "I'm feeling really anxious about my presentation tomorrow. I can't stop thinking about all the ways it could go wrong.",
#             'user_id': 'test_user_001',
#             'session_id': 'test_session_001'
#         },
#         {
#             'message': "I finally got the promotion I've been working towards for years! I can't believe it actually happened!",
#             'user_id': 'test_user_001',
#             'session_id': 'test_session_001'
#         },
#         {
#             'message': "I feel so alone. Nobody understands what I'm going through.",
#             'user_id': 'test_user_002',
#             'session_id': 'test_session_002'
#         }
#     ]

#     for test in test_messages:
#         print(f"\n{'='*60}")
#         print(f"User: {test['message']}")
#         print(f"{'='*60}")

#         # Generate response
#         result = await engine.generate_response(
#             user_message=test['message'],
#             user_id=test['user_id'],
#             session_id=test['session_id']
#         )

#         print(f"\nKumora: {result['response']}")

#         # Print metadata
#         metadata = result['metadata']
#         print(f"\nMetadata:")
#         print(f"- Primary Emotion: {metadata['emotion_analysis']['primary_emotion']}")
#         print(f"- Emotional Intensity: {metadata['emotion_analysis']['emotional_intensity']:.2f}")
#         print(f"- Support Type: {metadata.get('support_type', 'unknown')}")
#         print(f"- Model Used: {metadata.get('model_used', 'unknown')}")
#         print(f"- Generation Time: {metadata.get('generation_time', 0):.2f}s")
#         print(f"- Total Time: {metadata.get('total_time', 0):.2f}s")

#     # Test fallback
#     print(f"\n{'='*60}")
#     print("Testing GPT-4.1-mini Fallback...")
#     print(f"{'='*60}")

#     fallback_result = await engine.generate_response(
#         user_message="I'm worried about my health.",
#         user_id='test_user_003',
#         session_id='test_session_003',
#         use_fallback=True
#     )

#     print(f"\nKumora (Fallback): {fallback_result['response']}")
#     print(f"Model Used: {fallback_result['metadata'].get('model_used')}")
