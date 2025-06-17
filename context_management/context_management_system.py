"""
Kumora Context Management System
Robust Redis-based implementation for user profiles, session context, and long-term memory
"""

import redis
from redis import Redis
from redis.sentinel import Sentinel
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import asyncio
# import aioredis
from collections import defaultdict, deque
import numpy as np
from pydantic import BaseModel, Field, validator
import pickle
import zlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Data Models ====================

class EmotionalState(BaseModel):
    """Model for emotional state data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    primary_emotion: str
    detected_emotions: List[str]
    intensity: float = Field(ge=0.0, le=1.0)
    valence: str
    confidence: float = Field(ge=0.0, le=1.0)
    triggers: Optional[List[str]] = []
    context: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserDemographics(BaseModel):
    """User demographic information"""
    age_range: Optional[str] = None  # e.g., "25-34"
    timezone: Optional[str] = None
    language: str = "en"
    cultural_background: Optional[str] = None
    occupation_category: Optional[str] = None
    relationship_status: Optional[str] = None
    
    # Privacy-preserving fields (optional)
    has_children: Optional[bool] = None
    urban_rural: Optional[str] = None  # "urban", "suburban", "rural"
    
    @validator('age_range')
    def validate_age_range(cls, v):
        valid_ranges = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        if v and v not in valid_ranges:
            raise ValueError(f"Age range must be one of {valid_ranges}")
        return v


class PersonalGrowthGoal(BaseModel):
    """Personal growth goal tracking"""
    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    category: str  # e.g., "emotional_regulation", "self_esteem", "relationships"
    created_at: datetime = Field(default_factory=datetime.now)
    target_date: Optional[datetime] = None
    status: str = "active"  # active, paused, completed, abandoned
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    milestones: List[Dict[str, Any]] = []
    related_emotions: List[str] = []


class ConversationTopic(BaseModel):
    """Conversation topic tracking"""
    topic: str
    category: str  # e.g., "work", "relationships", "health", "personal"
    sentiment: str  # positive, negative, neutral, mixed
    first_mentioned: datetime = Field(default_factory=datetime.now)
    last_mentioned: datetime = Field(default_factory=datetime.now)
    frequency: int = 1
    emotional_associations: Dict[str, float] = {}  # emotion -> strength
    keywords: List[str] = []


class CopingStrategy(BaseModel):
    """Coping strategy tracking"""
    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str  # e.g., "mindfulness", "physical", "social", "creative"
    description: str
    effectiveness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    emotional_contexts: List[str] = []  # emotions where this was helpful
    user_feedback: List[Dict[str, Any]] = []


class EmotionalPattern(BaseModel):
    """Detected emotional pattern"""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str  # e.g., "cyclical", "trigger-based", "time-based"
    description: str
    emotions_involved: List[str]
    triggers: List[str] = []
    frequency: str  # e.g., "daily", "weekly", "monthly"
    confidence: float = Field(ge=0.0, le=1.0)
    first_detected: datetime = Field(default_factory=datetime.now)
    last_observed: datetime = Field(default_factory=datetime.now)
    occurrences: int = 1


# ==================== Redis Configuration ====================

class RedisConfig:
    """Redis configuration and connection management"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 use_sentinel: bool = False,
                 sentinel_hosts: Optional[List[Tuple[str, int]]] = None,
                 sentinel_service: str = 'mymaster',
                 connection_pool_kwargs: Optional[Dict] = None):
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.use_sentinel = use_sentinel
        self.sentinel_hosts = sentinel_hosts or [('localhost', 26379)]
        self.sentinel_service = sentinel_service
        
        # Connection pool settings
        self.pool_kwargs = connection_pool_kwargs or {
            'max_connections': 50,
            'socket_keepalive': True,
            'socket_keepalive_options': {},
            'health_check_interval': 30
        }
        
        self._redis_client = None
        self._async_redis_client = None
    
    @property
    def redis_client(self) -> Redis:
        """Get or create Redis client with connection pooling"""
        if self._redis_client is None:
            if self.use_sentinel:
                sentinel = Sentinel(self.sentinel_hosts)
                self._redis_client = sentinel.master_for(
                    self.sentinel_service,
                    password=self.password,
                    db=self.db,
                    **self.pool_kwargs
                )
            else:
                pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    **self.pool_kwargs
                )
                self._redis_client = redis.Redis(connection_pool=pool)
        
        return self._redis_client
    
    async def get_async_client(self):
        """Get async Redis client for high-performance operations"""
        if self._async_redis_client is None:
            self._async_redis_client = await aioredis.create_redis_pool(
                f'redis://{self.host}:{self.port}/{self.db}',
                password=self.password,
                encoding='utf-8'
            )
        return self._async_redis_client


# ==================== Key Management ====================

class RedisKeyManager:
    """Centralized Redis key management with namespacing"""
    
    # Key prefixes
    USER_PROFILE = "user:profile:{user_id}"
    USER_DEMOGRAPHICS = "user:demographics:{user_id}"
    USER_GOALS = "user:goals:{user_id}"
    USER_EMOTIONAL_HISTORY = "user:emotional_history:{user_id}"
    
    SESSION_CONTEXT = "session:{session_id}:context"
    SESSION_EMOTIONAL_STATE = "session:{session_id}:emotional_state"
    SESSION_TOPICS = "session:{session_id}:topics"
    SESSION_TRIGGERS = "session:{session_id}:triggers"
    
    LONGTERM_PATTERNS = "longterm:patterns:{user_id}"
    LONGTERM_COPING = "longterm:coping:{user_id}"
    LONGTERM_PROGRESS = "longterm:progress:{user_id}"
    LONGTERM_PREFERENCES = "longterm:preferences:{user_id}"
    
    # Analytics keys
    ANALYTICS_EMOTIONS = "analytics:emotions:{user_id}:{date}"
    ANALYTICS_PATTERNS = "analytics:patterns:{user_id}:{month}"
    
    # Real-time keys
    REALTIME_CHANNEL = "realtime:updates:{user_id}"
    
    @staticmethod
    def get_key(template: str, **kwargs) -> str:
        """Generate Redis key from template"""
        return template.format(**kwargs)


# ==================== User Profile Management ====================

class UserProfileManager:
    """Manages user profiles, demographics, and personal information"""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis = redis_config.redis_client
        self.key_manager = RedisKeyManager()
        
    def create_user_profile(self, user_id: str, demographics: Optional[UserDemographics] = None) -> bool:
        """Create a new user profile"""
        try:
            profile_key = self.key_manager.get_key(self.key_manager.USER_PROFILE, user_id=user_id)
            
            profile_data = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'last_active': datetime.now().isoformat(),
                'total_sessions': 0,
                'emotional_awareness_score': 0.5,  # Initial score
                'engagement_level': 'new'  # new, active, regular, champion
            }
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.hset(profile_key, mapping=profile_data)
            
            # Set demographics if provided
            if demographics:
                demo_key = self.key_manager.get_key(self.key_manager.USER_DEMOGRAPHICS, user_id=user_id)
                pipe.hset(demo_key, mapping=demographics.dict())
            
            # Initialize empty collections
            goals_key = self.key_manager.get_key(self.key_manager.USER_GOALS, user_id=user_id)
            pipe.zadd(goals_key, {f"init_{datetime.now().timestamp()}": 0})
            pipe.zrem(goals_key, f"init_{datetime.now().timestamp()}")  # Remove placeholder
            
            pipe.execute()
            logger.info(f"Created profile for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Retrieve user profile"""
        profile_key = self.key_manager.get_key(self.key_manager.USER_PROFILE, user_id=user_id)
        profile = self.redis.hgetall(profile_key)
        
        if profile:
            # Get demographics
            demo_key = self.key_manager.get_key(self.key_manager.USER_DEMOGRAPHICS, user_id=user_id)
            demographics = self.redis.hgetall(demo_key)
            if demographics:
                profile['demographics'] = demographics
            
            # Get goals count
            goals_key = self.key_manager.get_key(self.key_manager.USER_GOALS, user_id=user_id)
            profile['active_goals'] = self.redis.zcard(goals_key)
            
            return profile
        return None
    
    def update_demographics(self, user_id: str, demographics: UserDemographics) -> bool:
        """Update user demographics"""
        demo_key = self.key_manager.get_key(self.key_manager.USER_DEMOGRAPHICS, user_id=user_id)
        return bool(self.redis.hset(demo_key, mapping=demographics.dict()))
    
    def add_growth_goal(self, user_id: str, goal: PersonalGrowthGoal) -> bool:
        """Add a personal growth goal"""
        goals_key = self.key_manager.get_key(self.key_manager.USER_GOALS, user_id=user_id)
        
        # Store goal data as JSON with score as timestamp for sorting
        goal_data = json.dumps(goal.dict(), default=str)
        score = goal.created_at.timestamp()
        
        return bool(self.redis.zadd(goals_key, {goal_data: score}))
    
    def get_active_goals(self, user_id: str) -> List[PersonalGrowthGoal]:
        """Get all active goals for a user"""
        goals_key = self.key_manager.get_key(self.key_manager.USER_GOALS, user_id=user_id)
        
        # Get all goals sorted by creation date
        goal_data_list = self.redis.zrange(goals_key, 0, -1)
        
        goals = []
        for goal_data in goal_data_list:
            try:
                goal_dict = json.loads(goal_data)
                goal = PersonalGrowthGoal(**goal_dict)
                if goal.status == "active":
                    goals.append(goal)
            except Exception as e:
                logger.error(f"Error parsing goal data: {e}")
        
        return goals
    
    def update_goal_progress(self, user_id: str, goal_id: str, progress: float, 
                           milestone: Optional[Dict] = None) -> bool:
        """Update progress on a goal"""
        goals_key = self.key_manager.get_key(self.key_manager.USER_GOALS, user_id=user_id)
        
        # Get all goals
        goal_data_list = self.redis.zrange(goals_key, 0, -1, withscores=True)
        
        for goal_data, score in goal_data_list:
            try:
                goal_dict = json.loads(goal_data)
                if goal_dict.get('goal_id') == goal_id:
                    # Update progress
                    goal_dict['progress'] = progress
                    
                    # Add milestone if provided
                    if milestone:
                        milestone['timestamp'] = datetime.now().isoformat()
                        goal_dict['milestones'].append(milestone)
                    
                    # Remove old and add updated
                    pipe = self.redis.pipeline()
                    pipe.zrem(goals_key, goal_data)
                    pipe.zadd(goals_key, {json.dumps(goal_dict, default=str): score})
                    pipe.execute()
                    
                    return True
            except Exception as e:
                logger.error(f"Error updating goal progress: {e}")
        
        return False
    
    def get_emotional_history(self, user_id: str, days: int = 30) -> List[EmotionalState]:
        """Get emotional history for the specified number of days"""
        history_key = self.key_manager.get_key(self.key_manager.USER_EMOTIONAL_HISTORY, user_id=user_id)
        
        # Calculate time range
        end_time = datetime.now().timestamp()
        start_time = (datetime.now() - timedelta(days=days)).timestamp()
        
        # Get data from sorted set within time range
        history_data = self.redis.zrangebyscore(
            history_key, 
            start_time, 
            end_time,
            withscores=True
        )
        
        emotional_states = []
        for data, timestamp in history_data:
            try:
                state_dict = json.loads(data)
                emotional_state = EmotionalState(**state_dict)
                emotional_states.append(emotional_state)
            except Exception as e:
                logger.error(f"Error parsing emotional state: {e}")
        
        return emotional_states


# ==================== Session Context Manager ====================

class SessionContextManager:
    """Manages session-level context and state"""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis = redis_config.redis_client
        self.key_manager = RedisKeyManager()
        self.session_ttl = 3600 * 24  # 24 hours default
        
    def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Create a new session"""
        if not session_id:
            session_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        context_key = self.key_manager.get_key(self.key_manager.SESSION_CONTEXT, session_id=session_id)
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'message_count': 0,
            'emotional_trajectory': 'neutral',  # improving, stable, declining
            'support_mode': 'general'  # general, crisis, growth, validation
        }
        
        pipe = self.redis.pipeline()
        pipe.hset(context_key, mapping=session_data)
        pipe.expire(context_key, self.session_ttl)
        pipe.execute()
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def update_emotional_state(self, session_id: str, emotional_state: EmotionalState) -> bool:
        """Update current emotional state in session"""
        state_key = self.key_manager.get_key(
            self.key_manager.SESSION_EMOTIONAL_STATE, 
            session_id=session_id
        )
        
        # Store current state
        state_data = emotional_state.json()
        
        pipe = self.redis.pipeline()
        pipe.lpush(state_key, state_data)
        pipe.ltrim(state_key, 0, 9)  # Keep last 10 states
        pipe.expire(state_key, self.session_ttl)
        
        # Update session context
        context_key = self.key_manager.get_key(self.key_manager.SESSION_CONTEXT, session_id=session_id)
        pipe.hset(context_key, 'last_emotional_state', emotional_state.primary_emotion)
        pipe.hset(context_key, 'last_emotional_intensity', str(emotional_state.intensity))
        pipe.hincrby(context_key, 'message_count', 1)
        
        results = pipe.execute()
        
        # Analyze emotional trajectory
        self._update_emotional_trajectory(session_id)
        
        return all(results)
    
    def add_conversation_topic(self, session_id: str, topic: ConversationTopic) -> bool:
        """Add or update a conversation topic"""
        topics_key = self.key_manager.get_key(self.key_manager.SESSION_TOPICS, session_id=session_id)
        
        # Check if topic exists
        existing_topics = self.redis.hgetall(topics_key)
        
        topic_found = False
        for topic_id, topic_data in existing_topics.items():
            stored_topic = ConversationTopic(**json.loads(topic_data))
            if stored_topic.topic.lower() == topic.topic.lower():
                # Update existing topic
                stored_topic.last_mentioned = datetime.now()
                stored_topic.frequency += 1
                
                # Merge emotional associations
                for emotion, strength in topic.emotional_associations.items():
                    if emotion in stored_topic.emotional_associations:
                        # Average the strengths
                        stored_topic.emotional_associations[emotion] = (
                            stored_topic.emotional_associations[emotion] + strength
                        ) / 2
                    else:
                        stored_topic.emotional_associations[emotion] = strength
                
                self.redis.hset(topics_key, topic_id, stored_topic.json())
                topic_found = True
                break
        
        if not topic_found:
            # Add new topic
            topic_id = f"topic_{uuid.uuid4().hex[:8]}"
            self.redis.hset(topics_key, topic_id, topic.json())
        
        self.redis.expire(topics_key, self.session_ttl)
        return True
    
    def identify_triggers(self, session_id: str, trigger_words: List[str], 
                         emotional_response: str) -> bool:
        """Identify and store emotional triggers"""
        triggers_key = self.key_manager.get_key(self.key_manager.SESSION_TRIGGERS, session_id=session_id)
        
        trigger_data = {
            'timestamp': datetime.now().isoformat(),
            'triggers': trigger_words,
            'emotional_response': emotional_response,
            'context': self.get_session_context(session_id)
        }
        
        pipe = self.redis.pipeline()
        pipe.lpush(triggers_key, json.dumps(trigger_data))
        pipe.ltrim(triggers_key, 0, 19)  # Keep last 20 triggers
        pipe.expire(triggers_key, self.session_ttl)
        pipe.execute()
        
        return True
    
    def determine_support_type(self, session_id: str) -> str:
        """Determine the type of support needed based on session context"""
        context = self.get_session_context(session_id)
        if not context:
            return "general"
        
        # Get recent emotional states
        state_key = self.key_manager.get_key(
            self.key_manager.SESSION_EMOTIONAL_STATE, 
            session_id=session_id
        )
        recent_states = self.redis.lrange(state_key, 0, 4)  # Last 5 states
        
        if not recent_states:
            return "general"
        
        # Analyze emotional patterns
        crisis_emotions = ['Feeling overwhelmed', 'Low self-esteem', 'Loneliness or Isolation']
        growth_emotions = ['Motivation', 'Hopefulness', 'Empowerment']
        
        crisis_count = 0
        growth_count = 0
        avg_intensity = 0
        
        for state_data in recent_states:
            state = EmotionalState(**json.loads(state_data))
            avg_intensity += state.intensity
            
            if any(emotion in state.detected_emotions for emotion in crisis_emotions):
                crisis_count += 1
            if any(emotion in state.detected_emotions for emotion in growth_emotions):
                growth_count += 1
        
        avg_intensity /= len(recent_states)
        
        # Determine support type
        if crisis_count >= 3 or (crisis_count >= 2 and avg_intensity > 0.7):
            return "crisis"
        elif growth_count >= 2:
            return "growth"
        elif avg_intensity > 0.6:
            return "validation"
        else:
            return "general"
    
    def get_session_context(self, session_id: str) -> Optional[Dict]:
        """Get complete session context"""
        context_key = self.key_manager.get_key(self.key_manager.SESSION_CONTEXT, session_id=session_id)
        context = self.redis.hgetall(context_key)
        
        if context:
            # Add support type
            context['support_type'] = self.determine_support_type(session_id)
            
            # Get topics
            topics_key = self.key_manager.get_key(self.key_manager.SESSION_TOPICS, session_id=session_id)
            topics_data = self.redis.hgetall(topics_key)
            context['topics'] = [
                ConversationTopic(**json.loads(topic_data)).dict() 
                for topic_data in topics_data.values()
            ]
            
            # Get recent emotional states
            state_key = self.key_manager.get_key(
                self.key_manager.SESSION_EMOTIONAL_STATE, 
                session_id=session_id
            )
            recent_states = self.redis.lrange(state_key, 0, 2)  # Last 3 states
            context['recent_emotional_states'] = [
                json.loads(state) for state in recent_states
            ]
            
            return context
        
        return None
    
    def _update_emotional_trajectory(self, session_id: str):
        """Analyze and update emotional trajectory"""
        state_key = self.key_manager.get_key(
            self.key_manager.SESSION_EMOTIONAL_STATE, 
            session_id=session_id
        )
        states_data = self.redis.lrange(state_key, 0, 9)  # Last 10 states
        
        if len(states_data) < 3:
            return  # Not enough data
        
        # Analyze trajectory
        intensities = []
        valences = []
        
        for state_data in states_data:
            state = EmotionalState(**json.loads(state_data))
            intensities.append(state.intensity)
            valences.append(1 if state.valence == 'positive' else -1 if state.valence == 'negative' else 0)
        
        # Calculate trends
        intensity_trend = np.polyfit(range(len(intensities)), intensities, 1)[0]
        valence_trend = np.polyfit(range(len(valences)), valences, 1)[0]
        
        # Determine trajectory
        if valence_trend > 0.1:
            trajectory = "improving"
        elif valence_trend < -0.1:
            trajectory = "declining"
        else:
            trajectory = "stable"
        
        # Update context
        context_key = self.key_manager.get_key(self.key_manager.SESSION_CONTEXT, session_id=session_id)
        self.redis.hset(context_key, 'emotional_trajectory', trajectory)


# ==================== Long-term Memory Manager ====================

class LongTermMemoryManager:
    """Manages long-term patterns, preferences, and learning"""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis = redis_config.redis_client
        self.key_manager = RedisKeyManager()
        
    def detect_emotional_pattern(self, user_id: str, 
                               emotional_history: List[EmotionalState]) -> List[EmotionalPattern]:
        """Detect patterns in emotional history"""
        if len(emotional_history) < 7:  # Need at least a week of data
            return []
        
        patterns = []
        
        # 1. Time-based patterns (e.g., Monday blues)
        time_patterns = self._detect_time_patterns(emotional_history)
        patterns.extend(time_patterns)
        
        # 2. Trigger-based patterns
        trigger_patterns = self._detect_trigger_patterns(emotional_history)
        patterns.extend(trigger_patterns)
        
        # 3. Cyclical patterns
        cyclical_patterns = self._detect_cyclical_patterns(emotional_history)
        patterns.extend(cyclical_patterns)
        
        # Store detected patterns
        if patterns:
            self._store_patterns(user_id, patterns)
        
        return patterns
    
    def _detect_time_patterns(self, history: List[EmotionalState]) -> List[EmotionalPattern]:
        """Detect time-based patterns (day of week, time of day)"""
        patterns = []
        
        # Group by day of week
        day_emotions = defaultdict(list)
        hour_emotions = defaultdict(list)
        
        for state in history:
            day = state.timestamp.strftime('%A')
            hour = state.timestamp.hour
            
            day_emotions[day].append(state)
            
            # Group hours into periods
            if 6 <= hour < 12:
                period = "morning"
            elif 12 <= hour < 17:
                period = "afternoon"
            elif 17 <= hour < 22:
                period = "evening"
            else:
                period = "night"
            
            hour_emotions[period].append(state)
        
        # Analyze day patterns
        for day, states in day_emotions.items():
            if len(states) >= 3:  # Need multiple occurrences
                # Check for consistent emotions
                emotion_counts = defaultdict(int)
                for state in states:
                    emotion_counts[state.primary_emotion] += 1
                
                # If one emotion appears >60% of the time
                for emotion, count in emotion_counts.items():
                    if count / len(states) > 0.6:
                        pattern = EmotionalPattern(
                            pattern_type="time-based",
                            description=f"Tends to feel {emotion} on {day}s",
                            emotions_involved=[emotion],
                            frequency="weekly",
                            confidence=count / len(states)
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_trigger_patterns(self, history: List[EmotionalState]) -> List[EmotionalPattern]:
        """Detect patterns related to specific triggers"""
        patterns = []
        
        # Group by triggers
        trigger_emotions = defaultdict(list)
        
        for state in history:
            if state.triggers:
                for trigger in state.triggers:
                    trigger_emotions[trigger].append(state)
        
        # Analyze trigger patterns
        for trigger, states in trigger_emotions.items():
            if len(states) >= 3:
                # Check emotional response consistency
                emotion_counts = defaultdict(int)
                for state in states:
                    emotion_counts[state.primary_emotion] += 1
                
                # Find dominant emotional response
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                consistency = emotion_counts[dominant_emotion] / len(states)
                
                if consistency > 0.7:
                    pattern = EmotionalPattern(
                        pattern_type="trigger-based",
                        description=f"{trigger} consistently triggers {dominant_emotion}",
                        emotions_involved=[dominant_emotion],
                        triggers=[trigger],
                        frequency="varies",
                        confidence=consistency
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_cyclical_patterns(self, history: List[EmotionalState]) -> List[EmotionalPattern]:
        """Detect cyclical emotional patterns"""
        patterns = []
        
        # Convert to time series
        timestamps = [state.timestamp for state in history]
        intensities = [state.intensity for state in history]
        
        # Simple cycle detection (looking for patterns every 3-7 days)
        for cycle_length in range(3, 8):
            if len(history) >= cycle_length * 2:
                # Check if pattern repeats
                cycle_match = 0
                for i in range(cycle_length, len(intensities)):
                    if abs(intensities[i] - intensities[i - cycle_length]) < 0.2:
                        cycle_match += 1
                
                match_ratio = cycle_match / (len(intensities) - cycle_length)
                
                if match_ratio > 0.6:
                    pattern = EmotionalPattern(
                        pattern_type="cyclical",
                        description=f"Emotional intensity follows a {cycle_length}-day cycle",
                        emotions_involved=list(set(state.primary_emotion for state in history)),
                        frequency=f"{cycle_length}-day cycle",
                        confidence=match_ratio
                    )
                    patterns.append(pattern)
                    break
        
        return patterns
    
    def _store_patterns(self, user_id: str, patterns: List[EmotionalPattern]):
        """Store detected patterns"""
        patterns_key = self.key_manager.get_key(self.key_manager.LONGTERM_PATTERNS, user_id=user_id)
        
        for pattern in patterns:
            # Check if similar pattern exists
            existing_patterns = self.redis.hgetall(patterns_key)
            
            pattern_exists = False
            for pattern_id, pattern_data in existing_patterns.items():
                existing = EmotionalPattern(**json.loads(pattern_data))
                
                # Check similarity
                if (existing.pattern_type == pattern.pattern_type and
                    existing.description == pattern.description):
                    # Update existing pattern
                    existing.last_observed = datetime.now()
                    existing.occurrences += 1
                    existing.confidence = (existing.confidence + pattern.confidence) / 2
                    
                    self.redis.hset(patterns_key, pattern_id, existing.json())
                    pattern_exists = True
                    break
            
            if not pattern_exists:
                # Store new pattern
                self.redis.hset(patterns_key, pattern.pattern_id, pattern.json())
    
    def track_coping_strategy(self, user_id: str, strategy: CopingStrategy, 
                            effectiveness: float, context: EmotionalState) -> bool:
        """Track usage and effectiveness of coping strategies"""
        coping_key = self.key_manager.get_key(self.key_manager.LONGTERM_COPING, user_id=user_id)
        
        # Get existing strategy or create new
        existing_data = self.redis.hget(coping_key, strategy.strategy_id)
        
        if existing_data:
            existing_strategy = CopingStrategy(**json.loads(existing_data))
            
            # Update effectiveness (weighted average)
            weight = 0.3  # Give 30% weight to new rating
            existing_strategy.effectiveness_score = (
                existing_strategy.effectiveness_score * (1 - weight) + 
                effectiveness * weight
            )
            
            existing_strategy.usage_count += 1
            existing_strategy.last_used = datetime.now()
            
            # Add emotional context if not present
            if context.primary_emotion not in existing_strategy.emotional_contexts:
                existing_strategy.emotional_contexts.append(context.primary_emotion)
            
            strategy = existing_strategy
        else:
            # New strategy
            strategy.effectiveness_score = effectiveness
            strategy.usage_count = 1
            strategy.last_used = datetime.now()
            strategy.emotional_contexts = [context.primary_emotion]
        
        # Store updated strategy
        return bool(self.redis.hset(coping_key, strategy.strategy_id, strategy.json()))
    
    def get_recommended_coping_strategies(self, user_id: str, 
                                        current_emotion: str, 
                                        limit: int = 3) -> List[CopingStrategy]:
        """Get recommended coping strategies based on effectiveness and context"""
        coping_key = self.key_manager.get_key(self.key_manager.LONGTERM_COPING, user_id=user_id)
        
        all_strategies = self.redis.hgetall(coping_key)
        
        strategies = []
        for strategy_data in all_strategies.values():
            strategy = CopingStrategy(**json.loads(strategy_data))
            
            # Calculate relevance score
            relevance = 0.0
            
            # Base effectiveness
            relevance += strategy.effectiveness_score * 0.5
            
            # Context match
            if current_emotion in strategy.emotional_contexts:
                relevance += 0.3
            
            # Recency bonus
            if strategy.last_used:
                days_since_use = (datetime.now() - strategy.last_used).days
                if days_since_use < 7:
                    relevance += 0.2 * (1 - days_since_use / 7)
            
            strategy.relevance_score = relevance
            strategies.append(strategy)
        
        # Sort by relevance and return top N
        strategies.sort(key=lambda s: s.relevance_score, reverse=True)
        
        return strategies[:limit]
    
    def track_progress(self, user_id: str, metric: str, value: float, 
                      context: Optional[Dict] = None) -> bool:
        """Track user progress on various metrics"""
        progress_key = self.key_manager.get_key(self.key_manager.LONGTERM_PROGRESS, user_id=user_id)
        
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric,
            'value': value,
            'context': context or {}
        }
        
        # Store in sorted set with timestamp as score
        score = datetime.now().timestamp()
        
        return bool(self.redis.zadd(
            progress_key, 
            {json.dumps(progress_data): score}
        ))
    
    def learn_preferences(self, user_id: str, preference_type: str, 
                         preference_data: Dict) -> bool:
        """Learn and store user preferences"""
        pref_key = self.key_manager.get_key(self.key_manager.LONGTERM_PREFERENCES, user_id=user_id)
        
        # Get existing preferences
        existing = self.redis.hget(pref_key, preference_type)
        
        if existing:
            current_prefs = json.loads(existing)
            # Merge with new data
            for key, value in preference_data.items():
                if key in current_prefs:
                    # Handle different types of preferences
                    if isinstance(value, (int, float)):
                        # Average numeric preferences
                        current_prefs[key] = (current_prefs[key] + value) / 2
                    elif isinstance(value, list):
                        # Combine lists
                        current_prefs[key] = list(set(current_prefs[key] + value))
                    else:
                        # Override other types
                        current_prefs[key] = value
                else:
                    current_prefs[key] = value
            
            preference_data = current_prefs
        
        return bool(self.redis.hset(
            pref_key, 
            preference_type, 
            json.dumps(preference_data)
        ))
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Generate comprehensive insights about the user"""
        insights = {
            'patterns': [],
            'coping_effectiveness': {},
            'progress_summary': {},
            'preferences': {},
            'recommendations': []
        }
        
        # Get patterns
        patterns_key = self.key_manager.get_key(self.key_manager.LONGTERM_PATTERNS, user_id=user_id)
        patterns_data = self.redis.hgetall(patterns_key)
        
        for pattern_data in patterns_data.values():
            pattern = EmotionalPattern(**json.loads(pattern_data))
            insights['patterns'].append({
                'type': pattern.pattern_type,
                'description': pattern.description,
                'confidence': pattern.confidence
            })
        
        # Get coping strategies effectiveness
        coping_key = self.key_manager.get_key(self.key_manager.LONGTERM_COPING, user_id=user_id)
        coping_data = self.redis.hgetall(coping_key)
        
        for strategy_data in coping_data.values():
            strategy = CopingStrategy(**json.loads(strategy_data))
            insights['coping_effectiveness'][strategy.category] = {
                'average_effectiveness': strategy.effectiveness_score,
                'usage_count': strategy.usage_count,
                'best_for': strategy.emotional_contexts[:3]
            }
        
        # Get progress summary
        progress_key = self.key_manager.get_key(self.key_manager.LONGTERM_PROGRESS, user_id=user_id)
        
        # Get last 30 days of progress
        end_time = datetime.now().timestamp()
        start_time = (datetime.now() - timedelta(days=30)).timestamp()
        
        progress_data = self.redis.zrangebyscore(progress_key, start_time, end_time)
        
        metric_values = defaultdict(list)
        for data in progress_data:
            progress = json.loads(data)
            metric_values[progress['metric']].append(progress['value'])
        
        for metric, values in metric_values.items():
            if values:
                insights['progress_summary'][metric] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'trend': 'improving' if values[-1] > values[0] else 'declining'
                }
        
        # Get preferences
        pref_key = self.key_manager.get_key(self.key_manager.LONGTERM_PREFERENCES, user_id=user_id)
        insights['preferences'] = self.redis.hgetall(pref_key)
        
        # Generate recommendations based on insights
        insights['recommendations'] = self._generate_recommendations(insights)
        
        return insights
    
    def _generate_recommendations(self, insights: Dict) -> List[str]:
        """Generate personalized recommendations based on insights"""
        recommendations = []
        
        # Pattern-based recommendations
        for pattern in insights['patterns']:
            if pattern['type'] == 'time-based' and 'Monday' in pattern['description']:
                recommendations.append(
                    "Consider planning lighter activities on Mondays to ease into the week"
                )
            elif pattern['type'] == 'trigger-based':
                recommendations.append(
                    f"Develop specific coping strategies for {pattern['description']}"
                )
        
        # Progress-based recommendations
        for metric, data in insights['progress_summary'].items():
            if data['trend'] == 'improving':
                recommendations.append(
                    f"Great progress on {metric}! Keep up the current strategies"
                )
            else:
                recommendations.append(
                    f"Let's focus on improving {metric} with new approaches"
                )
        
        return recommendations[:5]  # Limit to top 5 recommendations


# ==================== Main Context Manager ====================

class KumoraContextManager:
    """Main interface for the complete context management system"""
    
    def __init__(self, redis_config: Optional[RedisConfig] = None):
        self.redis_config = redis_config or RedisConfig()
        
        # Initialize sub-managers
        self.user_profile = UserProfileManager(self.redis_config)
        self.session = SessionContextManager(self.redis_config)
        self.long_term = LongTermMemoryManager(self.redis_config)
        
        # Real-time updates
        self.pubsub = self.redis_config.redis_client.pubsub()
        
    def process_user_message(self, user_id: str, session_id: str, 
                           message: str, emotion_analysis: Dict) -> Dict:
        """Process a user message and update all relevant context"""
        
        # 1. Update session emotional state
        emotional_state = EmotionalState(
            primary_emotion=emotion_analysis['primary_emotion'],
            detected_emotions=emotion_analysis['detected_emotions'],
            intensity=emotion_analysis['emotional_intensity'],
            valence=emotion_analysis['emotional_valence'],
            confidence=emotion_analysis.get('confidence', 0.8),
            context=message[:200]  # Store snippet
        )
        
        self.session.update_emotional_state(session_id, emotional_state)
        
        # 2. Extract and store conversation topics
        # This would integrate with NLP topic extraction
        # For now, using a simple approach
        if len(message.split()) > 5:
            topic = ConversationTopic(
                topic=self._extract_topic(message),
                category=self._categorize_topic(message),
                sentiment=emotion_analysis['emotional_valence'],
                emotional_associations={
                    emotion: 0.8 for emotion in emotion_analysis['detected_emotions']
                }
            )
            self.session.add_conversation_topic(session_id, topic)
        
        # 3. Store in user's emotional history
        history_key = RedisKeyManager.get_key(
            RedisKeyManager.USER_EMOTIONAL_HISTORY, 
            user_id=user_id
        )
        
        self.redis_config.redis_client.zadd(
            history_key,
            {emotional_state.json(): emotional_state.timestamp.timestamp()}
        )
        
        # 4. Check for patterns (async would be better)
        recent_history = self.user_profile.get_emotional_history(user_id, days=7)
        if len(recent_history) >= 20:  # Enough data
            patterns = self.long_term.detect_emotional_pattern(user_id, recent_history)
        
        # 5. Determine support type
        support_type = self.session.determine_support_type(session_id)
        
        # 6. Get context for response generation
        context = self.get_comprehensive_context(user_id, session_id)
        
        # 7. Publish real-time update
        self._publish_update(user_id, {
            'type': 'emotional_state_update',
            'session_id': session_id,
            'emotional_state': emotional_state.dict(),
            'support_type': support_type
        })
        
        return context
    
    def get_comprehensive_context(self, user_id: str, session_id: str) -> Dict:
        """Get complete context for response generation"""
        
        # Get user profile
        profile = self.user_profile.get_user_profile(user_id)
        
        # Get session context
        session_context = self.session.get_session_context(session_id)
        
        # Get active goals
        active_goals = self.user_profile.get_active_goals(user_id)
        
        # Get recommended coping strategies
        current_emotion = session_context.get('last_emotional_state', 'neutral')
        coping_strategies = self.long_term.get_recommended_coping_strategies(
            user_id, current_emotion
        )
        
        # Get user insights
        insights = self.long_term.get_user_insights(user_id)
        
        return {
            'user_profile': profile,
            'session': session_context,
            'active_goals': [goal.dict() for goal in active_goals],
            'recommended_coping': [strategy.dict() for strategy in coping_strategies],
            'patterns': insights['patterns'],
            'support_type': session_context.get('support_type', 'general'),
            'emotional_trajectory': session_context.get('emotional_trajectory', 'stable'),
            'personalization': {
                'preferences': insights['preferences'],
                'effective_strategies': insights['coping_effectiveness']
            }
        }
    
    def _extract_topic(self, message: str) -> str:
        """Simple topic extraction (would be replaced with NLP)"""
        # For now, just use first noun phrase or general category
        keywords = ['work', 'family', 'relationship', 'health', 'anxiety', 'stress']
        
        message_lower = message.lower()
        for keyword in keywords:
            if keyword in message_lower:
                return keyword
        
        return "general"
    
    def _categorize_topic(self, message: str) -> str:
        """Categorize topic (would be enhanced with NLP)"""
        categories = {
            'work': ['job', 'work', 'boss', 'colleague', 'deadline', 'project'],
            'relationships': ['partner', 'friend', 'family', 'relationship', 'love'],
            'health': ['sick', 'tired', 'sleep', 'pain', 'doctor', 'health'],
            'personal': ['feel', 'think', 'want', 'need', 'myself']
        }
        
        message_lower = message.lower()
        
        for category, keywords in categories.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _publish_update(self, user_id: str, update_data: Dict):
        """Publish real-time update for subscribers"""
        channel = RedisKeyManager.get_key(RedisKeyManager.REALTIME_CHANNEL, user_id=user_id)
        
        self.redis_config.redis_client.publish(
            channel, 
            json.dumps(update_data, default=str)
        )
    
    def subscribe_to_updates(self, user_id: str, callback):
        """Subscribe to real-time updates for a user"""
        channel = RedisKeyManager.get_key(RedisKeyManager.REALTIME_CHANNEL, user_id=user_id)
        
        self.pubsub.subscribe(channel)
        
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                update_data = json.loads(message['data'])
                callback(update_data)


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Initialize context manager
    context_manager = KumoraContextManager()
    
    # Create user profile
    user_id = "user_123"
    demographics = UserDemographics(
        age_range="25-34",
        timezone="America/New_York",
        occupation_category="professional"
    )
    
    context_manager.user_profile.create_user_profile(user_id, demographics)
    
    # Create session
    session_id = context_manager.session.create_session(user_id)
    
    # Simulate processing a message
    emotion_analysis = {
        'primary_emotion': 'Anxiety',
        'detected_emotions': ['Anxiety', 'Feeling overwhelmed'],
        'emotional_intensity': 0.7,
        'emotional_valence': 'negative'
    }
    
    context = context_manager.process_user_message(
        user_id, 
        session_id,
        "I'm feeling really anxious about my presentation tomorrow",
        emotion_analysis
    )
    
    print("Comprehensive Context:")
    print(json.dumps(context, indent=2, default=str))