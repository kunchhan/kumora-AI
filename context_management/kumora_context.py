# kumora_context_inmemory.py
"""
In-Memory Context Management for Kumora
Fallback implementation when Redis is not available
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import uuid
import logging

from context_management.context_management_system import *

logger = logging.getLogger(__name__)


class InMemoryContextManager:
    """In-memory implementation of context management for testing without Redis"""
    
    def __init__(self):
        # Storage dictionaries
        self.user_profiles = {}
        self.user_demographics = {}
        self.user_goals = defaultdict(list)
        self.emotional_history = defaultdict(list)
        self.sessions = {}
        self.session_emotional_states = defaultdict(deque)
        self.session_topics = defaultdict(dict)
        self.patterns = defaultdict(list)
        self.coping_strategies = defaultdict(dict)
        self.preferences = defaultdict(dict)
        
        # Initialize sub-managers
        self.user_profile = InMemoryUserProfileManager(self)
        self.session = InMemorySessionManager(self)
        self.long_term = InMemoryLongTermManager(self)
        
        logger.info("Using in-memory context management (Redis not available)")
    
    def get_comprehensive_context(self, user_id: str, session_id: str) -> Dict:
        """Get complete context for response generation"""
        
        # Get user profile
        profile = self.user_profile.get_user_profile(user_id)
        if not profile:
            # Create default profile if doesn't exist
            self.user_profile.create_user_profile(user_id)
            profile = self.user_profile.get_user_profile(user_id)
        
        # Get session context
        session_context = self.session.get_session_context(session_id)
        if not session_context:
            # Create session if doesn't exist
            self.session.create_session(user_id, session_id)
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
            'active_goals': [goal.dict() if hasattr(goal, 'dict') else goal for goal in active_goals],
            'recommended_coping': [strategy.dict() if hasattr(strategy, 'dict') else strategy for strategy in coping_strategies],
            'patterns': insights.get('patterns', []),
            'support_type': session_context.get('support_type', 'general'),
            'emotional_trajectory': session_context.get('emotional_trajectory', 'stable'),
            'personalization': {
                'preferences': insights.get('preferences', {}),
                'effective_strategies': insights.get('coping_effectiveness', {})
            }
        }
    
    def process_user_message(self, user_id: str, session_id: str, 
                           message: str, emotion_analysis: Dict) -> Dict:
        """Process a user message and update all relevant context"""
        
        # Update session emotional state
        emotional_state = EmotionalState(
            primary_emotion=emotion_analysis['primary_emotion'],
            detected_emotions=emotion_analysis['detected_emotions'],
            intensity=emotion_analysis['emotional_intensity'],
            valence=emotion_analysis['emotional_valence'],
            confidence=emotion_analysis.get('confidence', 0.8),
            context=message[:200]
        )
        
        self.session.update_emotional_state(session_id, emotional_state)
        
        # Store in emotional history
        self.emotional_history[user_id].append({
            'state': emotional_state,
            'timestamp': datetime.now()
        })
        
        # Determine support type
        support_type = self.session.determine_support_type(session_id)
        
        # Get context for response generation
        context = self.get_comprehensive_context(user_id, session_id)
        
        return context


class InMemoryUserProfileManager:
    """In-memory user profile management"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def create_user_profile(self, user_id: str, demographics: Optional[UserDemographics] = None) -> bool:
        """Create a new user profile"""
        profile_data = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'total_sessions': 0,
            'emotional_awareness_score': 0.5,
            'engagement_level': 'new'
        }
        
        self.parent.user_profiles[user_id] = profile_data
        
        if demographics:
            self.parent.user_demographics[user_id] = demographics.dict() if hasattr(demographics, 'dict') else demographics
        
        logger.info(f"Created in-memory profile for user {user_id}")
        return True
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Retrieve user profile"""
        profile = self.parent.user_profiles.get(user_id)
        
        if profile:
            profile = profile.copy()
            
            # Add demographics
            if user_id in self.parent.user_demographics:
                profile['demographics'] = self.parent.user_demographics[user_id]
            
            # Add goals count
            profile['active_goals'] = len([g for g in self.parent.user_goals[user_id] 
                                         if g.get('status') == 'active'])
            
            return profile
        return None
    
    def get_active_goals(self, user_id: str) -> List[Dict]:
        """Get all active goals for a user"""
        return [g for g in self.parent.user_goals[user_id] if g.get('status') == 'active']
    
    def get_emotional_history(self, user_id: str, days: int = 30) -> List[EmotionalState]:
        """Get emotional history for the specified number of days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        history = []
        for entry in self.parent.emotional_history[user_id]:
            if entry['timestamp'] > cutoff:
                history.append(entry['state'])
        
        return history


class InMemorySessionManager:
    """In-memory session management"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Create a new session"""
        if not session_id:
            session_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'message_count': 0,
            'emotional_trajectory': 'neutral',
            'support_mode': 'general'
        }
        
        self.parent.sessions[session_id] = session_data
        logger.info(f"Created in-memory session {session_id} for user {user_id}")
        return session_id
    
    def update_emotional_state(self, session_id: str, emotional_state: EmotionalState) -> bool:
        """Update current emotional state in session"""
        # Store in deque (max 10 states)
        states_deque = self.parent.session_emotional_states[session_id]
        if len(states_deque) >= 10:
            states_deque.popleft()
        states_deque.append(emotional_state)
        
        # Update session data
        if session_id in self.parent.sessions:
            self.parent.sessions[session_id]['last_emotional_state'] = emotional_state.primary_emotion
            self.parent.sessions[session_id]['last_emotional_intensity'] = emotional_state.intensity
            self.parent.sessions[session_id]['message_count'] += 1
            self.parent.sessions[session_id]['last_updated'] = datetime.now().isoformat()
        
        return True
    
    def get_session_context(self, session_id: str) -> Optional[Dict]:
        """Get complete session context"""
        if session_id not in self.parent.sessions:
            return None
        
        context = self.parent.sessions[session_id].copy()
        
        # Add support type
        context['support_type'] = self.determine_support_type(session_id)
        
        # Get topics
        context['topics'] = list(self.parent.session_topics[session_id].values())
        
        # Get recent emotional states
        recent_states = list(self.parent.session_emotional_states[session_id])[-3:]
        context['recent_emotional_states'] = [
            state.dict() if hasattr(state, 'dict') else state 
            for state in recent_states
        ]
        
        return context
    
    def determine_support_type(self, session_id: str) -> str:
        """Determine the type of support needed based on session context"""
        states = list(self.parent.session_emotional_states[session_id])
        
        if not states:
            return "general"
        
        # Analyze recent states
        crisis_emotions = ['Feeling overwhelmed', 'Low self-esteem', 'Loneliness or Isolation']
        growth_emotions = ['Motivation', 'Hopefulness', 'Empowerment']
        
        crisis_count = 0
        growth_count = 0
        avg_intensity = 0
        
        for state in states[-5:]:  # Last 5 states
            avg_intensity += state.intensity
            
            if any(emotion in state.detected_emotions for emotion in crisis_emotions):
                crisis_count += 1
            if any(emotion in state.detected_emotions for emotion in growth_emotions):
                growth_count += 1
        
        if states:
            avg_intensity /= len(states[-5:])
        
        # Determine support type
        if crisis_count >= 3 or (crisis_count >= 2 and avg_intensity > 0.7):
            return "crisis"
        elif growth_count >= 2:
            return "growth"
        elif avg_intensity > 0.6:
            return "validation"
        else:
            return "general"


class InMemoryLongTermManager:
    """In-memory long-term memory management"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def get_recommended_coping_strategies(self, user_id: str, 
                                        current_emotion: str, 
                                        limit: int = 3) -> List[Dict]:
        """Get recommended coping strategies"""
        # For testing, return some default strategies
        default_strategies = [
            {
                'name': 'Deep Breathing',
                'category': 'mindfulness',
                'description': 'Take slow, deep breaths to calm your nervous system',
                'effectiveness_score': 0.8
            },
            {
                'name': 'Journaling',
                'category': 'creative',
                'description': 'Write down your thoughts and feelings',
                'effectiveness_score': 0.7
            },
            {
                'name': 'Walk in Nature',
                'category': 'physical',
                'description': 'Take a gentle walk outside to clear your mind',
                'effectiveness_score': 0.75
            }
        ]
        
        return default_strategies[:limit]
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Generate user insights"""
        return {
            'patterns': [],
            'coping_effectiveness': {},
            'progress_summary': {},
            'preferences': self.parent.preferences.get(user_id, {}),
            'recommendations': []
        }


# Create a function to get the appropriate context manager
def get_context_manager():
    """Get context manager - try Redis first, fallback to in-memory"""
    try:
        # Try to import and connect to Redis
        from context_management.context_management_system import KumoraContextManager, RedisConfig
        
        # Test Redis connection
        redis_config = RedisConfig()
        redis_config.redis_client.ping()
        
        logger.info("Redis connection successful, using Redis-based context management")
        return KumoraContextManager(redis_config)
    
    except Exception as e:
        logger.warning(f"Redis not available ({e}), using in-memory context management")
        return InMemoryContextManager()