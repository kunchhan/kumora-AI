# kumora_chat_terminal.py
"""
Kumora Terminal Chat Interface
Simple command-line interface for testing the Response Engine
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional
import uuid
from colorama import init, Fore, Style
import json

# Initialize colorama for cross-platform colored output
init()

# Import your Kumora components
from kumora_response_engine import *
from prompt_engineering_module.class_utils import *

class KumoraTerminalChat:
    """Terminal-based chat interface for Kumora"""
    
    def __init__(self):
        self.engine: Optional[KumoraResponseEngine] = None
        self.user_id = f"terminal_user_{uuid.uuid4().hex[:8]}"
        self.session_id = None
        self.conversation_history = []
        
    async def initialize(self):
        """Initialize the Kumora engine"""
        print(f"{Fore.CYAN}🌸 Initializing Kumora...{Style.RESET_ALL}")
        
        try:
            # Initialize with default config
            config = ModelConfig()
            self.engine = await initialize_kumora_engine(config)
            
            # Create a new session
            self.session_id = self.engine.context_manager.session.create_session(self.user_id)
            
            print(f"{Fore.GREEN}✓ Kumora is ready to chat!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Session ID: {self.session_id}{Style.RESET_ALL}\n")
            
            # Print welcome message
            self._print_welcome()
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to initialize Kumora: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    def _print_welcome(self):
        """Print welcome message"""
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🌸 Welcome to Kumora - Your Emotionally Intelligent Companion 🌸{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}I'm here to listen and support you through whatever you're experiencing.")
        print(f"Feel free to share your thoughts and feelings - this is a safe space.{Style.RESET_ALL}\n")
        print(f"{Fore.YELLOW}Commands:{Style.RESET_ALL}")
        print(f"  • Type 'quit' or 'exit' to end the conversation")
        print(f"  • Type 'clear' to clear the screen")
        print(f"  • Type 'debug' to toggle debug information")
        print(f"  • Type 'history' to see conversation history")
        print(f"  • Type 'health' to check system health\n")
    
    async def chat_loop(self):
        """Main chat loop"""
        debug_mode = False
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    await self._goodbye()
                    break
                    
                elif user_input.lower() == 'clear':
                    self._clear_screen()
                    continue
                    
                elif user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"{Fore.YELLOW}Debug mode: {'ON' if debug_mode else 'OFF'}{Style.RESET_ALL}\n")
                    continue
                    
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                    
                elif user_input.lower() == 'health':
                    await self._check_health()
                    continue
                    
                elif not user_input:
                    continue
                
                # Generate response
                print(f"{Fore.BLUE}Kumora: {Style.RESET_ALL}", end="", flush=True)
                
                response_data = await self.engine.generate_response(
                    user_message=user_input,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    conversation_history=self.conversation_history[-2:]
                )
                
                # Print response with typing effect
                response = response_data['response']
                self._print_with_typing_effect(response)
                print()  # New line after response
                
                # Show debug info if enabled
                if debug_mode:
                    self._print_debug_info(response_data['metadata'])
                
                # Add to history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'kumora': response,
                    'metadata': response_data['metadata']
                })
                
                print()  # Extra line for readability
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'quit' or 'exit' to end the conversation properly.{Style.RESET_ALL}")
                continue
                
            except Exception as e:
                print(f"\n{Fore.RED}✗ Error: {e}{Style.RESET_ALL}\n")
                continue
    
    def _print_with_typing_effect(self, text: str, delay: float = 0.01):
        """Print text with typing effect"""
        import time
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
    
    def _print_debug_info(self, metadata: dict):
        """Print debug information"""
        print(f"\n{Fore.YELLOW}--- Debug Info ---{Style.RESET_ALL}")
        
        # Emotion analysis
        if 'emotion_analysis' in metadata:
            ea = metadata['emotion_analysis']
            print(f"Primary Emotion: {ea['primary_emotion']}")
            print(f"Detected Emotions: {', '.join(ea['detected_emotions'])}")
            print(f"Intensity: {ea['emotional_intensity']:.2f}")
            print(f"Valence: {ea['emotional_valence']}")
            print(f"Confidence: {ea.get('emotional_confidence', 'N/A')}")
        
        # Generation metadata
        print(f"Support Type: {metadata.get('support_type', 'N/A')}")
        print(f"Model Used: {metadata.get('model_used', 'N/A')}")
        print(f"Generation Time: {metadata.get('generation_time', 0):.2f}s")
        print(f"Total Time: {metadata.get('total_time', 0):.2f}s")
        print(f"Cached: {metadata.get('cached', False)}")
        
        if metadata.get('fallback_triggered'):
            print(f"Fallback Reason: {metadata.get('fallback_reason', 'N/A')}")
        
        print(f"{Fore.YELLOW}--- End Debug ---{Style.RESET_ALL}")
    
    def _show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print(f"{Fore.YELLOW}No conversation history yet.{Style.RESET_ALL}\n")
            return
        
        print(f"\n{Fore.CYAN}=== Conversation History ==={Style.RESET_ALL}")
        for entry in self.conversation_history:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            print(f"\n{Fore.YELLOW}[{timestamp}]{Style.RESET_ALL}")
            print(f"{Fore.GREEN}You: {Style.RESET_ALL}{entry['user']}")
            print(f"{Fore.BLUE}Kumora: {Style.RESET_ALL}{entry['kumora']}")
        print(f"\n{Fore.CYAN}=== End of History ==={Style.RESET_ALL}\n")
    
    async def _check_health(self):
        """Check system health"""
        print(f"\n{Fore.YELLOW}Checking system health...{Style.RESET_ALL}")
        
        health_status = await self.engine.health_check()
        
        print(f"\n{Fore.CYAN}=== System Health ==={Style.RESET_ALL}")
        print(f"Overall Status: {self._format_health_status(health_status['status'])}")
        
        print(f"\n{Fore.CYAN}Components:{Style.RESET_ALL}")
        for component, status in health_status['components'].items():
            status_formatted = self._format_health_status(status)
            print(f"  • {component}: {status_formatted}")
        
        print(f"\n{Fore.CYAN}Metrics:{Style.RESET_ALL}")
        metrics = health_status['metrics']
        print(f"  • Total Requests: {metrics['total_requests']}")
        print(f"  • Llama Success Rate: {metrics['llama_success_rate']:.1%}")
        print(f"  • Fallback Rate: {metrics['fallback_rate']:.1%}")
        
        print(f"\n{Fore.CYAN}=== End Health Check ==={Style.RESET_ALL}\n")
    
    def _format_health_status(self, status: str) -> str:
        """Format health status with color"""
        if status == 'healthy':
            return f"{Fore.GREEN}✓ {status}{Style.RESET_ALL}"
        elif status == 'degraded':
            return f"{Fore.YELLOW}⚠ {status}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}✗ {status}{Style.RESET_ALL}"
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        self._print_welcome()
    
    async def _goodbye(self):
        """Say goodbye and save conversation"""
        print(f"\n{Fore.MAGENTA}Thank you for sharing with me today. Take care of yourself! 🌸{Style.RESET_ALL}")
        
        # Save conversation history
        if self.conversation_history:
            filename = f"kumora_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'user_id': self.user_id,
                    'session_id': self.session_id,
                    'conversation': self.conversation_history
                }, f, indent=2)
            print(f"{Fore.YELLOW}Conversation saved to: {filename}{Style.RESET_ALL}")


async def main():
    """Main entry point"""
    chat = KumoraTerminalChat()
    await chat.initialize()
    await chat.chat_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Chat ended by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")