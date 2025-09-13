import streamlit as st
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import datetime
import json
import os
import time
import threading
import logging
from PIL import Image
import io
import base64
from typing import Dict, List, Optional, Tuple
import queue
import concurrent.futures
import re
import numpy as np

# Define PLOTTING_AVAILABLE to check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced page configuration
st.set_page_config(
    page_title="RoboCouplers Voice AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/robocouplers/voice-ai',
        'Report a bug': 'https://github.com/robocouplers/voice-ai/issues',
        'About': "RoboCouplers Voice AI Assistant v2.0 - Advanced Voice-to-Voice Interaction"
    }
)

# Enhanced Configuration
CONFIG = {
    "GEMINI_API_KEY": "AIzaSyDZlQf8oDLlUjCpEVzTIKz67BpOX3ZUwA4",  # Replace with your actual key
    "SUPPORTED_LANGUAGES": {
        "en": {"name": "English", "code": "en-US", "flag": "🇺🇸"},
        "es": {"name": "Spanish", "code": "es-ES", "flag": "🇪🇸"},
        "fr": {"name": "French", "code": "fr-FR", "flag": "🇫🇷"},
        "de": {"name": "German", "code": "de-DE", "flag": "🇩🇪"},
        "hi": {"name": "Hindi", "code": "hi-IN", "flag": "🇮🇳"},
        "zh": {"name": "Chinese", "code": "zh-CN", "flag": "🇨🇳"},
        "ja": {"name": "Japanese", "code": "ja-JP", "flag": "🇯🇵"},
        "pt": {"name": "Portuguese", "code": "pt-BR", "flag": "🇧🇷"},
        "ru": {"name": "Russian", "code": "ru-RU", "flag": "🇷🇺"},
        "ar": {"name": "Arabic", "code": "ar-SA", "flag": "🇸🇦"}
    },
    "VOICE_SETTINGS": {
        "rate_range": (100, 250),
        "volume_range": (0.1, 1.0),
        "default_rate": 150,
        "default_volume": 0.9
    },
    "AUDIO_SETTINGS": {
        "timeout": 10,
        "phrase_time_limit": 15,
        "energy_threshold": 300,
        "dynamic_energy_threshold": True
    },
    "AI_SETTINGS": {
        "max_response_lines": 3,
        "fallback_models": [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-1.0-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-1.0-pro'
        ],
        "response_timeout": 30
    }
}

# Enhanced Session State Initialization
def initialize_session_state():
    """Initialize all session state variables with enhanced features"""
    defaults = {
        'history': [],
        'selected_voice': 'female1',
        'selected_language': 'en',
        'is_listening': False,
        'is_speaking': False,
        'show_settings': False,
        'show_history': False,
        'show_analytics': False,
        'voice_rate': CONFIG["VOICE_SETTINGS"]["default_rate"],
        'voice_volume': CONFIG["VOICE_SETTINGS"]["default_volume"],
        'energy_threshold': CONFIG["AUDIO_SETTINGS"]["energy_threshold"],
        'conversation_count': 0,
        'total_words_spoken': 0,
        'total_words_heard': 0,
        'average_response_time': 0,
        'user_preferences': {
            'auto_speak': True,
            'show_transcription': True,
            'save_audio': False,
            'theme': 'modern'
        },
        'audio_queue': queue.Queue(),
        'error_log': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Enhanced VoiceChatbot Class
class EnhancedVoiceChatbot:
    def _init_(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.is_initialized = False
        self.voice_options = {}
        self.setup_audio_system()
        self.setup_ai_system()
        
        # Test the TTS engine during initialization
        try:
            self.tts_engine.say("Voice output test. If you hear this, the TTS engine is working.")
            self.tts_engine.runAndWait()
            logger.info("TTS engine test successful.")
        except Exception as e:
            logger.error(f"TTS engine test failed: {e}")
        
    def setup_audio_system(self):
        """Enhanced audio system setup with error handling"""
        try:
            # Configure recognizer with enhanced settings
            self.recognizer.energy_threshold = st.session_state.energy_threshold
            self.recognizer.dynamic_energy_threshold = CONFIG["AUDIO_SETTINGS"]["dynamic_energy_threshold"]
            self.recognizer.pause_threshold = 0.8
            self.recognizer.operation_timeout = CONFIG["AUDIO_SETTINGS"]["timeout"]
            
            # Setup microphone
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Setup TTS engine with enhanced voice options
            self.setup_enhanced_voices()
            self.is_initialized = True
            logger.info("Audio system initialized successfully")
            
        except Exception as e:
            logger.error(f"Audio system initialization failed: {e}")
            st.error(f"Audio system error: {e}")
            
    def setup_enhanced_voices(self):
        """Setup enhanced voice options with more control"""
        try:
            voices = self.tts_engine.getProperty('voices')
            
            self.voice_options = {
                'male1': {
                    'name': 'Professional Male',
                    'id': 0,
                    'gender': 'male',
                    'description': 'Clear, professional male voice'
                },
                'male2': {
                    'name': 'Casual Male',
                    'id': 1 if len(voices) > 1 else 0,
                    'gender': 'male',
                    'description': 'Friendly, casual male voice'
                },
                'female1': {
                    'name': 'Professional Female',
                    'id': 2 if len(voices) > 2 else 0,
                    'gender': 'female',
                    'description': 'Clear, professional female voice'
                },
                'female2': {
                    'name': 'Warm Female',
                    'id': 3 if len(voices) > 3 else 0,
                    'gender': 'female',
                    'description': 'Warm, friendly female voice'
                }
            }
            
            # Add additional voices if available
            for i, voice in enumerate(voices[4:8], 4):
                voice_key = f'voice{i+1}'
                gender = 'female' if 'female' in voice.name.lower() else 'male'
                self.voice_options[voice_key] = {
                    'name': f'System Voice {i+1}',
                    'id': i,
                    'gender': gender,
                    'description': f'System {gender} voice'
                }
                
        except Exception as e:
            logger.error(f"Voice setup error: {e}")
            # Fallback to basic voice setup
            self.voice_options = {
                'male1': {'name': 'Default Male', 'id': 0, 'gender': 'male', 'description': 'Default voice'},
                'female1': {'name': 'Default Female', 'id': 0, 'gender': 'female', 'description': 'Default voice'}
            }
    
    def setup_ai_system(self):
        """Setup AI system with enhanced configuration"""
        try:
            if CONFIG["GEMINI_API_KEY"] and CONFIG["GEMINI_API_KEY"] != "your_gemini_api_key_here":
                genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
                logger.info("Gemini AI configured successfully")
            else:
                logger.warning("Gemini API key not configured")
        except Exception as e:
            logger.error(f"AI system setup error: {e}")
    
    def set_voice_properties(self, voice_type: str, rate: int = None, volume: float = None):
        """Enhanced voice property setting"""
        try:
            voices = self.tts_engine.getProperty('voices')
            voice_config = self.voice_options.get(voice_type, self.voice_options['female1'])
            
            if voice_config['id'] < len(voices):
                self.tts_engine.setProperty('voice', voices[voice_config['id']].id)
            
            # Set rate and volume
            self.tts_engine.setProperty('rate', rate or st.session_state.voice_rate)
            self.tts_engine.setProperty('volume', volume or st.session_state.voice_volume)
            
            logger.info(f"Voice properties set: {voice_type}, rate: {rate}, volume: {volume}")
            
        except Exception as e:
            logger.error(f"Voice property setting error: {e}")
    
    def enhanced_speech_recognition(self) -> Tuple[str, Dict]:
        """Enhanced speech recognition with detailed feedback"""
        start_time = time.time()
        recognition_data = {
            'success': False,
            'text': '',
            'confidence': 0,
            'duration': 0,
            'language_detected': st.session_state.selected_language,
            'error': None
        }
        
        try:
            # Update UI
            status_placeholder = st.empty()
            with status_placeholder:
                st.info("🎤 Listening... Speak clearly into your microphone!")
            
            # Configure for selected language
            language_code = CONFIG["SUPPORTED_LANGUAGES"][st.session_state.selected_language]["code"]
            
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=CONFIG["AUDIO_SETTINGS"]["timeout"],
                    phrase_time_limit=CONFIG["AUDIO_SETTINGS"]["phrase_time_limit"]
                )
            
            with status_placeholder:
                st.info("🔄 Processing your speech...")
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio, language=language_code)
            
            recognition_data.update({
                'success': True,
                'text': text,
                'duration': time.time() - start_time,
                'confidence': 0.95  # Google API doesn't return confidence, using default
            })
            
            # Update statistics
            word_count = len(text.split())
            st.session_state.total_words_heard += word_count
            
            with status_placeholder:
                st.success(f"✅ Speech recognized: '{text}'")
            
            return text, recognition_data
            
        except sr.WaitTimeoutError:
            error_msg = "⏱ No speech detected. Please try again."
            recognition_data['error'] = "timeout"
        except sr.UnknownValueError:
            error_msg = "🤷 Could not understand the speech. Please speak more clearly."
            recognition_data['error'] = "unknown_value"
        except sr.RequestError as e:
            error_msg = f"🌐 Speech service error: {str(e)[:50]}..."
            recognition_data['error'] = "request_error"
        except Exception as e:
            error_msg = f"❌ Microphone error: {str(e)[:50]}..."
            recognition_data['error'] = "general_error"
        
        recognition_data['duration'] = time.time() - start_time
        
        # Log error
        error_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'speech_recognition',
            'error': recognition_data['error'],
            'message': error_msg
        }
        st.session_state.error_log.append(error_entry)
        
        return error_msg, recognition_data
    
    def enhanced_ai_response(self, user_input: str) -> Tuple[str, Dict]:
        """Enhanced AI response generation with multiple fallbacks"""
        start_time = time.time()
        response_data = {
            'success': False,
            'text': '',
            'model_used': None,
            'duration': 0,
            'word_count': 0,
            'language': st.session_state.selected_language,
            'error': None
        }
        
        try:
            if CONFIG["GEMINI_API_KEY"] == "your_gemini_api_key_here":
                fallback_response = self.get_fallback_response(user_input)
                response_data.update({
                    'success': True,
                    'text': fallback_response,
                    'model_used': 'fallback',
                    'duration': time.time() - start_time,
                    'word_count': len(fallback_response.split())
                })
                return fallback_response, response_data
            
            # Enhanced prompt based on language and context
            language_name = CONFIG["SUPPORTED_LANGUAGES"][st.session_state.selected_language]["name"]
            
            enhanced_prompt = f"""You are a helpful AI voice assistant. Respond in {language_name} language.
            
User Query: {user_input}

Instructions:
- Provide a helpful, accurate response in EXACTLY 3 lines or less
- Each line should be concise (max 25 words per line)
- Make it conversational and natural for voice output
- Avoid bullet points, numbers, or special formatting
- If the query requires a longer explanation, prioritize the most important information
- Respond in {language_name} language

Response format:
Line 1: Direct answer or main point
Line 2: Key supporting detail or explanation
Line 3: Additional context or conclusion (if needed)"""

            # Try multiple models with fallback
            for model_name in CONFIG["AI_SETTINGS"]["fallback_models"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        enhanced_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=150,
                            top_p=0.8,
                            top_k=40
                        )
                    )
                    
                    if response and response.text:
                        # Process and clean response
                        clean_response = self.clean_ai_response(response.text)
                        
                        response_data.update({
                            'success': True,
                            'text': clean_response,
                            'model_used': model_name,
                            'duration': time.time() - start_time,
                            'word_count': len(clean_response.split())
                        })
                        
                        # Update statistics
                        st.session_state.total_words_spoken += response_data['word_count']
                        
                        return clean_response, response_data
                        
                except Exception as model_error:
                    logger.warning(f"Model {model_name} failed: {model_error}")
                    continue
            
            # If all models fail, use fallback
            fallback_response = self.get_fallback_response(user_input)
            response_data.update({
                'success': True,
                'text': fallback_response,
                'model_used': 'fallback',
                'duration': time.time() - start_time,
                'word_count': len(fallback_response.split()),
                'error': 'all_models_failed'
            })
            return fallback_response, response_data
            
        except Exception as e:
            error_msg = f"AI processing error: {str(e)[:100]}..."
            response_data.update({
                'duration': time.time() - start_time,
                'error': 'processing_error'
            })
            
            # Log error
            error_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': 'ai_response',
                'error': str(e),
                'user_input': user_input[:100]
            }
            st.session_state.error_log.append(error_entry)
            
            return error_msg, response_data
    
    def clean_ai_response(self, response_text: str) -> str:
        """Clean and format AI response for voice output"""
        # Remove special characters and formatting
        clean_text = re.sub(r'[•\-*]', '', response_text)
        clean_text = re.sub(r'\n+', '\n', clean_text)
        
        # Split into lines and limit to 3
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        
        if len(lines) > CONFIG["AI_SETTINGS"]["max_response_lines"]:
            lines = lines[:CONFIG["AI_SETTINGS"]["max_response_lines"]]
        
        # Ensure each line is not too long for voice
        processed_lines = []
        for line in lines:
            if len(line.split()) > 30:  # If line too long, truncate
                words = line.split()[:28]
                line = ' '.join(words) + "..."
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def get_fallback_response(self, user_input: str) -> str:
        """Generate fallback responses when AI is not available"""
        fallback_responses = {
            'greeting': "Hello! I'm your voice assistant. How can I help you today?\nI can answer questions, have conversations, and assist with various tasks.\nPlease make sure your Gemini API key is configured for full AI responses.",
            'time': f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.\nToday's date is {datetime.datetime.now().strftime('%B %d, %Y')}.\nI hope you're having a great day!",
            'weather': "I'd love to help with weather information.\nPlease configure the Gemini API key for real-time responses.\nYou can check your local weather app for current conditions.",
            'default': "I hear you asking about something interesting.\nTo provide detailed responses, please set up the Gemini API key.\nI'm here and ready to help once that's configured!"
        }
        
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return fallback_responses['greeting']
        elif any(word in user_lower for word in ['time', 'date', 'today']):
            return fallback_responses['time']
        elif any(word in user_lower for word in ['weather', 'temperature', 'rain']):
            return fallback_responses['weather']
        else:
            return fallback_responses['default']
    
    def enhanced_text_to_speech(self, text: str) -> Dict:
        """Enhanced text-to-speech with better control and feedback"""
        speech_data = {
            'success': False,
            'duration': 0,
            'text_length': len(text),
            'estimated_speech_time': len(text.split()) * 0.6,  # Rough estimate
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Set voice properties
            self.set_voice_properties(
                st.session_state.selected_voice,
                st.session_state.voice_rate,
                st.session_state.voice_volume
            )
            
            # Clean text for better speech
            speech_text = self.prepare_text_for_speech(text)
            
            # Update UI
            st.session_state.is_speaking = True
            
            # Add debug logging to trace TTS execution
            logger.debug("Starting enhanced_text_to_speech method.")
            logger.debug(f"Text to be spoken: {text}")
            logger.debug(f"Selected voice: {st.session_state.selected_voice}")
            logger.debug(f"Speech rate: {st.session_state.voice_rate}, Volume: {st.session_state.voice_volume}")
            
            # Ensure TTS engine is initialized
            if not self.tts_engine:
                logger.error("TTS engine is not initialized.")
                raise RuntimeError("TTS engine initialization failed.")
            
            # Check if auto-speak is enabled
            if not st.session_state.user_preferences.get('auto_speak', True):
                logger.info("Auto-speak is disabled. Skipping voice output.")
                return speech_data
            
            # Speak the text
            self.tts_engine.say(speech_text)
            self.tts_engine.runAndWait()
            
            speech_data.update({
                'success': True,
                'duration': time.time() - start_time
            })
            
            st.session_state.is_speaking = False
            logger.info(f"Speech completed successfully in {speech_data['duration']:.2f} seconds")
            
        except Exception as e:
            speech_data.update({
                'duration': time.time() - start_time,
                'error': str(e)
            })
            st.session_state.is_speaking = False
            logger.error(f"Text-to-speech error: {e}")
            
            # Log error
            error_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': 'text_to_speech',
                'error': str(e),
                'text_preview': text[:50]
            }
            st.session_state.error_log.append(error_entry)
        
        return speech_data
    
    def prepare_text_for_speech(self, text: str) -> str:
        """Prepare text for optimal speech synthesis"""
        # Replace newlines with pauses
        speech_text = text.replace('\n', '. ')
        
        # Remove excessive whitespace
        speech_text = re.sub(r'\s+', ' ', speech_text)
        
        # Add pauses for better speech rhythm
        speech_text = re.sub(r'([.!?])\s*', r'\1 ... ', speech_text)
        
        return speech_text.strip()

# Enhanced UI Components
def create_enhanced_sidebar():
    """Create enhanced sidebar with more features"""
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.15);'>
            <div style='font-size: 3.5em; margin-bottom: 10px; animation: glow 2s ease-in-out infinite alternate;'>🤖</div>
            <h2 style='color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>RoboCouplers</h2>
            <p style='font-size: 12px; color: #f0f0f0; margin: 5px 0 0 0;'>Advanced Voice AI Assistant v2.0</p>
        </div>
        <style>
            @keyframes glow {
                from { text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #667eea; }
                to { text-shadow: 0 0 20px #fff, 0 0 30px #764ba2, 0 0 40px #764ba2; }
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Language selection with flags
        st.markdown("### 🌐 Language Selection")
        language_options = {code: f"{info['flag']} {info['name']}" 
                          for code, info in CONFIG["SUPPORTED_LANGUAGES"].items()}
        
        selected_language = st.selectbox(
            "Choose your language:",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.selected_language)
        )
        st.session_state.selected_language = selected_language
        
        st.markdown("---")
        
        # Settings panel
        if st.button("⚙ ADVANCED SETTINGS", use_container_width=True, type="secondary"):
            st.session_state.show_settings = not st.session_state.show_settings
        
        if st.session_state.show_settings:
            create_settings_panel()
        
        st.markdown("---")
        
        # History panel
        if st.button("📝 CONVERSATION HISTORY", use_container_width=True, type="secondary"):
            st.session_state.show_history = not st.session_state.show_history
        
        if st.session_state.show_history:
            create_history_panel()
        
        st.markdown("---")
        
        # Analytics panel
        if st.button("📊 ANALYTICS", use_container_width=True, type="secondary"):
            st.session_state.show_analytics = not st.session_state.show_analytics
        
        if st.session_state.show_analytics:
            create_analytics_panel()
        
        st.markdown("---")
        
        # System status
        create_system_status()

def create_settings_panel():
    """Create enhanced settings panel"""
    st.markdown("### 🎛 Voice & Audio Settings")
    
    # Voice selection with descriptions
    st.markdown("🎵 Voice Selection:")
    voice_options = st.session_state.chatbot.voice_options
    
    for voice_key, voice_info in voice_options.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                f"{voice_info['name']}\n{voice_info['description']}", 
                key=f"voice_{voice_key}",
                help=f"Select {voice_info['name']} - {voice_info['description']}"
            ):
                st.session_state.selected_voice = voice_key
                st.success(f"✅ {voice_info['name']} selected!")
        
        with col2:
            if st.button("🔊", key=f"test_{voice_key}", help="Test this voice"):
                def test_voice():
                    st.session_state.chatbot.set_voice_properties(voice_key)
                    st.session_state.chatbot.tts_engine.say(f"Hello! This is {voice_info['name']}.")
                    st.session_state.chatbot.tts_engine.runAndWait()
                
                test_thread = threading.Thread(target=test_voice)
                test_thread.daemon = True
                test_thread.start()
    
    st.markdown("🎚 Voice Controls:")
    
    # Speech rate control
    st.session_state.voice_rate = st.slider(
        "Speech Rate (words per minute)",
        min_value=CONFIG["VOICE_SETTINGS"]["rate_range"][0],
        max_value=CONFIG["VOICE_SETTINGS"]["rate_range"][1],
        value=st.session_state.voice_rate,
        step=10,
        help="Adjust how fast the AI speaks"
    )
    
    # Volume control
    st.session_state.voice_volume = st.slider(
        "Voice Volume",
        min_value=CONFIG["VOICE_SETTINGS"]["volume_range"][0],
        max_value=CONFIG["VOICE_SETTINGS"]["volume_range"][1],
        value=st.session_state.voice_volume,
        step=0.1,
        format="%.1f",
        help="Adjust voice volume level"
    )
    
    # Audio input sensitivity
    st.session_state.energy_threshold = st.slider(
        "Microphone Sensitivity",
        min_value=100,
        max_value=1000,
        value=st.session_state.energy_threshold,
        step=50,
        help="Adjust microphone sensitivity for voice detection"
    )
    
    st.markdown("🔧 Preferences:")
    
    # User preferences
    st.session_state.user_preferences['auto_speak'] = st.checkbox(
        "🗣 Auto-speak responses",
        value=st.session_state.user_preferences['auto_speak'],
        help="Automatically speak AI responses"
    )
    
    st.session_state.user_preferences['show_transcription'] = st.checkbox(
        "📝 Show transcription",
        value=st.session_state.user_preferences['show_transcription'],
        help="Display speech-to-text transcription"
    )
    
    st.session_state.user_preferences['save_audio'] = st.checkbox(
        "💾 Save conversations",
        value=st.session_state.user_preferences['save_audio'],
        help="Save conversation history automatically"
    )

def create_history_panel():
    """Create enhanced history panel with search and filters"""
    st.markdown("### 📚 Conversation History")
    
    if st.session_state.history:
        # Search functionality
        search_term = st.text_input("🔍 Search conversations", placeholder="Enter keywords...")
        
        # Filter conversations
        filtered_history = st.session_state.history
        if search_term:
            filtered_history = [
                item for item in st.session_state.history
                if search_term.lower() in item['question'].lower() or search_term.lower() in item['answer'].lower()
            ]
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(st.session_state.history))
        with col2:
            st.metric("Found", len(filtered_history))
        with col3:
            avg_length = sum(len(item['answer'].split()) for item in st.session_state.history) / len(st.session_state.history) if st.session_state.history else 0
            st.metric("Avg Words", f"{avg_length:.1f}")
        
        # Display conversations
        for i, item in enumerate(reversed(filtered_history[-10:])):
            with st.expander(f"💬 {item['question'][:30]}...", expanded=(i==0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"🕐 Time:** {item['timestamp']}")
                    st.markdown(f"❓ Question:** {item['question']}")
                    st.markdown(f"🤖 Answer:** {item['answer']}")
                    
                    # Additional metadata if available
                    if 'response_data' in item:
                        data = item['response_data']
                        if data.get('duration'):
                            st.caption(f"Response time: {data['duration']:.2f}s | Model: {data.get('model_used', 'N/A')}")
                
                with col2:
                    # Action buttons
                    if st.button("🔊", key=f"replay_{i}", help="Replay answer"):
                        def replay_answer():
                            st.session_state.chatbot.enhanced_text_to_speech(item['answer'])
                        
                        replay_thread = threading.Thread(target=replay_answer)
                        replay_thread.daemon = True
                        replay_thread.start()
                        st.info("🔊 Replaying answer...")
                    
                    if st.button("📋", key=f"copy_{i}", help="Copy to clipboard"):
                        st.code(f"Q: {item['question']}\nA: {item['answer']}")
        
        st.markdown("---")
        
        # History management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 Clear History", use_container_width=True, type="primary"):
                if st.button("⚠ Confirm Clear", key="confirm_clear"):
                    st.session_state.history = []
                    st.success("🧹 History cleared!")
                    st.rerun()
        
        with col2:
            if st.button("📥 Export History", use_container_width=True):
                history_data = {
                    'exported_at': datetime.datetime.now().isoformat(),
                    'total_conversations': len(st.session_state.history),
                    'conversations': st.session_state.history
                }
                history_json = json.dumps(history_data, indent=2)
                st.download_button(
                    label="💾 Download JSON",
                    data=history_json,
                    file_name=f"robocouplers_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("📭 No conversations yet.")
        st.markdown("Start by clicking the *🎤 SPEAK* button!")

def create_analytics_panel():
    """Create analytics panel with conversation statistics"""
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.history:
        # Basic statistics
        total_conversations = len(st.session_state.history)
        total_words_spoken = st.session_state.total_words_spoken
        total_words_heard = st.session_state.total_words_heard
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Conversations", total_conversations)
            st.metric("Words Spoken by AI", total_words_spoken)
        
        with col2:
            st.metric("Words Heard from You", total_words_heard)
            if total_conversations > 0:
                avg_words = total_words_spoken / total_conversations
                st.metric("Avg Words per Response", f"{avg_words:.1f}")
        
        # Conversation timeline (simple text-based if plotting not available)
        st.markdown("📈 Recent Activity:")
        if PLOTTING_AVAILABLE and len(st.session_state.history) > 1:
            try:
                # Create simple activity chart
                timestamps = [datetime.datetime.fromisoformat(item['timestamp']) for item in st.session_state.history[-10:]]
                conversation_count = list(range(1, len(timestamps) + 1))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(timestamps, conversation_count, marker='o', linewidth=2, markersize=6)
                ax.set_title('Conversation Activity')
                ax.set_xlabel('Time')
                ax.set_ylabel('Conversation Number')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.text("Activity chart unavailable")
        else:
            # Text-based timeline
            for i, item in enumerate(st.session_state.history[-5:]):
                timestamp = item['timestamp']
                question_preview = item['question'][:40] + "..." if len(item['question']) > 40 else item['question']
                st.text(f"{timestamp}: {question_preview}")
        
        # Language usage
        if st.session_state.history:
            st.markdown("🌐 Language Usage:")
            language_counts = {}
            for item in st.session_state.history:
                lang = item.get('language', 'en')
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            for lang, count in language_counts.items():
                lang_name = CONFIG["SUPPORTED_LANGUAGES"].get(lang, {}).get("name", lang)
                percentage = (count / len(st.session_state.history)) * 100
                st.text(f"{lang_name}: {count} conversations ({percentage:.1f}%)")
        
        # Error statistics
        if st.session_state.error_log:
            st.markdown("⚠ Error Summary:")
            error_types = {}
            for error in st.session_state.error_log[-10:]:
                error_type = error.get('type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                st.text(f"{error_type.replace('_', ' ').title()}: {count}")
    
    else:
        st.info("📊 Start conversations to see analytics!")

def create_system_status():
    """Create system status panel"""
    st.markdown("### 🔧 System Status")
    
    # Current settings display
    current_voice = st.session_state.chatbot.voice_options.get(
        st.session_state.selected_voice, {'name': 'Unknown'}
    )['name']
    
    current_language = CONFIG["SUPPORTED_LANGUAGES"][st.session_state.selected_language]["name"]
    
    status_data = {
        "🎵 Voice": current_voice,
        "🌐 Language": current_language,
        "💬 Conversations": len(st.session_state.history),
        "🎚 Speech Rate": f"{st.session_state.voice_rate} WPM",
        "🔊 Volume": f"{st.session_state.voice_volume:.1f}",
        "🎤 Mic Sensitivity": st.session_state.energy_threshold
    }
    
    for key, value in status_data.items():
        st.text(f"{key}: {value}")
    
    # System health indicators
    st.markdown("🔋 System Health:")
    
    # Check API status
    api_status = "🟢 Connected" if CONFIG["GEMINI_API_KEY"] != "your_gemini_api_key_here" else "🔴 Not Configured"
    st.text(f"AI Service: {api_status}")
    
    # Check audio systems
    audio_status = "🟢 Ready" if st.session_state.chatbot.is_initialized else "🔴 Error"
    st.text(f"Audio System: {audio_status}")
    
    # Memory usage indicator
    history_size = len(json.dumps(st.session_state.history))
    memory_status = "🟢 Good" if history_size < 10000 else "🟡 High" if history_size < 50000 else "🔴 Very High"
    st.text(f"Memory Usage: {memory_status}")

def save_enhanced_conversation(question: str, answer: str, recognition_data: Dict = None, response_data: Dict = None):
    """Save conversation with enhanced metadata"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conversation_entry = {
        'timestamp': timestamp,
        'question': question,
        'answer': answer,
        'language': st.session_state.selected_language,
        'voice_used': st.session_state.selected_voice,
        'session_id': id(st.session_state)  # Simple session identifier
    }
    
    # Add recognition metadata if available
    if recognition_data:
        conversation_entry['recognition_data'] = {
            'success': recognition_data.get('success', False),
            'confidence': recognition_data.get('confidence', 0),
            'duration': recognition_data.get('duration', 0),
            'error': recognition_data.get('error')
        }
    
    # Add response metadata if available
    if response_data:
        conversation_entry['response_data'] = {
            'model_used': response_data.get('model_used'),
            'duration': response_data.get('duration', 0),
            'word_count': response_data.get('word_count', 0),
            'success': response_data.get('success', False)
        }
    
    st.session_state.history.append(conversation_entry)
    st.session_state.conversation_count += 1

def create_main_interface():
    """Create the main voice interaction interface"""
    # Header section with enhanced styling
    st.markdown("""
    <div style='text-align: center; padding: 40px 0; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 20px; margin: 20px 0;'>
        <h1 style='font-size: 4em; margin-bottom: 20px; color: #333; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
            🗣 VOICE ASSISTANT 🤖
        </h1>
        <p style='font-size: 1.2em; color: #666; margin: 0;'>
            Speak naturally • Get instant AI responses • Available in 10+ languages
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Voice waveform visualization with enhanced animation
    if not st.session_state.is_listening and not st.session_state.is_speaking:
        waveform_color = "background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 25%, #45B7D1 50%, #96CEB4 75%, #FFEAA7 100%);"
        waveform_text = "🎵 ▓▓▓▓▓ READY TO LISTEN ▓▓▓▓▓ 🎵"
    elif st.session_state.is_listening:
        waveform_color = "background: linear-gradient(90deg, #ff4757 0%, #ff6b7a 50%, #ff4757 100%);"
        waveform_text = "🎙 ▓▓▓▓▓ LISTENING... ▓▓▓▓▓ 🎙"
    else:
        waveform_color = "background: linear-gradient(90deg, #5f27cd 0%, #a55eea 50%, #5f27cd 100%);"
        waveform_text = "🔊 ▓▓▓▓▓ SPEAKING... ▓▓▓▓▓ 🔊"
    
    st.markdown(f"""
    <div style='text-align: center; margin: 40px 0;'>
        <div style='{waveform_color} 
                    height: 70px; width: 90%; margin: 0 auto; border-radius: 35px; 
                    display: flex; align-items: center; justify-content: center;
                    animation: pulse 2s infinite; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    border: 3px solid rgba(255,255,255,0.3);'>
            <span style='color: white; font-weight: bold; font-size: 18px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
                {waveform_text}
            </span>
        </div>
    </div>
    <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 0.8; }}
            50% {{ transform: scale(1.05); opacity: 1; }}
            100% {{ transform: scale(1); opacity: 0.8; }}
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Main interaction area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Main speak button with enhanced styling
        speak_button = st.button(
            "🎤 SPEAK NOW",
            use_container_width=True,
            type="primary",
            help="Click and speak clearly. The AI will respond in voice!",
            key="main_speak_button",
            disabled=st.session_state.is_listening or st.session_state.is_speaking
        )
        
        if speak_button:
            handle_voice_interaction()
        
        # Status display
        if st.session_state.is_listening:
            st.info("🎤 *Listening...* Speak clearly into your microphone!")
        elif st.session_state.is_speaking:
            st.success("🔊 *Speaking...* AI is responding!")
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions")
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🎯 Voice Test", use_container_width=True, help="Test current voice settings"):
            test_message = f"Hello! This is your {st.session_state.chatbot.voice_options[st.session_state.selected_voice]['name']} speaking at {st.session_state.voice_rate} words per minute."
            
            def test_voice():
                st.session_state.chatbot.enhanced_text_to_speech(test_message)
            
            test_thread = threading.Thread(target=test_voice)
            test_thread.daemon = True
            test_thread.start()
            st.success("🔊 Testing voice...")
    
    with action_col2:
        if st.button("🔄 Reset Audio", use_container_width=True, help="Reset audio systems"):
            try:
                st.session_state.chatbot.setup_audio_system()
                st.success("🔄 Audio system reset!")
            except Exception as e:
                st.error(f"Reset failed: {e}")
    
    with action_col3:
        if st.button("📊 Quick Stats", use_container_width=True, help="Show quick statistics"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Conversations", len(st.session_state.history))
                st.metric("Current Voice", st.session_state.chatbot.voice_options[st.session_state.selected_voice]['name'])
            with col_b:
                st.metric("Language", CONFIG["SUPPORTED_LANGUAGES"][st.session_state.selected_language]["name"])
                st.metric("Errors", len(st.session_state.error_log))
    
    with action_col4:
        if st.button("🆘 Help Guide", use_container_width=True, help="Show help information"):
            create_help_guide()

def handle_voice_interaction():
    """Handle the complete voice interaction flow"""
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    
    try:
        # Step 1: Speech Recognition
        st.session_state.is_listening = True
        
        with status_placeholder:
            st.info("🎤 *Step 1:* Listening for your voice...")
        
        user_input, recognition_data = st.session_state.chatbot.enhanced_speech_recognition()
        st.session_state.is_listening = False
        
        # Check if recognition was successful
        if not recognition_data['success']:
            with result_placeholder:
                st.error(f"❌ *Speech Recognition Failed:* {user_input}")
            time.sleep(3)
            status_placeholder.empty()
            result_placeholder.empty()
            return
        
        # Step 2: Display recognized text
        with result_placeholder:
            st.success(f"📝 *You said:* {user_input}")
        
        # Step 3: AI Processing
        with status_placeholder:
            st.info("🤖 *Step 2:* AI is processing your request...")
        
        ai_response, response_data = st.session_state.chatbot.enhanced_ai_response(user_input)
        
        # Step 4: Display AI response
        with result_placeholder:
            st.success(f"📝 *You said:* {user_input}")
            st.info(f"🤖 *AI Response:*\n\n{ai_response}")
        
        # Step 5: Text-to-Speech (if enabled)
        if st.session_state.user_preferences['auto_speak']:
            with status_placeholder:
                st.info("🔊 *Step 3:* Converting to speech...")
            
            st.session_state.is_speaking = True
            
            def speak_response():
                speech_data = st.session_state.chatbot.enhanced_text_to_speech(ai_response)
                st.session_state.is_speaking = False
                return speech_data
            
            # Run speech in separate thread
            speech_thread = threading.Thread(target=speak_response)
            speech_thread.daemon = True
            speech_thread.start()
            
            # Wait briefly for speech to start
            time.sleep(1)
        
        # Step 6: Save conversation
        save_enhanced_conversation(user_input, ai_response, recognition_data, response_data)
        
        # Step 7: Show completion status
        with status_placeholder:
            completion_message = "✅ *Complete!* "
            if st.session_state.user_preferences['auto_speak']:
                completion_message += "Response played and saved to history."
            else:
                completion_message += "Response ready. Enable auto-speak in settings for voice output."
            
            st.success(completion_message)
        
        # Auto-clear status after delay
        time.sleep(4)
        status_placeholder.empty()
        
        # Keep result visible longer
        time.sleep(2)
        result_placeholder.empty()
        
    except Exception as e:
        # Handle any unexpected errors
        st.session_state.is_listening = False
        st.session_state.is_speaking = False
        
        error_msg = f"Unexpected error: {str(e)[:100]}..."
        with result_placeholder:
            st.error(f"❌ *System Error:* {error_msg}")
        
        # Log error
        error_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'system_error',
            'error': str(e),
            'context': 'voice_interaction'
        }
        st.session_state.error_log.append(error_entry)
        
        time.sleep(3)
        status_placeholder.empty()
        result_placeholder.empty()

def create_help_guide():
    """Create help guide modal"""
    st.markdown("""
    ### 🆘 Help Guide - RoboCouplers Voice Assistant
    
    *🎤 How to Use:*
    1. Click the *🎤 SPEAK NOW* button
    2. Speak clearly into your microphone
    3. Wait for AI to process and respond
    4. Listen to the voice response (if auto-speak enabled)
    
    *⚙ Settings:*
    - *Voice Selection:* Choose from male/female voices
    - *Speech Rate:* Adjust how fast AI speaks (100-250 WPM)
    - *Volume:* Control voice output volume
    - *Language:* Select from 10+ supported languages
    
    *🔧 Troubleshooting:*
    - *No audio detected:* Check microphone permissions
    - *Poor recognition:* Speak clearly, reduce background noise
    - *AI not responding:* Verify internet connection and API key
    - *Voice not working:* Test different voice options in settings
    
    *📊 Features:*
    - *History:* Review past conversations
    - *Analytics:* View usage statistics
    - *Export:* Download conversation history
    - *Multi-language:* Support for 10+ languages
    
    *🔐 Privacy:*
    - Conversations stored locally in session
    - No data transmitted except to AI service
    - Clear history anytime
    """)

def create_footer():
    """Create enhanced footer with system info"""
    st.markdown("---")
    
    # System status footer
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col2:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        session_duration = time.time() - st.session_state.get('session_start_time', time.time())
        
        # System metrics
        system_info = {
            "🎵 Active Voice": st.session_state.chatbot.voice_options[st.session_state.selected_voice]['name'],
            "🌐 Language": CONFIG["SUPPORTED_LANGUAGES"][st.session_state.selected_language]['name'],
            "💬 Conversations": len(st.session_state.history),
            "⏱ Session Time": f"{int(session_duration//60)}m {int(session_duration%60)}s",
            "🕐 Current Time": current_time,
            "📊 System Status": "🟢 Active"
        }
        
       

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Track session start time
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = time.time()
    
    # Initialize chatbot
    # Ensure the chatbot initialization happens after the EnhancedVoiceChatbot class definition
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = EnhancedVoiceChatbot()
    
    # Create sidebar
    create_enhanced_sidebar()
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # API Configuration Check
        if CONFIG["GEMINI_API_KEY"] == "your_gemini_api_key_here":
            st.error("""
            ### ⚠ Configuration Required
            *To enable full AI responses:*
            1. Get your Gemini API key from: https://aistudio.google.com/app/apikey
            2. Replace the API key in line 50 of the code
            3. Save and restart the application
            
            *Current Status:* Basic fallback responses only
            """)
        
        # Main interface
        create_main_interface()
        
        # Instructions section
        st.markdown("---")
        create_instructions_section()
        
        # Footer
        create_footer()

def create_instructions_section():
    """Create enhanced instructions section"""
    instruction_col1, instruction_col2, instruction_col3 = st.columns([1, 4, 1])
    
    with instruction_col2:
         unsafe_allow_html=True

# Run the app
if __name__ == "_main_":
    main()