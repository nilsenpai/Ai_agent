# config.py - Advanced configuration settings

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration class for the voice agent"""

    # Latency optimization settings
    target_total_latency: float = 2.0  # Target total latency in seconds
    max_response_tokens: int = 150  # Limit LLM response length
    enable_preemptive_synthesis: bool = True  # Start TTS before LLM completion

    # Interruption handling
    allow_interruptions: bool = True
    interrupt_speech_duration: float = 0.5  # Time to detect interruption
    interrupt_min_words: int = 2  # Min words to trigger interruption

    # Audio processing
    min_endpointing_delay: float = 0.3  # VAD endpointing delay
    sample_rate: int = 24000  # Audio sample rate

    # STT Configuration
    stt_model: str = "nova-2-general"  # Deepgram model
    stt_language: str = "en"
    stt_smart_format: bool = True
    stt_interim_results: bool = True  # Real-time transcription
    stt_endpointing: int = 300  # Quick endpointing (ms)

    # LLM Configuration
    llm_model: str = "llama-3.1-70b-versatile"  # Groq model
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    llm_base_url: str = "https://api.groq.com/openai/v1"

    # TTS Configuration
    tts_voice: str = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Cartesia Sonic
    tts_model: str = "sonic-english"
    tts_sample_rate: int = 24000

    # Metrics and logging
    metrics_file: str = "call_metrics.xlsx"
    enable_detailed_logging: bool = True
    log_audio_metrics: bool = True

    # System prompt
    system_prompt: str = (
        "You are a helpful AI voice assistant. Keep your responses concise and natural. "
        "Respond in a conversational tone and keep answers brief to maintain low latency. "
        "If interrupted, acknowledge the interruption gracefully. "
        "Focus on being helpful while keeping responses under 2 sentences when possible."
    )


# Alternative configurations for different use cases
@dataclass
class HighQualityConfig(AgentConfig):
    """Configuration optimized for quality over speed"""
    target_total_latency: float = 4.0
    max_response_tokens: int = 300
    llm_model: str = "llama-3.1-70b-versatile"
    tts_voice: str = "premium-voice-id"  # Higher quality voice
    stt_model: str = "nova-2-general"  # Most accurate STT


@dataclass
class UltraFastConfig(AgentConfig):
    """Configuration optimized for minimum latency"""
    target_total_latency: float = 1.0
    max_response_tokens: int = 50
    llm_model: str = "llama-3.1-8b-instant"  # Fastest Groq model
    interrupt_speech_duration: float = 0.3
    min_endpointing_delay: float = 0.2
    stt_endpointing: int = 200


@dataclass
class ProductionConfig(AgentConfig):
    """Configuration for production deployment"""
    enable_detailed_logging: bool = False
    log_audio_metrics: bool = False
    target_total_latency: float = 1.5
    max_response_tokens: int = 200
    system_prompt: str = (
        "You are a professional AI assistant. Provide helpful, accurate, and concise responses. "
        "Maintain a friendly but professional tone. Keep responses brief and to the point."
    )


def get_config(config_type: str = "default") -> AgentConfig:
    """Get configuration based on type"""
    configs = {
        "default": AgentConfig(),
        "high_quality": HighQualityConfig(),
        "ultra_fast": UltraFastConfig(),
        "production": ProductionConfig()
    }

    return configs.get(config_type, AgentConfig())


def load_config_from_env() -> AgentConfig:
    """Load configuration from environment variables"""
    config = AgentConfig()

    # Override with environment variables if present
    config.target_total_latency = float(os.getenv("TARGET_LATENCY", config.target_total_latency))
    config.max_response_tokens = int(os.getenv("MAX_TOKENS", config.max_response_tokens))
    config.llm_model = os.getenv("LLM_MODEL", config.llm_model)
    config.stt_model = os.getenv("STT_MODEL", config.stt_model)
    config.tts_voice = os.getenv("TTS_VOICE", config.tts_voice)
    config.allow_interruptions = os.getenv("ALLOW_INTERRUPTIONS", "true").lower() == "true"

    return config


# Latency thresholds for performance monitoring
LATENCY_THRESHOLDS = {
    "excellent": {
        "eou_delay": 0.1,
        "ttft": 0.2,
        "ttfb": 0.4,
        "total_latency": 1.0
    },
    "good": {
        "eou_delay": 0.2,
        "ttft": 0.4,
        "ttfb": 0.8,
        "total_latency": 2.0
    },
    "acceptable": {
        "eou_delay": 0.3,
        "ttft": 0.6,
        "ttfb": 1.2,
        "total_latency": 3.0
    }
}


def evaluate_latency_performance(metrics: dict) -> str:
    """Evaluate latency performance against thresholds"""
    for level, thresholds in LATENCY_THRESHOLDS.items():
        if all(metrics.get(key, float('inf')) <= threshold
               for key, threshold in thresholds.items()):
            return level
    return "poor"