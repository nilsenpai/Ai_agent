#!/usr/bin/env python3

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
from dataclasses import dataclass, field
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, cartesia
from livekit import rtc
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")


@dataclass
class CallMetrics:
    """Data class to store call session metrics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Latency metrics (in seconds)
    eou_delays: list = field(default_factory=list)  # End of Utterance delays
    ttft_times: list = field(default_factory=list)  # Time to First Token
    ttfb_times: list = field(default_factory=list)  # Time to First Byte (audio)
    total_latencies: list = field(default_factory=list)  # End-to-end latencies

    # Usage counters
    user_messages: int = 0
    agent_responses: int = 0
    interruptions: int = 0
    errors: int = 0

    # Audio metrics
    total_audio_duration: float = 0.0
    total_silence_duration: float = 0.0


class MetricsLogger:
    """Handles logging and Excel export of call metrics"""

    def __init__(self, excel_file: str = "call_metrics.xlsx"):
        self.excel_file = excel_file
        self.sessions: Dict[str, CallMetrics] = {}

    def start_session(self, session_id: str) -> CallMetrics:
        """Start tracking metrics for a new session"""
        metrics = CallMetrics(
            session_id=session_id,
            start_time=datetime.now()
        )
        self.sessions[session_id] = metrics
        logger.info(f"Started metrics tracking for session {session_id}")
        return metrics

    def end_session(self, session_id: str):
        """End session and log final metrics"""
        if session_id in self.sessions:
            self.sessions[session_id].end_time = datetime.now()
            self._export_to_excel(session_id)
            logger.info(f"Ended metrics tracking for session {session_id}")

    def log_latency(self, session_id: str, metric_type: str, value: float):
        """Log latency metrics"""
        if session_id not in self.sessions:
            return

        metrics = self.sessions[session_id]
        if metric_type == "eou_delay":
            metrics.eou_delays.append(value)
        elif metric_type == "ttft":
            metrics.ttft_times.append(value)
        elif metric_type == "ttfb":
            metrics.ttfb_times.append(value)
        elif metric_type == "total_latency":
            metrics.total_latencies.append(value)

    def log_event(self, session_id: str, event_type: str):
        """Log various events"""
        if session_id not in self.sessions:
            return

        metrics = self.sessions[session_id]
        if event_type == "user_message":
            metrics.user_messages += 1
        elif event_type == "agent_response":
            metrics.agent_responses += 1
        elif event_type == "interruption":
            metrics.interruptions += 1
        elif event_type == "error":
            metrics.errors += 1

    def _export_to_excel(self, session_id: str):
        """Export session metrics to Excel"""
        if session_id not in self.sessions:
            return

        metrics = self.sessions[session_id]

        # Calculate averages
        avg_eou_delay = sum(metrics.eou_delays) / len(metrics.eou_delays) if metrics.eou_delays else 0
        avg_ttft = sum(metrics.ttft_times) / len(metrics.ttft_times) if metrics.ttft_times else 0
        avg_ttfb = sum(metrics.ttfb_times) / len(metrics.ttfb_times) if metrics.ttfb_times else 0
        avg_total_latency = sum(metrics.total_latencies) / len(
            metrics.total_latencies) if metrics.total_latencies else 0

        # Calculate session duration
        duration = (metrics.end_time - metrics.start_time).total_seconds() if metrics.end_time else 0

        # Prepare data for Excel
        session_data = {
            'Session ID': [session_id],
            'Start Time': [metrics.start_time.strftime('%Y-%m-%d %H:%M:%S')],
            'End Time': [metrics.end_time.strftime('%Y-%m-%d %H:%M:%S') if metrics.end_time else 'N/A'],
            'Duration (seconds)': [duration],
            'Average EOU Delay (s)': [round(avg_eou_delay, 3)],
            'Average TTFT (s)': [round(avg_ttft, 3)],
            'Average TTFB (s)': [round(avg_ttfb, 3)],
            'Average Total Latency (s)': [round(avg_total_latency, 3)],
            'Max Total Latency (s)': [round(max(metrics.total_latencies), 3) if metrics.total_latencies else 0],
            'User Messages': [metrics.user_messages],
            'Agent Responses': [metrics.agent_responses],
            'Interruptions': [metrics.interruptions],
            'Errors': [metrics.errors],
            'Total Audio Duration (s)': [round(metrics.total_audio_duration, 2)],
        }

        df = pd.DataFrame(session_data)

        # Append to existing Excel file or create new one
        try:
            existing_df = pd.read_excel(self.excel_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            combined_df = df

        combined_df.to_excel(self.excel_file, index=False)
        logger.info(f"Metrics exported to {self.excel_file}")


class EnhancedVoiceAssistant(VoiceAssistant):
    """Enhanced Voice Assistant with metrics tracking and interruption handling"""

    def __init__(self, metrics_logger: MetricsLogger, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self.metrics_logger = metrics_logger
        self.session_id = session_id
        self.last_speech_time = 0
        self.response_start_time = 0
        self.is_speaking = False

    async def _handle_interruption(self):
        """Handle user interruptions during agent speech"""
        if self.is_speaking:
            self.metrics_logger.log_event(self.session_id, "interruption")
            logger.info("User interruption detected - stopping current response")
            # Stop current TTS playback
            await self._stop_current_response()

    async def _stop_current_response(self):
        """Stop the current agent response"""
        # Implementation depends on TTS provider - this is a placeholder
        self.is_speaking = False

    async def _on_user_speech_committed(self, user_msg: str):
        """Called when user speech is finalized"""
        speech_end_time = time.time()

        # Calculate EOU delay (time from speech end to processing start)
        eou_delay = time.time() - speech_end_time
        self.metrics_logger.log_latency(self.session_id, "eou_delay", eou_delay)
        self.metrics_logger.log_event(self.session_id, "user_message")

        logger.info(f"User said: {user_msg} (EOU delay: {eou_delay:.3f}s)")

        # Record response start time for TTFT calculation
        self.response_start_time = time.time()

        return await super()._on_user_speech_committed(user_msg)

    async def _on_agent_response_start(self):
        """Called when agent starts generating response"""
        if self.response_start_time > 0:
            ttft = time.time() - self.response_start_time
            self.metrics_logger.log_latency(self.session_id, "ttft", ttft)
            logger.info(f"TTFT: {ttft:.3f}s")

    async def _on_agent_speech_start(self):
        """Called when agent starts speaking (TTS begins)"""
        if self.response_start_time > 0:
            ttfb = time.time() - self.response_start_time
            self.metrics_logger.log_latency(self.session_id, "ttfb", ttfb)
            logger.info(f"TTFB: {ttfb:.3f}s")

        self.is_speaking = True
        self.metrics_logger.log_event(self.session_id, "agent_response")

    async def _on_agent_speech_end(self):
        """Called when agent finishes speaking"""
        if self.response_start_time > 0:
            total_latency = time.time() - self.response_start_time
            self.metrics_logger.log_latency(self.session_id, "total_latency", total_latency)
            logger.info(f"Total latency: {total_latency:.3f}s")
            self.response_start_time = 0

        self.is_speaking = False


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent"""

    # Initialize metrics logger
    metrics_logger = MetricsLogger()
    session_id = f"session_{int(time.time())}"
    metrics = metrics_logger.start_session(session_id)

    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to room {ctx.room.name}")

    # Initialize AI services with optimized settings for low latency
    try:
        # STT: Deepgram (fast and accurate)
        stt = deepgram.STT(
            model="nova-2-general",
            language="en",
            smart_format=True,
            interim_results=True,  # Enable real-time transcription
            endpointing=300,  # Quick endpointing for low latency
        )

        # LLM: Use Groq for fast inference
        llm_instance = openai.LLM(
            model="llama-3.1-70b-versatile",  # Fast Groq model
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=150,  # Limit response length for faster generation
        )

        # TTS: Cartesia for low-latency streaming
        tts = cartesia.TTS(
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # Sonic voice
            model="sonic-english",
            api_key=os.getenv("CARTESIA_API_KEY"),
            sample_rate=24000,  # Optimized sample rate
        )

    except Exception as e:
        logger.error(f"Failed to initialize AI services: {e}")
        metrics_logger.log_event(session_id, "error")
        return

    # Create enhanced voice assistant
    assistant = EnhancedVoiceAssistant(
        metrics_logger=metrics_logger,
        session_id=session_id,
        vad=rtc.VAD.for_speech_detection(),
        stt=stt,
        llm=llm_instance,
        tts=tts,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are a helpful AI voice assistant. Keep your responses concise and natural. "
                "Respond in a conversational tone and keep answers brief to maintain low latency. "
                "If interrupted, acknowledge the interruption gracefully."
            ),
        ),
        fnc_ctx=None,  # No function calling for simplicity
        allow_interruptions=True,  # Enable interruption handling
        interrupt_speech_duration=0.5,  # Quick interruption detection
        interrupt_min_words=2,  # Minimum words to trigger interruption
        min_endpointing_delay=0.3,  # Fast endpointing
        preemptive_synthesis=True,  # Start TTS early
    )

    # Start the assistant
    assistant.start(ctx.room)
    logger.info("Voice assistant started - ready for conversation!")

    # Monitor the session
    try:
        await assistant.aclose()
    except Exception as e:
        logger.error(f"Assistant error: {e}")
        metrics_logger.log_event(session_id, "error")
    finally:
        # End session and export metrics
        metrics_logger.end_session(session_id)
        logger.info("Voice assistant session ended")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Validate required environment variables
    required_env_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "GROQ_API_KEY",
        "CARTESIA_API_KEY"
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)

    # Start the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
    )