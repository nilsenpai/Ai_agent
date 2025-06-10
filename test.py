#!/usr/bin/env python3

"""
Test script to validate the voice agent setup and API connections
"""

import os
import asyncio
import time
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup-test")


async def test_api_connections():
    """Test all API connections"""

    print("🔍 Testing API Connections...\n")

    # Load environment variables
    load_dotenv()

    # Test required environment variables
    required_vars = [
        "LIVEKIT_URL"