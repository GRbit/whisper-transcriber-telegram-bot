import asyncio
import base64
import os
import logging
from typing import Dict, List

import websockets

logger = logging.getLogger(__name__)

async def transcribe_audio(audio_file_path: str) -> str:
    async def connect_websocket():
        uri = "ws://localhost:8000/asr"
        async with websockets.connect(uri) as websocket:
            # Send audio file in chunks
            with open(audio_file_path, 'rb') as audio_file:
                while chunk := audio_file.read(4096):
                    await websocket.send(chunk)

            # Process transcription results
            transcription = ""
            async for message in websocket:
                try:
                    if isinstance(message, dict) and message.get('type') == 'segments':
                        for segment in message['segments']:
                            if 'text' in segment:
                                transcription += segment['text']
                    elif message.get('type') == 'ready_to_stop':
                        logger.info("Transcription complete")
                        return transcription
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    return await connect_websocket()