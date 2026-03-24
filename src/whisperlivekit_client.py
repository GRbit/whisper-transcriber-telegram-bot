import asyncio
import json
import logging
import os
from typing import Optional, Union

import aiohttp

logger = logging.getLogger(__name__)


class WLKError(Exception):
    pass


def _to_float(t: Union[float, str, None]) -> float:
    """Helper to ensure we have a float duration, supporting WLK string format H:MM:SS.cc"""
    if t is None:
        return 0.0
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        try:
            # Format could be H:MM:SS.cc or H:MM:SS
            parts = t.split(':')
            if len(parts) == 3:
                h = int(parts[0])
                m = int(parts[1])
                s_str = parts[2]
                if '.' in s_str:
                    s_parts = s_str.split('.')
                    s = int(s_parts[0])
                    cs = int(s_parts[1])
                    return h * 3600 + m * 60 + s + cs / 100.0
                else:
                    return h * 3600 + m * 60 + int(s_str)
        except (ValueError, IndexError):
            pass
    return 0.0


def _seconds_to_srt_time(t: Union[float, str, None]) -> str:
    t = _to_float(t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_vtt_time(t: Union[float, str, None]) -> str:
    return _seconds_to_srt_time(t).replace(',', '.')


def _build_txt(segments: list) -> str:
    parts = []
    for seg in segments:
        if seg.get('speaker') == -2:
            continue
        text = seg.get('text', '').strip()
        if text:
            parts.append(text)
    return '\n'.join(parts)


def _build_srt(segments: list) -> str:
    entries = []
    idx = 1
    for seg in segments:
        if seg.get('speaker') == -2:
            continue
        text = seg.get('text', '').strip()
        if not text:
            continue
        start = _seconds_to_srt_time(seg.get('start'))
        end = _seconds_to_srt_time(seg.get('end'))
        entries.append(f"{idx}\n{start} --> {end}\n{text}")
        idx += 1
    return '\n\n'.join(entries)


def _build_vtt(segments: list) -> str:
    lines = ['WEBVTT', '']
    for seg in segments:
        if seg.get('speaker') == -2:
            continue
        text = seg.get('text', '').strip()
        if not text:
            continue
        start = _seconds_to_vtt_time(seg.get('start'))
        end = _seconds_to_vtt_time(seg.get('end'))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append('')
    return '\n'.join(lines)


async def transcribe_via_wlk(
    wlk_url: str,
    audio_path: str,
    timeout: int = 300,
    chunk_size: int = 4096,
    verify_ssl: bool = True,
) -> tuple:
    """Connect to a WhisperLiveKit WebSocket server and transcribe the given audio file.

    Returns (txt_content, srt_content, vtt_content).
    """
    if not os.path.exists(audio_path):
        raise WLKError(f"Audio file not found: {audio_path}")

    # Ensure endpoint is correctly formatted
    endpoint = wlk_url.rstrip('/')
    if not endpoint.endswith('/asr'):
        endpoint += '/asr'
        
    ssl_param = None if verify_ssl else False
    responses = []

    async def _send_audio(ws):
        with open(audio_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                await ws.send_bytes(chunk)
        
        # Signal end of audio - CRITICAL for WLK to finish processing
        await ws.send_bytes(b"")
        logger.info("Audio transmission to WLK complete, sent EOF signal")

    async def _recv_messages(ws):
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get('type')

                if msg_type == 'ready_to_stop':
                    logger.info("WLK sent ready_to_stop")
                    return

                if msg_type == 'config':
                    continue
                
                responses.append(data)

            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                return

    try:
        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.ws_connect(endpoint, ssl=ssl_param) as ws:
                logger.info(f"Connected to WLK WebSocket at {endpoint}")
                
                send_task = asyncio.create_task(_send_audio(ws))
                recv_task = asyncio.create_task(_recv_messages(ws))
                
                await asyncio.wait_for(
                    asyncio.gather(send_task, recv_task),
                    timeout=float(timeout),
                )
    except asyncio.TimeoutError:
        raise WLKError(f"WLK transcription timed out after {timeout}s")
    except Exception as e:
        raise WLKError(f"WLK transcription error: {e}")

    # Process responses to get final segments
    segments = []
    if responses:
        # Get the latest response with 'lines'
        for resp in reversed(responses):
            if 'lines' in resp and resp['lines']:
                segments = resp['lines']
                break
        
        if not segments:
            # Fallback to last buffer if no lines were committed
            buffer = responses[-1].get('buffer_transcription', '')
            if buffer:
                segments = [{'text': buffer, 'start': 0.0, 'end': 0.0, 'speaker': 1}]

    if not segments:
        logger.warning("WLK returned no transcription results")
        return '', '', ''

    return _build_txt(segments), _build_srt(segments), _build_vtt(segments)
