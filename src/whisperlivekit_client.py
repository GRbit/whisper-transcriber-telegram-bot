import asyncio
import json
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class WLKError(Exception):
    pass


def _seconds_to_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_vtt_time(t: float) -> str:
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
        start = _seconds_to_srt_time(seg.get('start', 0.0))
        end = _seconds_to_srt_time(seg.get('end', 0.0))
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
        start = _seconds_to_vtt_time(seg.get('start', 0.0))
        end = _seconds_to_vtt_time(seg.get('end', 0.0))
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

    endpoint = wlk_url.rstrip('/') + '/asr'
    ssl_param = None if verify_ssl else False

    # For legacy API: track the latest complete lines snapshot.
    # For new incremental API: accumulate segments by ID.
    final_lines: list = []
    segments_by_id: dict = {}
    use_legacy: Optional[bool] = None

    async def _send_audio(ws):
        with open(audio_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                await ws.send_bytes(chunk)
        logger.info("Audio transmission to WLK complete")

    async def _recv_messages(ws):
        nonlocal use_legacy, final_lines
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get('type')

                if msg_type == 'ready_to_stop':
                    logger.info("WLK sent ready_to_stop")
                    return

                if msg_type == 'config':
                    logger.info(f"WLK config received: {data}")
                    continue

                # Legacy API sends full state snapshots in 'lines'.
                if 'lines' in data and use_legacy is not False:
                    use_legacy = True
                    final_lines = data['lines']
                elif 'segments' in data:
                    # New incremental API: merge by segment ID.
                    use_legacy = False
                    for seg in data['segments']:
                        seg_id = seg['id']
                        if seg_id not in segments_by_id:
                            segments_by_id[seg_id] = dict(seg)
                        else:
                            existing = segments_by_id[seg_id]
                            existing['text'] = existing.get('text', '') + seg.get('text', '')
                            if seg.get('translation'):
                                existing['translation'] = existing.get('translation', '') + seg['translation']
                            if seg.get('end') is not None:
                                existing['end'] = seg['end']
                            if seg.get('language'):
                                existing['language'] = seg['language']

            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                logger.info(f"WLK WebSocket {msg.type.name.lower()}")
                return

    try:
        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.ws_connect(endpoint, ssl=ssl_param) as ws:
                logger.info(f"Connected to WLK WebSocket at {endpoint}")
                await asyncio.wait_for(
                    asyncio.gather(_send_audio(ws), _recv_messages(ws)),
                    timeout=float(timeout),
                )
    except asyncio.TimeoutError:
        raise WLKError(f"WLK transcription timed out after {timeout}s")
    except aiohttp.ClientError as e:
        raise WLKError(f"WLK WebSocket connection failed: {e}")
    except WLKError:
        raise
    except Exception as e:
        raise WLKError(f"Unexpected error during WLK transcription: {e}")

    if use_legacy:
        segments = final_lines
    elif segments_by_id:
        segments = sorted(segments_by_id.values(), key=lambda s: s.get('start', 0.0))
    else:
        logger.warning("WLK returned no transcription segments")
        return '', '', ''

    return _build_txt(segments), _build_srt(segments), _build_vtt(segments)
