from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioMeta:
    sample_rate: int
    bit_rate_kbps: int = 128


class PodcastAudioGenerator:
    """Text-to-speech generator producing an MP3 suitable for in-browser playback."""

    def __init__(self, coqui_model: Optional[str] = None):
        self.coqui_model = coqui_model

    def synthesize_to_mp3(self, text: str, out_path: Path, meta: Optional[AudioMeta] = None) -> AudioMeta:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Improve pronunciation of abbreviations (e.g., AI, B2B) for TTS.
        try:
            import os

            normalize = os.getenv("POST_ANALYSIS_TTS_NORMALIZE_ABBREVIATIONS", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        except Exception:
            normalize = True

        if normalize:
            text = _normalize_for_tts(text)

        # Allow a lightweight test stub without Coqui installed.
        stub = ("POST_ANALYSIS_TTS_PROVIDER" in __import__("os").environ and __import__("os").getenv("POST_ANALYSIS_TTS_PROVIDER") == "stub")
        if stub:
            fake = b"ID3" + (b"\x00" * 128) + text.encode("utf-8")[:256]
            out_path.write_bytes(fake)
            return AudioMeta(sample_rate=22050)

        try:
            from TTS.api import TTS
        except Exception as e:
            raise ImportError("Missing dependency: Coqui TTS. Install `TTS`. ") from e

        try:
            import lameenc
        except Exception as e:
            raise ImportError("Missing dependency: lameenc (MP3 encoder). Install `lameenc`. ") from e

        try:
            import numpy as np
        except Exception as e:
            raise ImportError("Missing dependency: numpy. Install `numpy`. ") from e

        model_name = self.coqui_model
        if not model_name:
            import os
            model_name = os.getenv("COQUI_TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)

        # Simple pacing: paragraphs become separate synth calls.
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        waves = []
        for p in parts:
            waves.append(np.asarray(tts.tts(p), dtype=np.float32))
            # 250ms pause
            sr = int(getattr(tts.synthesizer, "output_sample_rate", 22050))
            waves.append(np.zeros(int(sr * 0.25), dtype=np.float32))

        sr = int(getattr(tts.synthesizer, "output_sample_rate", 22050))
        wave = np.concatenate(waves) if waves else np.zeros(int(sr * 0.2), dtype=np.float32)
        wave = np.clip(wave, -1.0, 1.0)
        pcm16 = (wave * 32767.0).astype(np.int16).tobytes()

        encoder = lameenc.Encoder()
        encoder.set_bit_rate((meta.bit_rate_kbps if meta else 128))
        encoder.set_in_sample_rate(sr)
        encoder.set_channels(1)
        encoder.set_quality(2)

        mp3 = encoder.encode(pcm16)
        mp3 += encoder.flush()

        out_path.write_bytes(mp3)
        return AudioMeta(sample_rate=sr, bit_rate_kbps=(meta.bit_rate_kbps if meta else 128))


_ABBREVIATION_MAP: dict[str, str] = {
    # Common domain abbreviations where letter-by-letter is clearer.
    # Use punctuation to strongly cue letter-by-letter pronunciation in TTS.
    "AI": "A. I.",
    "B2B": "B. 2. B.",
    "B2C": "B. 2. C.",
    "ROI": "R. O. I.",
    "KPI": "K. P. I.",
    "CRM": "C. R. M.",
    "LTV": "L. T. V.",
    "CAC": "C. A. C.",
    "SLA": "S. L. A.",
    "OKR": "O. K. R.",
}


def _normalize_for_tts(text: str) -> str:
    """Normalize text to improve TTS pronunciation.

    Coqui models sometimes pronounce short all-caps tokens as a word (e.g., "AI" -> "aye").
    This function rewrites common patterns to a form that is typically spoken correctly.
    """

    if not text:
        return text

    # Normalize B2B/B2C-like tokens even if written with dashes/spaces.
    # Examples: B2B, B-2-B, b 2 b
    def _b2x(m: re.Match[str]) -> str:
        # Punctuation helps TTS spell it out.
        return f"{m.group(1).upper()}. {m.group(2)}. {m.group(3).upper()}."

    out = re.sub(r"\b([A-Za-z])\s*[-–—]?\s*(\d)\s*[-–—]?\s*([A-Za-z])\b", _b2x, text)

    # Apply explicit replacements first.
    for abbr, spoken in _ABBREVIATION_MAP.items():
        out = re.sub(rf"\b{re.escape(abbr)}\b", spoken, out, flags=re.IGNORECASE)

    # Generic rule: turn 2-6 letter all-caps tokens into spaced letters.
    # This helps with things like "USA", "API", "SQL".
    # Keep it conservative to avoid mangling longer words.
    def _spell(m: re.Match[str]) -> str:
        token = m.group(0)
        # If already handled by explicit map, don't touch.
        if token in _ABBREVIATION_MAP:
            return _ABBREVIATION_MAP[token]

        # Use dotted spelling to strongly encourage letter-by-letter.
        return ". ".join(token) + "."

    out = re.sub(r"\b[A-Z]{2,6}\b", _spell, out)
    return out
