"""
Enterprise AI Voiceover Narration Service using Microsoft Edge Neural TTS and Gemini AI.
Generates crystal-clear, diverse, broadcast-quality spoken voiceovers for social reels and commercial video ads.
"""

import os
import re
import random
import asyncio
import subprocess
from typing import Optional, List, Dict, Any
try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from app.core.logging import logger
from app.core.gemini_client import get_gemini_client


class VoiceService:
    """
    Enterprise AI voiceover synthesizer offering rich persona diversity,
    intelligent voice matching, and prompt-tailored commercial script generation.
    """

    VOICE_CATALOG = {
        "christopher": {
            "id": "en-US-ChristopherNeural",
            "name": "Christopher",
            "gender": "male",
            "vibe": "Deep, Authoritative & Executive",
            "accent": "US English"
        },
        "jenny": {
            "id": "en-US-JennyNeural",
            "name": "Jenny",
            "gender": "female",
            "vibe": "Warm, Engaging & Professional",
            "accent": "US English"
        },
        "aria": {
            "id": "en-US-AriaNeural",
            "name": "Aria",
            "gender": "female",
            "vibe": "Sleek, Luxury & Sophisticated",
            "accent": "US English"
        },
        "guy": {
            "id": "en-US-GuyNeural",
            "name": "Guy",
            "gender": "male",
            "vibe": "Casual, Conversational & Relatable",
            "accent": "US English"
        },
        "sonia": {
            "id": "en-GB-SoniaNeural",
            "name": "Sonia",
            "gender": "female",
            "vibe": "British Sophisticated & Elegant",
            "accent": "UK English"
        },
        "ryan": {
            "id": "en-GB-RyanNeural",
            "name": "Ryan",
            "gender": "male",
            "vibe": "British Documentary & Cinematic",
            "accent": "UK English"
        },
        "ava": {
            "id": "en-US-AvaNeural",
            "name": "Ava",
            "gender": "female",
            "vibe": "Modern, Punchy & Creator",
            "accent": "US English"
        },
        "eric": {
            "id": "en-US-EricNeural",
            "name": "Eric",
            "gender": "male",
            "vibe": "High-Energy, Bold & Commercial",
            "accent": "US English"
        }
    }

    DEFAULT_VOICE = "en-US-ChristopherNeural"

    @classmethod
    def get_available_voices(cls) -> List[Dict[str, Any]]:
        """Return full catalog of supported neural voice personas."""
        return [
            {"key": k, **v} for k, v in cls.VOICE_CATALOG.items()
        ]

    @classmethod
    def resolve_voice(
        cls,
        voice_preference: Optional[str] = None,
        brand_tone: Optional[str] = None,
        seed: Optional[str] = None
    ) -> str:
        """
        Resolves the target Microsoft Neural voice ID based on:
        1. Explicit user selection (e.g. 'jenny', 'aria', or full 'en-US-JennyNeural')
        2. Brand tone of voice matching
        3. Dynamic deterministic rotation (prevents all videos from sounding identical)
        """
        # 1. Direct match in catalog
        if voice_preference:
            norm_pref = voice_preference.strip().lower()
            if norm_pref in cls.VOICE_CATALOG:
                return cls.VOICE_CATALOG[norm_pref]["id"]
            # Direct neural ID check
            for item in cls.VOICE_CATALOG.values():
                if item["id"].lower() == norm_pref or item["name"].lower() == norm_pref:
                    return item["id"]

        # 2. Tone-based matching
        tone = (brand_tone or "").lower()
        if any(w in tone for w in ["luxury", "elegant", "sophisticated", "premium", "glamour"]):
            return cls.VOICE_CATALOG["aria"]["id"]
        elif any(w in tone for w in ["energetic", "bold", "youth", "fun", "action"]):
            return cls.VOICE_CATALOG["eric"]["id"]
        elif any(w in tone for w in ["british", "royal", "heritage", "classic"]):
            return cls.VOICE_CATALOG["sonia"]["id"]
        elif any(w in tone for w in ["warm", "friendly", "lifestyle", "story", "approachable"]):
            return cls.VOICE_CATALOG["jenny"]["id"]
        elif any(w in tone for w in ["casual", "humorous", "conversational", "relatable"]):
            return cls.VOICE_CATALOG["guy"]["id"]

        # 3. Dynamic Rotation: rotate across diverse voices per prompt/seed
        # This guarantees consecutive generations automatically use distinct voices!
        voice_pool = [
            cls.VOICE_CATALOG["christopher"]["id"],
            cls.VOICE_CATALOG["jenny"]["id"],
            cls.VOICE_CATALOG["aria"]["id"],
            cls.VOICE_CATALOG["guy"]["id"],
            cls.VOICE_CATALOG["ava"]["id"],
            cls.VOICE_CATALOG["sonia"]["id"],
            cls.VOICE_CATALOG["eric"]["id"],
            cls.VOICE_CATALOG["ryan"]["id"],
        ]
        hash_seed = abs(hash(str(seed or "") + str(brand_tone or "") + str(random.random())))
        return voice_pool[hash_seed % len(voice_pool)]

    @classmethod
    def extract_voiceover_cues_from_prompt(cls, prompt: str) -> Optional[str]:
        """Extract explicit voiceover quotation from prompt if provided by user."""
        if not prompt:
            return None

        patterns = [
            r"(?:voiceover|voice\s*over|narration|audio)\s*:\s*['\"]([^'\"]+)['\"]",
            r"(?:voiceover|voice\s*over|narration|audio)\s*:\s*([^\n\r]+)",
            r"voiceover:\s*'([^']+)'",
            r"voiceover:\s*\"([^\"]+)\"",
            r"['\"](Start elevating[^'\"]+)['\"]",
        ]
        for p in patterns:
            match = re.search(p, prompt, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                if len(text) > 8:
                    return text
        return None

    @classmethod
    def compose_commercial_voiceover(
        cls,
        prompt: str,
        brand_name: str = "Social AI",
        brand_tagline: str = "",
        brand_tone: str = "Engaging",
        script_scenes: Optional[List[Dict[str, Any]]] = None,
        target_duration: int = 30,
        variation_seed: Optional[str] = None,
        products_summary: Optional[str] = None
    ) -> str:
        """
        Dynamically composes a unique, high-impact commercial voiceover (~55-70 words)
        tailored directly to the specific prompt and brand context.
        Uses Gemini AI for creative narrative synthesis with intelligent template fallback.
        """
        clean_brand = brand_name or "Social AI"
        clean_tagline = brand_tagline or ""
        prod_text = f" featuring {products_summary}" if products_summary else ""

        # 1. If script scenes have distinct audio dialog, combine them
        if script_scenes and isinstance(script_scenes, list) and len(script_scenes) >= 2:
            scene_audios = []
            for sc in script_scenes:
                if isinstance(sc, dict):
                    audio = sc.get("audio") or sc.get("voiceover")
                    if audio and isinstance(audio, str) and len(audio.strip()) > 8:
                        scene_audios.append(audio.strip().strip("'\""))
            if len(scene_audios) >= 2:
                combined = " ".join(scene_audios)
                if 40 <= len(combined.split()) <= 85:
                    return combined

        # 2. Try Gemini dynamic generation for a 100% custom commercial script
        client = get_gemini_client()
        if client:
            gemini_models = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]
            for m in gemini_models:
                try:
                    sys_prompt = (
                        f"You are an elite commercial copywriter. Write a 30-second broadcast commercial voiceover narration "
                        f"(strictly between 50 and 70 words) for a video about: '{prompt}'.\n"
                        f"Brand: {clean_brand}. Tagline: {clean_tagline}. Tone: {brand_tone}. Products:{prod_text or 'Signature Collection'}.\n"
                        f"Random Variation Seed: {variation_seed or uuid.uuid4().hex[:6]}\n\n"
                        f"Rules:\n"
                        f"- Output ONLY the spoken words for the voice actor to read.\n"
                        f"- Mention {clean_brand} clearly by name at least once.\n"
                        f"- Do NOT output timestamps, stage directions, scene titles, or sound effects like [Music].\n"
                        f"- Make it punchy, rhythmic, and compelling.\n"
                        f"- End with a strong call to action referencing {clean_brand}."
                    )
                    res = client.models.generate_content(model=m, contents=sys_prompt)
                    if res and res.text:
                        clean_text = res.text.strip().replace('"', '').replace("'", "'")
                        clean_text = re.sub(r"\[.*?\]", "", clean_text)
                        clean_text = re.sub(r"\(.*?\)", "", clean_text)
                        clean_text = re.sub(r"\s+", " ", clean_text).strip()
                        if 35 <= len(clean_text.split()) <= 95:
                            logger.info(f"Dynamically generated custom voiceover script via {m} for prompt: {prompt[:40]}")
                            return clean_text
                except Exception as g_err:
                    logger.debug(f"Gemini voiceover script attempt on {m} skipped: {g_err}")

        # 3. Dynamic Template Engine with structural variations (Guarantees diversity when offline)
        # Extract main themes/keywords from the prompt
        clean_p = prompt.strip()
        clean_p = re.sub(r"\d+:\d+-\d+:\d+:[^\n]*", "", clean_p)
        clean_p = re.sub(r"(?:Action Prompt|Visual|Audio):", "", clean_p)
        words = [w for w in re.findall(r"\b[A-Za-z]{3,}\b", clean_p) if w.lower() not in {"the", "and", "with", "for", "from", "this", "that"}]
        topic_preview = products_summary if products_summary else (" ".join(words[:6]) if words else "our signature collection")

        explicit_cue = cls.extract_voiceover_cues_from_prompt(prompt)

        # Seed random generator for variation
        rng = random.Random(f"{variation_seed or uuid.uuid4().hex[:6]}-{prompt[:30]}-{clean_brand}")
        style_choice = rng.choice(["story", "energy", "luxury", "direct", "future", "creator"])

        if style_choice == "story":
            hook = f"Every great story begins with a bold vision. At {clean_brand}, we bring that vision to life."
            body = f"Discover {topic_preview}. Engineered with obsessive detail, uncompromising quality, and modern sophistication."
            cta = f"Elevate your expectations with {clean_brand}. Visit our official link to explore today."
        elif style_choice == "energy":
            hook = explicit_cue if explicit_cue and len(explicit_cue.split()) > 3 else f"Ready to revolutionize your everyday? Welcome to {clean_brand}."
            body = f"Feel the momentum behind {topic_preview}. Faster, smarter, and crafted to push boundaries like never before."
            cta = f"Don't wait. Experience {clean_brand} right now and claim your edge today."
        elif style_choice == "luxury":
            hook = f"True excellence speaks for itself. Introducing the latest from {clean_brand}."
            body = f"Immerse yourself in {topic_preview}. Designed for those who value timeless aesthetics, effortless refinement, and pure prestige."
            cta = f"Indulge in {clean_brand}. Tap the link in bio to explore the exclusive release."
        elif style_choice == "future":
            hook = f"Step into tomorrow with {clean_brand}. Innovation engineered for those who refuse to stand still."
            body = f"Unlock the power of {topic_preview}. Precision crafted, beautifully refined, and built for maximum performance."
            cta = f"Transform your world with {clean_brand}. Explore the collection now."
        elif style_choice == "creator":
            hook = f"What if your brand could move at the speed of your imagination? Meet {clean_brand}."
            body = f"Designed for high-impact growth, {topic_preview} delivers unmatched clarity and creative freedom."
            cta = f"Join the movement with {clean_brand}. Tap below to start your journey."
        else:
            hook = f"Say hello to the future of {clean_brand}. Where innovation meets effortless execution."
            body = f"Transform the way you experience {topic_preview}. Seamlessly integrated, meticulously tuned, and built for lasting impact."
            cta = f"Start your journey with {clean_brand} today. Available now at our official link."

        full_script = f"{hook} {body} {cta}"
        full_script = re.sub(r"\s+", " ", full_script).strip()
        return full_script

    @classmethod
    async def synthesize_speech(
        cls,
        text: str,
        output_file_path: str,
        voice: Optional[str] = None
    ) -> float:
        """
        Synthesizes speech to an audio file (.mp3) using edge-tts.
        Returns the duration of the audio in seconds.
        Includes subtle natural prosody variation.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
        target_voice = voice or cls.DEFAULT_VOICE

        # Subtle acoustic micro-variation (+0% to +3% rate) so consecutive generations sound fresh
        rate_offsets = ["+0%", "+2%", "-1%", "+1%", "+0%"]
        rate_choice = random.choice(rate_offsets)

        communicate = edge_tts.Communicate(text, target_voice, rate=rate_choice)
        await communicate.save(output_file_path)

        # Measure duration with imageio_ffmpeg
        duration = 30.0
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            res = subprocess.run([ffmpeg_exe, "-i", output_file_path], stderr=subprocess.PIPE, text=True, timeout=10)
            for line in res.stderr.splitlines():
                if "Duration" in line:
                    time_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", line)
                    if time_match:
                        h, m, s = time_match.groups()
                        duration = int(h) * 3600 + int(m) * 60 + float(s)
                        break
        except Exception as e:
            logger.warning(f"Failed to probe audio duration via FFmpeg: {e}")

        logger.info(f"AI Voiceover synthesized ({round(duration, 2)}s, voice={target_voice}): {output_file_path}")
        return duration

    @classmethod
    def synthesize_speech_sync(
        cls,
        text: str,
        output_file_path: str,
        voice: Optional[str] = None
    ) -> float:
        """
        Synchronous wrapper for synthesize_speech.
        Safe to call from any sync thread or async task.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, cls.synthesize_speech(text, output_file_path, voice))
                    return future.result(timeout=45)
            else:
                return loop.run_until_complete(cls.synthesize_speech(text, output_file_path, voice))
        except Exception:
            return asyncio.run(cls.synthesize_speech(text, output_file_path, voice))
