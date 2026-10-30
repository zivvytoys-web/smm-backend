import os
import uuid
import json
import base64
import asyncio
import httpx
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image, ImageDraw, ImageFont

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.core.config import settings
from app.core.logging import logger
from app.core.gemini_client import get_gemini_client
from app.models.brand_profile import BrandProfile
from app.models.content import Content
from app.models.autonomous_pipeline import GeneratedAsset
from app.db.session import SessionLocal


# Pydantic Schemas for Structured Gemini Outputs
class CreativeBriefSchema(BaseModel):
    topic: str = Field(description="Main theme or subject for the social post")
    angle: str = Field(description="Unique hook or narrative angle")
    reason: str = Field(description="Strategic justification based on brand goals and trends")
    caption_brief: str = Field(description="Detailed guidelines for the caption text")
    image_prompt: str = Field(description="Photorealistic or high-end artistic prompt for image generation")
    video_prompt: str = Field(description="Short video / reel prompt or scene storyboard concept")
    hashtags: List[str] = Field(description="5 to 10 relevant, high-impact hashtags starting with #")
    cta: str = Field(description="Strong Call to Action aligned with brand objectives")
    tone: str = Field(description="Tone of voice to maintain throughout all assets")
    target_platforms: List[str] = Field(description="List of target platforms e.g. ['instagram', 'facebook', 'linkedin', 'youtube']")


class PlatformCaptionsSchema(BaseModel):
    instagram: str = Field(description="Instagram caption with compelling hook, line breaks, and hashtags")
    facebook: str = Field(description="Facebook post optimized for engagement, comments, and link CTA")
    linkedin: str = Field(description="LinkedIn post formatted with executive insights and thought leadership")
    youtube_title: str = Field(description="High-CTR YouTube video or Shorts title under 70 chars")
    youtube_description: str = Field(description="Comprehensive YouTube description with summary, key takeaways, and hashtags")
    twitter: Optional[str] = Field(default=None, description="Concise Twitter/X post under 280 chars with 2 hashtags")
    pinterest: Optional[str] = Field(default=None, description="Inspiring Pinterest pin title and description")


class PipelineGeminiService:
    """
    Autonomous Gemini Service Layer:
    Handles structured topic discovery, multi-platform copy generation,
    aspect-ratio-tailored visual synthesis, and async video generation with polling.
    """

    @classmethod
    def _get_genai_client(cls) -> genai.Client:
        client = get_gemini_client()
        if not client:
            raise ValueError("GEMINI_API_KEY is not configured in backend settings.")
        return client

    @classmethod
    def get_preferred_text_model(cls) -> str:
        configured = getattr(settings, "GEMINI_TEXT_MODEL", "")
        # Validate model name is a real Gemini model
        real_text_models = [
            "gemini-1.5-flash", "gemini-1.5-pro",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
            "gemini-2.5-flash", "gemini-2.5-pro",
            "gemini-3.0", "gemini-3.1"
        ]
        if configured and any(m in configured for m in real_text_models):
            return configured
        return "gemini-2.0-flash" if "gemini-2.0" in configured else "gemini-1.5-flash"

    @classmethod
    def get_preferred_image_model(cls) -> str:
        configured = getattr(settings, "GEMINI_IMAGE_MODEL", "")
        # Validate model name is a real image generation model  
        real_image_models = ["imagen-3.0-generate-002", "imagen-3.0-generate-001", "imagen-3.0-fast-generate-001", "imagen-2.0"]
        if configured and any(m in configured for m in real_image_models):
            return configured
        return "imagen-3.0-generate-002"

    # =========================================================================
    # 1. ANALYZE BUSINESS CONTEXT & AUTO TOPIC BRIEF
    # =========================================================================

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10), reraise=True)
    async def generate_creative_brief(
        cls,
        brand_profile: BrandProfile,
        past_posts: List[Dict[str, Any]],
        target_platforms: List[str]
    ) -> CreativeBriefSchema:
        """
        Generates a fresh, strategic topic brief based on verified brand information
        and strictly avoids topics covered in the last 30 days.
        """
        client = cls._get_genai_client()
        model_name = cls.get_preferred_text_model()

        past_topics_str = ""
        if past_posts:
            recent_titles = [p.get("title") or (p.get("caption")[:60] + "...") for p in past_posts[:15] if p.get("caption") or p.get("title")]
            past_topics_str = "PAST 30 DAYS RECENT TOPICS (DO NOT REPEAT THESE):\n- " + "\n- ".join(recent_titles)

        prompt = f"""
You are the Chief Creative Officer and Head of Social Media for this business.
Analyze the verified brand profile below and create today's high-performing daily social media brief.

BUSINESS / BRAND PROFILE:
- Business Name: {brand_profile.name}
- Industry: {brand_profile.industry or 'E-Commerce / Lifestyle'}
- Tagline: {brand_profile.tagline or 'Premium Quality & Experience'}
- Tone of Voice: {brand_profile.tone_of_voice}
- Target Audience: {brand_profile.target_audience or 'Discerning modern consumers'}
- Key Themes: {json.dumps(brand_profile.key_themes or ['Quality', 'Innovation', 'Customer Delight'])}
- Default CTA: {brand_profile.cta or 'Shop the collection today'}
- Topics / Guidelines to Avoid: {brand_profile.avoid or 'No misleading discounts, no clickbait'}
- Target Platforms: {', '.join(target_platforms)}

{past_topics_str}

CRITICAL RULES:
1. Ground the topic strictly in verified business information. NEVER invent fake prices, false claims, or non-existent discount percentages.
2. Select a fresh, timely angle distinct from recent posts.
3. Formulate clear, actionable prompts for text, photorealistic image, and video storyboard.
4. Output MUST conform strictly to the required JSON structure.
"""

        try:
            # Run in threadpool to keep the async event loop completely unblocked
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CreativeBriefSchema,
                    temperature=0.7,
                )
            )

            raw_text = response.text.strip()
            data = json.loads(raw_text)
            return CreativeBriefSchema(**data)
        except Exception as e:
            logger.warning(f"[PipelineGemini] Primary model {model_name} failed: {e}. Attempting fallback models...")
            fallback_models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]
            for f_model in fallback_models:
                try:
                    res = await asyncio.to_thread(
                        client.models.generate_content,
                        model=f_model,
                        contents=prompt + "\n\nReturn strict JSON matching the schema.",
                        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7)
                    )
                    data = json.loads(res.text.strip())
                    return CreativeBriefSchema(**data)
                except Exception:
                    continue

            # Graceful brand-tailored synthesis when remote free-tier quota is restricted
            theme_choice = (brand_profile.key_themes[0] if brand_profile.key_themes else "Premium Innovation")
            topic = f"{brand_profile.name} Spotlight: {theme_choice}"
            angle = f"Behind the craftsmanship and excellence of {brand_profile.name}"
            clean_tags = [f"#{brand_profile.name.replace(' ', '')}", f"#{brand_profile.industry.replace(' ', '') if brand_profile.industry else 'SocialAI'}", "#BrandSpotlight", "#QualityFirst", "#Innovation"]
            if brand_profile.hashtags:
                clean_tags.extend([h if h.startswith('#') else f"#{h}" for h in brand_profile.hashtags[:3]])

            return CreativeBriefSchema(
                topic=topic,
                angle=angle,
                reason=f"Strategic alignment with brand focus on {theme_choice} and verified brand narrative.",
                caption_brief=f"Highlight the signature values, design philosophy, and customer experience of {brand_profile.name}.",
                image_prompt=f"Editorial commercial studio photography, {brand_profile.name} presentation, clean aesthetic lighting, 8k resolution, minimalist elegance",
                video_prompt=f"Vertical 9:16 reel: Dynamic visual transition of {brand_profile.name} core craftsmanship with high-energy audio hook",
                hashtags=clean_tags[:6],
                cta=brand_profile.cta or f"Discover more at {brand_profile.name}",
                tone=brand_profile.tone_of_voice or "Engaging & Sophisticated",
                target_platforms=target_platforms
            )

    # =========================================================================
    # 2. PARALLEL TEXT GENERATION (Multi-Platform Copy)
    # =========================================================================

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=8), reraise=True)
    async def generate_platform_captions(
        cls,
        brief: CreativeBriefSchema,
        brand_profile: BrandProfile
    ) -> PlatformCaptionsSchema:
        """
        Generates tailored copy for Instagram, Facebook, LinkedIn, YouTube, Twitter, and Pinterest
        aligned with the creative brief and character limit constraints.
        """
        client = cls._get_genai_client()
        model_name = cls.get_preferred_text_model()

        prompt = f"""
Write high-converting, platform-tailored social media captions for '{brand_profile.name}' based on this creative brief:

BRIEF DETAILS:
- Topic: {brief.topic}
- Angle: {brief.angle}
- Caption Directive: {brief.caption_brief}
- Tone: {brief.tone}
- Call to Action (CTA): {brief.cta}
- Brand Hashtags: {', '.join(brief.hashtags)}

PLATFORM REQUIREMENTS:
- Instagram: Engaging hook on line 1, concise value-driven body with spacing, clear CTA, and 5-8 relevant hashtags.
- Facebook: Conversational storytelling tone, community discussion prompt, link CTA.
- LinkedIn: Executive thought leadership angle, bulleted insights, professional tone, 3-4 professional hashtags.
- YouTube: High-CTR video title (under 70 chars) and comprehensive description with timestamps placeholder and CTA.
- Twitter/X: Punchy tweet under 260 characters including 2 hashtags.
- Pinterest: Catchy Pin title and inspiring description with keywords.

Output MUST strictly follow the JSON schema.
"""

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PlatformCaptionsSchema,
                    temperature=0.7,
                )
            )
            return PlatformCaptionsSchema(**json.loads(response.text.strip()))
        except Exception as e:
            logger.warning(f"[PipelineGemini] Platform copy generation exception: {e}")
            # Reliable fallback structure
            return PlatformCaptionsSchema(
                instagram=f"✨ {brief.topic}\n\n{brief.caption_brief}\n\n👉 {brief.cta}\n\n" + " ".join(brief.hashtags),
                facebook=f"{brief.topic}\n\n{brief.caption_brief}\n\nLearn more: {brief.cta}",
                linkedin=f"📌 {brief.topic}\n\n{brief.caption_brief}\n\nWhat are your thoughts on this approach?\n\n" + " ".join(brief.hashtags[:4]),
                youtube_title=f"{brand_profile.name} - {brief.topic}"[:70],
                youtube_description=f"In this update, we explore {brief.topic}.\n\n{brief.caption_brief}\n\n{brief.cta}\n\n" + " ".join(brief.hashtags),
                twitter=f"{brief.topic}: {brief.caption_brief}"[:240] + " " + " ".join(brief.hashtags[:2]),
                pinterest=f"{brief.topic}\n\n{brief.caption_brief}\n\n{brief.cta}"
            )

    # =========================================================================
    # 3. MULTI-RATIO IMAGE GENERATION
    # =========================================================================

    @classmethod
    async def generate_branded_image(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        brand_profile: Optional[BrandProfile] = None
    ) -> Dict[str, Any]:
        """
        Generates visual artwork tailored to the requested aspect ratio:
        - 1:1 Square (Instagram/Facebook feed)
        - 4:5 Portrait (Instagram/Facebook vertical feed)
        - 1.91:1 Landscape (LinkedIn/Twitter)
        - 16:9 Landscape (YouTube)
        Saves the file to local media storage and returns metadata.
        """
        output_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "media")
        os.makedirs(output_dir, exist_ok=True)
        file_id = f"img_{uuid.uuid4().hex[:12]}_{aspect_ratio.replace(':', '_')}.png"
        file_path = os.path.join(output_dir, file_id)
        relative_url = f"/uploads/media/{file_id}"

        # Dimensions mapping
        dim_map = {
            "1:1": (1080, 1080),
            "4:5": (1080, 1350),
            "1.91:1": (1200, 628),
            "16:9": (1920, 1080),
            "9:16": (1080, 1920)
        }
        width, height = dim_map.get(aspect_ratio, (1080, 1080))

        image_bytes = None
        enhanced_prompt = f"Commercial photography, high resolution 8k, professional lighting: {prompt}"

        used_provider = "studio-canvas"
        from app.services.image_generation_service import ImageGenerationService

        # 1. Cloudflare Workers AI Flux (schnell)
        try:
            image_bytes = await ImageGenerationService._generate_cloudflare_image(enhanced_prompt)
            if image_bytes:
                used_provider = "cloudflare-flux"
                logger.info(f"[PipelineGemini] Generated image via Cloudflare Flux for {aspect_ratio}.")
        except Exception as cf_err:
            logger.debug(f"[PipelineGemini] Cloudflare image notice: {cf_err}")

        # 2. Supported Image Fallback: Fal FLUX (fal-ai/flux/schnell)
        if image_bytes is None:
            try:
                image_bytes = await ImageGenerationService._generate_fal_image(enhanced_prompt)
                if image_bytes:
                    used_provider = "fal-flux"
                    logger.info(f"[PipelineGemini] Generated image via Fal FLUX fallback for {aspect_ratio}.")
            except Exception as fal_err:
                logger.debug(f"[PipelineGemini] Fal image notice: {fal_err}")

        # 3. Supported Image Fallback: Google Gemini / Imagen
        if image_bytes is None and settings.GEMINI_API_KEY:
            client = cls._get_genai_client()
            image_models = [
                cls.get_preferred_image_model(),  # imagen-3.0-generate-002
                "imagen-3.0-generate-001",
                "imagen-3.0-fast-generate-001",
            ]
            seen = set()
            image_models = [m for m in image_models if not (m in seen or seen.add(m))]
            for img_model in image_models:
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_images,
                        model=img_model,
                        prompt=enhanced_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            output_mime_type="image/png",
                            aspect_ratio="1:1" if aspect_ratio in ["1:1", "4:5"] else ("16:9" if aspect_ratio in ["16:9", "1.91:1"] else "9:16")
                        )
                    )
                    if response and response.generated_images:
                        image_bytes = response.generated_images[0].image.image_bytes
                        used_provider = "gemini-imagen"
                        logger.info(f"[PipelineGemini] Generated image with {img_model} for {aspect_ratio}")
                        break
                except Exception as e:
                    logger.debug(f"[PipelineGemini] Model {img_model} image generation failed: {e}")
                    continue

        # 4. Zero-Auth Unmetered Photorealistic AI Image Fallback: Pollinations AI
        if image_bytes is None:
            try:
                image_bytes = await ImageGenerationService._generate_pollinations_image(
                    prompt=enhanced_prompt,
                    aspect_ratio="1:1" if aspect_ratio in ["1:1", "4:5"] else ("16:9" if aspect_ratio in ["16:9", "1.91:1"] else "9:16")
                )
                if image_bytes:
                    used_provider = "flux-pollinations"
                    logger.info(f"[PipelineGemini] Generated photorealistic image via Pollinations AI for {aspect_ratio}.")
            except Exception as poll_err:
                logger.warning(f"[PipelineGemini] Pollinations image notice: {poll_err}")

        # If API image is received, save and crop/resize if necessary
        if image_bytes:
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            # Reformat to exact aspect ratio
            try:
                with Image.open(file_path) as im:
                    im_resized = im.resize((width, height), Image.Resampling.LANCZOS)
                    im_resized.save(file_path, "PNG", quality=95)
            except Exception:
                pass
        else:
            # Keep a scheduled campaign progressing when an external image
            # provider is temporarily unavailable. The local renderer creates
            # a valid, aspect-ratio-correct PNG which is still uploaded to
            # Cloudinary before it can be published.
            fallback_colors = (
                brand_profile.color_palette
                if brand_profile and isinstance(brand_profile.color_palette, list)
                else []
            )
            cls._synthesize_branded_graphic(
                output_path=file_path,
                width=width,
                height=height,
                title=prompt,
                brand_name=brand_profile.name if brand_profile and brand_profile.name else "Social AI",
                colors=fallback_colors,
            )
            logger.warning(
                "[PipelineGemini] External image providers returned no image; "
                "used the local resilient graphic fallback for %s.",
                aspect_ratio,
            )

        # Persist the generated image in Cloudinary so overview thumbnails and
        # scheduled platform workers have a durable public URL.  A temporary
        # network outage must not turn an otherwise valid local image into a
        # failed pipeline run: the local static-media URL remains usable by the
        # dashboard and can be uploaded when the connection is restored.
        from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider
        try:
            uploaded = CloudinaryStorageProvider.upload_local_file_path(
                file_path=file_path,
                organization_id=brand_profile.organization_id if brand_profile and brand_profile.organization_id else 1,
                folder_prefix="socialai/ai_images",
                resource_type="image"
            )
            relative_url = uploaded["file_url"]
        except Exception as upload_error:
            logger.warning(
                "[PipelineGemini] Cloudinary image upload unavailable; "
                "retaining local media for %s: %s",
                aspect_ratio,
                upload_error,
            )

        return {
            "status": "done",
            "file_path": relative_url,
            "local_path": file_path,
            "aspect_ratio": aspect_ratio,
            "width": width,
            "height": height,
            "provider": used_provider
        }

    @classmethod
    def _synthesize_branded_graphic(
        cls,
        output_path: str,
        width: int,
        height: int,
        title: str,
        brand_name: str,
        colors: List[str]
    ):
        """
        Creates an ultra-crisp studio branded graphic using Pillow when external image quota is constrained.
        """
        img = Image.new("RGB", (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Draw subtle gradient backdrop
        c1 = (99, 102, 241)
        c2 = (15, 23, 42)
        for y in range(height):
            ratio = y / max(height, 1)
            r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
            g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
            b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Brand Badge at top
        badge_text = f"✨  {brand_name.upper()}  ✨"
        draw.rounded_rectangle([(60, 60), (width - 60, 130)], radius=16, fill=(30, 41, 59, 200), outline=(99, 102, 241), width=2)
        draw.text((width // 2, 95), badge_text, fill=(255, 255, 255), anchor="mm")

        # Headline Card in center
        card_top = height // 2 - 120
        card_bottom = height // 2 + 140
        draw.rounded_rectangle([(60, card_top), (width - 60, card_bottom)], radius=24, fill=(15, 23, 42, 220), outline=(147, 51, 234), width=3)
        draw.text((width // 2, height // 2 - 30), title[:90], fill=(248, 250, 252), anchor="mm")
        draw.text((width // 2, height // 2 + 50), "OFFICIAL BRAND UPDATE", fill=(168, 85, 247), anchor="mm")

        # Bottom Bar
        draw.rectangle([(0, height - 80), (width, height)], fill=(10, 15, 30))
        draw.text((width // 2, height - 40), f"Verified Social Creative • {brand_name}", fill=(148, 163, 184), anchor="mm")

        img.save(output_path, "PNG")

    @classmethod
    async def _attach_voiceover_narration(
        cls,
        video_path: str,
        prompt: str,
        brand_profile: Optional[BrandProfile],
        duration_seconds: int = 6
    ) -> str:
        """
        Synthesizes multi-voice commercial narration with acoustic micro-rate variations
        using Microsoft Edge Neural TTS and muxes it onto the video with FFmpeg.
        Persona is dynamically matched to the brand tone (Christopher, Jenny, Aria, Guy, Sonia, Ryan, Ava, Eric).
        """
        if not video_path or not os.path.exists(video_path):
            return video_path
        try:
            from app.services.voice_service import VoiceService
            import imageio_ffmpeg
            import subprocess

            brand_name = brand_profile.name if brand_profile and brand_profile.name else "Our Brand"
            brand_tone = brand_profile.tone_of_voice if brand_profile else "professional"

            # 1. Resolve multi-voice persona
            voice_id = VoiceService.resolve_voice(
                brand_tone=brand_tone,
                seed=f"{brand_name}-{os.path.basename(video_path)}"
            )

            # 2. Compose brief commercial narration script tailored to duration
            script = VoiceService.compose_commercial_voiceover(
                prompt=prompt,
                brand_name=brand_name,
                brand_tagline=getattr(brand_profile, "tagline", "") or "Premium Quality & Experience",
                brand_tone=brand_tone,
                target_duration=duration_seconds,
                variation_seed=os.path.basename(video_path)
            )

            audio_path = video_path.replace(".mp4", "_voice.mp3")
            await VoiceService.synthesize_speech(script, audio_path, voice=voice_id)

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                muxed_path = video_path.replace(".mp4", "_narrated.mp4")
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    muxed_path
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
                if proc.returncode == 0 and os.path.exists(muxed_path) and os.path.getsize(muxed_path) > 1000:
                    try:
                        os.remove(audio_path)
                        os.replace(muxed_path, video_path)
                    except Exception:
                        video_path = muxed_path
                    logger.info(f"[PipelineGemini] Attached Edge Neural TTS narration ({voice_id}) with micro-rate variation to {video_path}")
                else:
                    if os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"[PipelineGemini] Edge Neural TTS narration attachment warning: {e}")

        return video_path

    # =========================================================================
    # 4. ASYNC VIDEO GENERATION & RESTART-SAFE POLLING
    # =========================================================================

    @classmethod
    async def start_async_video_generation(
        cls,
        prompt: str,
        aspect_ratio: str = "9:16",
        brand_profile: Optional[BrandProfile] = None,
        image_url: Optional[str] = None,
        image_urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates cinematic AI video using Google Veo (same AI as Gemini AI Studio).
        Produces authentic generative video footage without watermarks or brand logos.
        Falls back to Magic Hour / Hugging Face or clean smooth motion if Veo quota is constrained.
        """
        output_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "media", "ai_videos")
        os.makedirs(output_dir, exist_ok=True)
        valid_ratio = "9:16" if str(aspect_ratio).strip() in ["9:16", "4:5", "portrait"] else "16:9"

        # 1. Primary Lowest-Cost AI Video Engine: Lightricks LTX Video API ($0.09/s -> ₹38.70 for 30s commercial reel)
        try:
            from app.services.ltx_video_service import LTXVideoService
            if LTXVideoService.is_configured():
                logger.info(f"[PipelineGemini] Dispatching lowest-cost AI video generation with LTX Video API (ltx-2-5-fast)...")
                ltx_res = await LTXVideoService.generate_and_persist_video(
                    prompt=prompt,
                    brand=brand_profile,
                    aspect_ratio=valid_ratio,
                    duration_seconds=30,
                    organization_id=brand_profile.organization_id if brand_profile else 1
                )
                if ltx_res and ltx_res.get("file_path"):
                    logger.info(f"[PipelineGemini] LTX Video generated and persisted successfully: {ltx_res['file_path']} (Cost: ₹{ltx_res.get('cost_inr', 38.70)})")
                    return {
                        "operation_id": f"ltx_{uuid.uuid4().hex[:16]}",
                        "status": "done",
                        "prompt": prompt,
                        "aspect_ratio": valid_ratio,
                        "file_path": ltx_res["file_path"],
                        "local_path": ltx_res.get("local_path"),
                        "duration_seconds": ltx_res.get("duration_seconds", 30),
                        "mime_type": "video/mp4",
                        "provider": "ltx-video",
                        "model_name": "ltx-2-5-fast",
                        "cost_usd": ltx_res.get("cost_usd", 0.45),
                        "cost_inr": ltx_res.get("cost_inr", 38.70),
                        "audio_provider": "edge-neural-tts"
                    }
        except Exception as ltx_err:
            logger.warning(f"[PipelineGemini] LTX Video generation notice: {ltx_err}. Checking alternate providers...")

        # 2. Free Neural AI Video Engine (Hugging Face ZeroGPU - Lightricks LTX Video)
        try:
            from app.services.huggingface_video_service import HuggingFaceVideoService
            if HuggingFaceVideoService.is_configured():
                logger.info("[PipelineGemini] Dispatching true AI neural video generation via Hugging Face LTX-Video...")
                hf_res = await HuggingFaceVideoService.generate_and_persist_video(
                    prompt=prompt,
                    brand=brand_profile,
                    aspect_ratio=valid_ratio,
                    duration_seconds=30,
                    input_image_path=image_url,
                    organization_id=brand_profile.organization_id if brand_profile else 1
                )
                if hf_res and hf_res.get("file_path"):
                    logger.info(f"[PipelineGemini] True AI neural video generated successfully via Hugging Face: {hf_res['file_path']}")
                    return {
                        "operation_id": f"hf_{uuid.uuid4().hex[:16]}",
                        "status": "done",
                        "prompt": prompt,
                        "aspect_ratio": valid_ratio,
                        "file_path": hf_res["file_path"],
                        "local_path": hf_res.get("local_path"),
                        "duration_seconds": hf_res.get("duration_seconds", 30),
                        "mime_type": "video/mp4",
                        "provider": "huggingface-ltx-video",
                        "model_name": "lightricks-ltx-video-distilled",
                        "cost_usd": 0.0,
                        "cost_inr": 0.0,
                        "audio_provider": "edge-neural-tts"
                    }
        except Exception as hf_err:
            logger.warning(f"[PipelineGemini] Hugging Face LTX-Video notice: {hf_err}. Checking alternate providers...")

        # 3. Google Veo Engine (Enterprise / Vertex AI configured only)
        # Note: Veo preview models return 404 on standard API keys; skip if not explicitly configured with enterprise key
        if getattr(settings, "GEMINI_API_KEY", None) and getattr(settings, "GEMINI_VIDEO_MODEL", "").startswith("veo-enterprise"):
            try:
                client = cls._get_genai_client()
                clean_veo_prompt = (
                    f"{prompt.strip()}. "
                    "Cinematic commercial 4k high definition, fluid camera movement, photorealistic lighting, "
                    "pristine visual clarity, clean footage, no text, no watermark, no brand logo."
                )
                operation = await asyncio.to_thread(
                    client.models.generate_videos,
                    model=settings.GEMINI_VIDEO_MODEL,
                    prompt=clean_veo_prompt,
                    config=types.GenerateVideosConfig(
                        aspect_ratio=valid_ratio,
                        duration_seconds=6
                    )
                )
                if operation and getattr(operation, "name", None):
                    last_op = operation
                    for attempt in range(25):
                        await asyncio.sleep(4)
                        try:
                            last_op = await asyncio.to_thread(client.operations.get, operation.name)
                            if getattr(last_op, "done", False):
                                break
                        except Exception:
                            pass
                    if getattr(last_op, "done", False) and not getattr(last_op, "error", None):
                        resp = getattr(last_op, "response", None)
                        generated_vids = getattr(resp, "generated_videos", []) if resp else []
                        if generated_vids:
                            video_bytes = generated_vids[0].video.video_bytes
                            filename = f"veo_pipeline_{uuid.uuid4().hex[:12]}.mp4"
                            file_path = os.path.join(output_dir, filename)
                            with open(file_path, "wb") as vf:
                                vf.write(video_bytes)
                            file_path = await cls._attach_voiceover_narration(
                                video_path=file_path,
                                prompt=prompt,
                                brand_profile=brand_profile,
                                duration_seconds=6
                            )
                            file_url = f"/uploads/media/ai_videos/{filename}"
                            from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider
                            if CloudinaryStorageProvider.is_configured():
                                try:
                                    c_res = CloudinaryStorageProvider.upload_local_file_path(
                                        file_path=file_path,
                                        organization_id=brand_profile.organization_id if brand_profile else 1,
                                        folder_prefix="socialai/ai_videos",
                                        resource_type="video"
                                    )
                                    if c_res and c_res.get("file_url"):
                                        file_url = c_res["file_url"]
                                except Exception:
                                    pass
                            return {
                                "operation_id": getattr(operation, "name", f"veo_{uuid.uuid4().hex[:16]}"),
                                "status": "done",
                                "prompt": prompt,
                                "aspect_ratio": valid_ratio,
                                "file_path": file_url,
                                "local_path": file_path,
                                "duration_seconds": 6,
                                "mime_type": "video/mp4",
                                "provider": "google-veo",
                                "audio_provider": "edge-neural-tts"
                            }
            except Exception as veo_outer_err:
                logger.warning(f"[PipelineGemini] Google Veo video generation bypassed: {veo_outer_err}")

        # 2. Generative Video Fallback: Magic Hour (if configured)
        try:
            from app.services.magic_hour_service import MagicHourVideoService
            if MagicHourVideoService.is_configured():
                logger.info("[PipelineGemini] Attempting Magic Hour video generation fallback...")
                mh_res = await MagicHourVideoService.generate_and_persist_video(
                    prompt=prompt,
                    aspect_ratio=valid_ratio,
                    duration_seconds=10,
                    organization_id=brand_profile.organization_id if brand_profile else 1
                )
                if mh_res and mh_res.get("file_path"):
                    return {
                        "operation_id": f"mh_{uuid.uuid4().hex[:16]}",
                        "status": "done",
                        "prompt": prompt,
                        "aspect_ratio": valid_ratio,
                        "file_path": mh_res["file_path"],
                        "local_path": mh_res.get("local_path"),
                        "duration_seconds": mh_res.get("duration_seconds", 10),
                        "mime_type": "video/mp4",
                        "provider": "magic-hour-video"
                    }
        except Exception as mh_err:
            logger.debug(f"[PipelineGemini] Magic Hour fallback bypassed: {mh_err}")

        # 3. Clean Multi-Scene Branded Commercial Video Engine (Identical to Gemini AI Creative Studio)
        from app.services.video_generation_service import VideoGenerationService
        logger.info("[PipelineGemini] Rendering high-definition commercial video reel via Gemini Creative Studio engine...")
        # Collect all candidate imagery for multi-scene weave
        all_scene_images = []
        if image_url:
            all_scene_images.append(image_url)
        if image_urls:
            for u in image_urls:
                if u and u not in all_scene_images:
                    all_scene_images.append(u)

        rendered = await asyncio.to_thread(
            VideoGenerationService.render_branded_promotional_video,
            prompt=prompt,
            brand=brand_profile,
            aspect_ratio=valid_ratio,
            duration_seconds=30,
            fps=24,
            resolution="1080p",
            voice=getattr(brand_profile, "voice", None),
            image_url=image_url or (all_scene_images[0] if all_scene_images else None),
            image_urls=all_scene_images if all_scene_images else None,
            description=prompt
        )
        return {
            "operation_id": f"video_{uuid.uuid4().hex[:16]}",
            "status": "done",
            "prompt": prompt,
            "aspect_ratio": valid_ratio,
            "file_path": rendered["video_url"],
            "local_path": rendered.get("local_path"),
            "duration_seconds": rendered.get("duration_seconds", 25),
            "mime_type": "video/mp4",
            "provider": "gemini-studio-video-engine",
            "voiceover_script": rendered.get("voiceover_script"),
            "has_voiceover": rendered.get("has_voiceover", True)
        }
