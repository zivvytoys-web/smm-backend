from app.core.security import decrypt_token
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.db.session import SessionLocal
from app.models.autonomous_pipeline import AutonomousPost, PostVariant, BusinessSchedule
from app.models.social_account import SocialAccount
from app.models.brand_profile import BrandProfile
from app.integrations.social import (
    FacebookClient,
    InstagramClient,
    LinkedInClient,
    YouTubeClient,
    TwitterClient,
    PinterestClient,
    SnapchatClient,
    TikTokClient
)


PLATFORM_PUBLISHERS = {
    "facebook": FacebookClient,
    "instagram": InstagramClient,
    "linkedin": LinkedInClient,
    "youtube": YouTubeClient,
    "twitter": TwitterClient,
    "pinterest": PinterestClient,
    "snapchat": SnapchatClient,
    "tiktok": TikTokClient,
}


async def publish_due_posts() -> Dict[str, Any]:
    """
    Global Publishing Worker Job (executed every minute by APScheduler).
    1. Expires any pending_approval posts whose scheduled_time has elapsed.
    2. Finds all PostVariants with status='scheduled' and scheduled_time <= now_utc,
       and automatically dispatches them to their respective social networks.
    """
    db: Session = SessionLocal()
    now_utc = datetime.utcnow()
    processed_count = 0
    success_count = 0
    failed_count = 0

    try:
        # 0. Reset rate_limited variants whose cooldown window has passed → back to 'scheduled'
        cooled_variants = db.query(PostVariant).filter(
            PostVariant.status == "rate_limited",
            PostVariant.retry_after <= now_utc
        ).all()
        for cv in cooled_variants:
            cv.status = "scheduled"
            cv.retry_after = None
            cv.error = "[Rate-limit cooldown expired — retrying publish]"
            logger.info(f"[Publisher] Rate-limit cooldown expired for variant {cv.id} on {cv.platform}. Rescheduling.")
        if cooled_variants:
            db.commit()

        # 1. Expire unapproved posts only if older than 48 hours past scheduled deadline
        # Prevents premature expiration of current posts so users can review and approve seamlessly
        grace_expiry_cutoff = now_utc - timedelta(days=2)
        expired_posts = db.query(AutonomousPost).filter(
            AutonomousPost.status == "pending_approval",
            AutonomousPost.scheduled_time <= grace_expiry_cutoff
        ).all()

        for exp_post in expired_posts:
            exp_post.status = "expired"
            exp_post.rejection_reason = "Expired: Not approved within 48 hours of scheduled posting deadline."
            for v in exp_post.variants:
                if v.status == "scheduled" or v.status == "pending":
                    v.status = "expired"
                    v.error = "Expired before approval"
            logger.info(f"[Publisher] Expired unapproved post {exp_post.id} (Scheduled for {exp_post.scheduled_time})")
        
        if expired_posts:
            db.commit()

        # 1.5. Synthesize variants for scheduled posts that currently have 0 variants
        empty_sched_posts = db.query(AutonomousPost).filter(
            AutonomousPost.status == "scheduled",
            AutonomousPost.scheduled_time <= now_utc
        ).all()
        for empty_p in empty_sched_posts:
            if not empty_p.variants:
                connected = db.query(SocialAccount).filter(
                    SocialAccount.brand_profile_id == empty_p.brand_profile_id,
                    SocialAccount.is_connected == True
                ).all()
                if connected:
                    brief = empty_p.brief
                    raw_json = brief.raw_json if brief and brief.raw_json else {}
                    topic_str = brief.topic if brief else "Featured Post"
                    captions_map = raw_json.get("captions") or {}
                    from app.models.autonomous_pipeline import GeneratedAsset
                    assets = db.query(GeneratedAsset).filter(GeneratedAsset.brief_id == empty_p.brief_id).all()
                    first_img = next((a for a in assets if a.type == "image"), None)
                    first_vid = next((a for a in assets if a.type == "video"), None)
                    for sa in connected:
                        m_path = first_vid.file_path if (sa.platform in {"youtube", "instagram", "tiktok"} and first_vid) else (first_img.file_path if first_img else None)
                        m_type = "video" if (sa.platform in {"youtube", "instagram", "tiktok"} and first_vid) else "image"
                        db.add(PostVariant(
                            post_id=empty_p.id,
                            platform=sa.platform,
                            caption=captions_map.get(sa.platform, topic_str),
                            title=topic_str if sa.platform == "youtube" else None,
                            hashtags=brief.hashtags if brief else [],
                            media_path=m_path,
                            media_type=m_type,
                            aspect_ratio="9:16" if m_type == "video" else "1:1",
                            scheduled_time=empty_p.scheduled_time,
                            status="scheduled"
                        ))
                    db.commit()

        # 2. Query due scheduled variants (excluding those still in rate-limit cooldown)
        from sqlalchemy import or_
        due_variants = db.query(PostVariant).filter(
            PostVariant.status == "scheduled",
            or_(
                PostVariant.scheduled_time <= now_utc,
                PostVariant.scheduled_time.is_(None)
            ),
            # Skip variants still inside a rate-limit cooldown window
            or_(
                PostVariant.retry_after.is_(None),
                PostVariant.retry_after <= now_utc
            )
        ).limit(20).all()

        if not due_variants:
            return {"processed": 0, "success": 0, "failed": 0}

        logger.info(f"[Publisher] Found {len(due_variants)} scheduled post variants ready for publishing.")

        for variant in due_variants:
            processed_count += 1
            post = variant.post
            if not post:
                continue

            # Mark variant as posting
            variant.status = "posting"
            post.status = "posting"
            db.commit()

            # Find matching connected social account
            social_account = db.query(SocialAccount).filter(
                SocialAccount.brand_profile_id == post.brand_profile_id,
                SocialAccount.platform == variant.platform,
                SocialAccount.is_connected == True
            ).first()

            publisher_cls = PLATFORM_PUBLISHERS.get(variant.platform.lower())
            
            try:
                if not social_account:
                    raise ValueError(f"No connected {variant.platform} account is assigned to this brand.")
                if not publisher_cls:
                    raise ValueError(f"Publishing is not supported for platform '{variant.platform}'.")

                # Initialize client with decrypted credentials
                raw_token = decrypt_token(social_account.access_token) if social_account.access_token else None
                raw_refresh = decrypt_token(social_account.refresh_token) if getattr(social_account, "refresh_token", None) else None

                try:
                    client = publisher_cls(access_token=raw_token, refresh_token=raw_refresh)
                except TypeError:
                    client = publisher_cls(access_token=raw_token)

                publish_kwargs = {
                    "media_type": variant.media_type,
                    "aspect_ratio": variant.aspect_ratio,
                    "title": variant.title,
                    "hashtags": variant.hashtags,
                }
                plat_lower = variant.platform.lower()
                if plat_lower == "facebook":
                    publish_kwargs["page_id"] = social_account.account_id or "me"
                elif plat_lower == "instagram":
                    publish_kwargs["account_id"] = social_account.account_id
                    publish_kwargs["post_format"] = "reel" if variant.media_type == "video" else "post"
                elif plat_lower == "youtube":
                    publish_kwargs["title"] = variant.title or (variant.caption[:50] if variant.caption else "Brand Video")
                elif plat_lower == "snapchat":
                    publish_kwargs["ad_account_id"] = social_account.account_id

                media_urls = [variant.media_path] if variant.media_path else []

                # Dispatch directly to client's publish_post method
                publish_result = await client.publish_post(
                    content_text=variant.caption or "",
                    media_urls=media_urls,
                    **publish_kwargs
                )

                is_ok = bool(
                    publish_result and (
                        publish_result.get("success")
                        or publish_result.get("status") in {"success", "published"}
                        or publish_result.get("id")
                        or publish_result.get("platform_post_id")
                    )
                )

                if is_ok:
                    variant.status = "posted"
                    variant.platform_post_id = str(publish_result.get("platform_post_id") or publish_result.get("id") or publish_result.get("post_id") or "")
                    variant.platform_url = publish_result.get("platform_url") or publish_result.get("permalink") or publish_result.get("url")
                    variant.publish_metadata = publish_result
                    variant.error = None
                    success_count += 1
                    logger.info(f"[Publisher] Successfully published post {post.id} variant {variant.id} to {variant.platform}")
                else:
                    error = (publish_result or {}).get("error") or "Provider did not confirm publication."
                    err_str = str(error)
                    is_rate_limited = (
                        "2207051" in err_str
                        or "action blocked" in err_str.lower()
                        or "security hold" in err_str.lower()
                        or "rate limit" in err_str.lower()
                        or "application request limit" in err_str.lower()
                    )
                    if is_rate_limited:
                        cooldown_hours = 2
                        variant.status = "rate_limited"
                        variant.retry_after = now_utc + timedelta(hours=cooldown_hours)
                        variant.error = err_str
                        variant.publish_metadata = publish_result or {"error": error}
                        logger.warning(
                            f"[Publisher] Instagram rate-limit soft-block on variant {variant.id} "
                            f"(post {post.id}). Cooling down for {cooldown_hours}h."
                        )
                    else:
                        variant.status = "failed"
                        variant.error = err_str
                        variant.publish_metadata = publish_result or {"error": error}
                        failed_count += 1
                        logger.warning(f"[Publisher] Provider did not confirm publication for post {post.id} on {variant.platform}: {err_str}")

            except Exception as e:
                err_str = str(e)
                is_rate_limited = (
                    "2207051" in err_str
                    or "action blocked" in err_str.lower()
                    or "security hold" in err_str.lower()
                    or "rate limit" in err_str.lower()
                    or "application request limit" in err_str.lower()
                    or "user request limit" in err_str.lower()
                )
                if is_rate_limited:
                    # Temporarily back off: set 2-hour cooldown, keep status as rate_limited
                    cooldown_hours = 2
                    variant.status = "rate_limited"
                    variant.retry_after = now_utc + timedelta(hours=cooldown_hours)
                    variant.error = err_str
                    logger.warning(
                        f"[Publisher] Instagram rate-limit / action-block on variant {variant.id} "
                        f"(post {post.id}). Cooling down for {cooldown_hours}h — will auto-retry at "
                        f"{variant.retry_after.strftime('%H:%M UTC')}."
                    )
                else:
                    failed_count += 1
                    variant.status = "failed"
                    variant.error = err_str
                    logger.error(f"[Publisher] Error publishing to {variant.platform} for post {post.id}: {err_str}")

            db.commit()

            # Check if all variants for the post are finished
            all_variants = db.query(PostVariant).filter(PostVariant.post_id == post.id).all()
            if all(v.status in ["posted", "failed", "skipped", "expired"] for v in all_variants):
                if any(v.status == "posted" for v in all_variants):
                    post.status = "posted"
                    post.published_at = datetime.utcnow()
                    try:
                        from app.services.analytics_service import AnalyticsService
                        async def _delayed_pipeline_sync(b_id: int):
                            await asyncio.sleep(5)
                            from app.db.session import SessionLocal
                            s_db = SessionLocal()
                            try:
                                brand = s_db.query(BrandProfile).filter(BrandProfile.id == b_id).first()
                                if brand:
                                    await AnalyticsService.sync_analytics(db=s_db, organization_id=brand.organization_id, brand_id=b_id)
                            except Exception as e:
                                logger.debug(f"Delayed pipeline sync after posting: {e}")
                            finally:
                                s_db.close()
                        asyncio.create_task(_delayed_pipeline_sync(post.brand_profile_id))
                    except Exception as sync_dispatch_err:
                        logger.debug(f"Could not dispatch pipeline analytics sync: {sync_dispatch_err}")
                else:
                    post.status = "failed"
            db.commit()

        return {"processed": processed_count, "success": success_count, "failed": failed_count}

    except Exception as e:
        logger.exception(f"[Publisher] Exception in publish_due_posts loop: {e}")
        return {"error": str(e)}
    finally:
        db.close()
