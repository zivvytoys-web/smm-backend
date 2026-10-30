import pytz
from datetime import datetime, date, time, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

from app.core.logging import logger
from app.db.dependencies import get_db
from app.core.permissions import get_current_active_user
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.social_account import SocialAccount
from app.models.autonomous_pipeline import (
    BusinessSchedule,
    PipelineRun,
    AutonomousPost,
    PostVariant,
    ContentBrief
)
from app.schemas.autonomous_pipeline import (
    BusinessScheduleBase,
    BusinessScheduleUpdate,
    BusinessScheduleResponse,
    AutomationStatusResponse,
    ScheduledJobInfo,
    AutomationLogEntry,
    ContentRescheduleRequest,
    ContentEditRequest,
    AutonomousPostResponse,
    AutonomousPostUpdate,
    PostApproveRequest,
    PostRejectRequest,
    PostRegenerateRequest,
    RunNowRequest,
    RunNowResponse
)
from app.services.pipeline_service import run_pipeline, regenerate_post_part, calculate_utc_scheduled_time
from app.scheduler.autonomous_scheduler import AutonomousScheduler
from app.services.autonomous_publisher import publish_due_posts

router = APIRouter(prefix="/automation", tags=["Autonomous Social AI Automation"])


def _get_user_brand(db: Session, current_user: User, brand_id: Optional[int] = None) -> Optional[BrandProfile]:
    """Resolves the authorized brand profile for the user safely."""
    if brand_id:
        brand = db.query(BrandProfile).filter(BrandProfile.id == brand_id).first()
        if brand:
            return brand

    if getattr(current_user, "organization_id", None):
        brand = db.query(BrandProfile).filter(
            BrandProfile.organization_id == current_user.organization_id
        ).first()
        if brand:
            return brand

    # Workspace fallback: Return any available brand profile in the database
    brand = db.query(BrandProfile).first()
    if brand:
        return brand

    # If database has zero brands, ensure a parent organization exists before inserting
    try:
        from app.models.organization import Organization
        org_id = getattr(current_user, "organization_id", None)
        org = None
        if org_id:
            org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            org = db.query(Organization).first()
        if not org:
            org = Organization(name="Default Organization", slug="default-org")
            db.add(org)
            db.commit()
            db.refresh(org)

        current_user.organization_id = org.id
        brand = BrandProfile(
            organization_id=org.id,
            name=f"{getattr(current_user, 'full_name', 'Default')}'s Brand",
            industry="Retail & Lifestyle",
            tone_of_voice="Engaging & Professional"
        )
        db.add(brand)
        db.commit()
        db.refresh(brand)
        return brand
    except Exception as create_err:
        logger.warning(f"Failed to auto-seed brand profile: {create_err}")
        return None


def _enrich_schedule_response(schedule: BusinessSchedule, brand_id: int, db: Session) -> BusinessScheduleResponse:
    resp = BusinessScheduleResponse.model_validate(schedule)
    try:
        scheduler = AutonomousScheduler.get_scheduler()
        job = scheduler.get_job(f"generate_{brand_id}")
        if job and getattr(job, "next_run_time", None):
            resp.next_generation_time = job.next_run_time
    except Exception as e:
        logger.debug(f"[Automation API] Error checking APScheduler job for brand {brand_id}: {e}")

    # Fallback calculation if job.next_run_time was not returned by the scheduler
    if resp.next_generation_time is None and schedule.is_active:
        try:
            tz = pytz.timezone(schedule.timezone or "Asia/Kolkata")
            now_tz = datetime.now(tz)
            gh, gm = map(int, (schedule.generation_time or "09:00").split(":"))
            if schedule.generation_date:
                gd = datetime.strptime(schedule.generation_date, "%Y-%m-%d").date()
                cand_dt = tz.localize(datetime(gd.year, gd.month, gd.day, gh, gm, 0))
            else:
                today_cand = tz.localize(datetime(now_tz.year, now_tz.month, now_tz.day, gh, gm, 0))
                if today_cand > now_tz:
                    cand_dt = today_cand
                else:
                    tomorrow = now_tz.date() + timedelta(days=1)
                    cand_dt = tz.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day, gh, gm, 0))
            resp.next_generation_time = cand_dt
        except Exception as e:
            logger.debug(f"[Automation API] Error calculating fallback generation time: {e}")

    try:
        tz = pytz.timezone(schedule.timezone or "Asia/Kolkata")
        today = datetime.now(tz).date()
        today_run = db.query(PipelineRun).filter(
            PipelineRun.brand_profile_id == brand_id,
            PipelineRun.run_date == today
        ).first()
        if today_run:
            resp.today_run_status = today_run.status

        latest_post = db.query(AutonomousPost).filter(
            AutonomousPost.brand_profile_id == brand_id
        ).order_by(AutonomousPost.created_at.desc()).first()

        if latest_post:
            resp.today_post_id = latest_post.id
            resp.today_post_status = latest_post.status
            for v in latest_post.variants:
                if getattr(v, "platform_url", None):
                    resp.today_post_url = v.platform_url
                    break

        if latest_post and latest_post.status == "scheduled" and latest_post.scheduled_time:
            resp.next_posting_time = latest_post.scheduled_time
        else:
            resp.next_posting_time = calculate_utc_scheduled_time(
                schedule.posting_time, schedule.timezone, getattr(schedule, "posting_date", None)
            )
    except Exception as e:
        logger.debug(f"[Automation API] Error enriching schedule metadata: {e}")

    resp.engine_status = "active" if schedule.is_active else "paused"
    return resp


# =========================================================================
# 1. AUTOMATION SETTINGS APIS
# =========================================================================

@router.get("/settings", response_model=BusinessScheduleResponse)
def get_automation_settings(
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).first()

    if not schedule:
        schedule = BusinessSchedule(
            brand_profile_id=brand.id,
            generation_time="09:00",
            posting_time="18:00",
            timezone="Asia/Kolkata",
            auto_approve=False,
            auto_schedule=True,
            is_active=True,
            target_platforms=["instagram", "facebook", "linkedin", "youtube"]
        )
        db.add(schedule)
        db.commit()
        db.refresh(schedule)
        AutonomousScheduler.reschedule_business_pipeline_job(brand.id, schedule)

    return _enrich_schedule_response(schedule, brand.id, db)


@router.put("/settings", response_model=BusinessScheduleResponse)
def update_automation_settings(
    payload: BusinessScheduleUpdate,
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).first()

    if not schedule:
        schedule = BusinessSchedule(brand_profile_id=brand.id)
        db.add(schedule)

    update_data = payload.model_dump(exclude_unset=True)

    # Normalize empty string dates to None
    if "generation_date" in update_data and (update_data["generation_date"] == "" or update_data["generation_date"] is None):
        update_data["generation_date"] = None
    if "posting_date" in update_data and (update_data["posting_date"] == "" or update_data["posting_date"] is None):
        update_data["posting_date"] = None

    # Validation: Timezone
    if "timezone" in update_data and update_data["timezone"]:
        tz_name = update_data["timezone"]
        if tz_name not in pytz.all_timezones:
            raise HTTPException(status_code=400, detail=f"Invalid timezone '{tz_name}'. Must be a valid IANA timezone.")

    tz = pytz.timezone(update_data.get("timezone") or schedule.timezone or "Asia/Kolkata")
    now_tz = datetime.now(tz)

    gen_date_str = update_data.get("generation_date", schedule.generation_date)
    gen_time_str = update_data.get("generation_time", schedule.generation_time or "09:00")
    candidate_gen = None

    if gen_date_str:
        try:
            gd = datetime.strptime(gen_date_str.strip(), "%Y-%m-%d").date()
            gh, gm = map(int, gen_time_str.split(":"))
            candidate_gen = tz.localize(datetime(gd.year, gd.month, gd.day, gh, gm, 0))
            if candidate_gen < now_tz:
                raise HTTPException(
                    status_code=400,
                    detail=f"Generation slot ({gen_date_str} at {gen_time_str}) cannot be scheduled in the past. Current time is {now_tz.strftime('%Y-%m-%d %H:%M')} ({tz.zone})."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid generation date/time format: {e}")

    post_date_str = update_data.get("posting_date", schedule.posting_date)
    post_time_str = update_data.get("posting_time", schedule.posting_time or "18:00")
    if post_date_str:
        try:
            pd = datetime.strptime(post_date_str.strip(), "%Y-%m-%d").date()
            ph, pm = map(int, post_time_str.split(":"))
            candidate_post = tz.localize(datetime(pd.year, pd.month, pd.day, ph, pm, 0))
            if candidate_gen:
                if candidate_post <= candidate_gen:
                    raise HTTPException(
                        status_code=400,
                        detail="Target posting time must be scheduled after the generation time."
                    )
            elif candidate_post < now_tz:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target posting slot ({post_date_str} at {post_time_str}) cannot be scheduled in the past."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid posting date/time format: {e}")

    for field, val in update_data.items():
        setattr(schedule, field, val)

    # If posting time/date changed, update pending/scheduled posts
    if "posting_time" in update_data or "posting_date" in update_data or "timezone" in update_data:
        new_posting_time = calculate_utc_scheduled_time(
            schedule.posting_time, schedule.timezone, target_date=schedule.posting_date
        )
        existing = db.query(AutonomousPost).filter(
            AutonomousPost.brand_profile_id == brand.id,
            AutonomousPost.status.in_(["pending_approval", "scheduled"]),
        ).all()
        for post in existing:
            post.scheduled_time = new_posting_time
            for variant in post.variants:
                if variant.status in {"pending", "pending_approval", "scheduled"}:
                    variant.scheduled_time = new_posting_time

    # If auto_approve and auto_schedule were enabled, promote pending posts
    if schedule.auto_approve and schedule.auto_schedule:
        pending_posts = db.query(AutonomousPost).filter(
            AutonomousPost.brand_profile_id == brand.id,
            AutonomousPost.status == "pending_approval",
        ).all()
        for pending in pending_posts:
            pending.status = "scheduled"
            pending.auto_approved = True
            for variant in pending.variants:
                if variant.status in {"pending", "pending_approval"}:
                    variant.status = "scheduled"

    db.commit()
    db.refresh(schedule)

    # Reschedule in APScheduler
    AutonomousScheduler.reschedule_business_pipeline_job(brand.id, schedule)
    AutonomousScheduler.reschedule_business_publications(brand.id)
    logger.info(f"[Automation API] Successfully updated settings for brand {brand.id}.")

    return _enrich_schedule_response(schedule, brand.id, db)


@router.post("/enable")
def enable_automation(
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).first()
    if not schedule:
        schedule = BusinessSchedule(brand_profile_id=brand.id)
        db.add(schedule)

    schedule.is_active = True
    db.commit()
    db.refresh(schedule)
    AutonomousScheduler.reschedule_business_pipeline_job(brand.id, schedule)
    AutonomousScheduler.reschedule_business_publications(brand.id)
    return {"success": True, "message": "Automation scheduler successfully enabled.", "is_active": True}


@router.post("/pause")
def pause_automation(
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).first()
    if not schedule:
        schedule = BusinessSchedule(brand_profile_id=brand.id)
        db.add(schedule)

    schedule.is_active = False
    db.commit()
    db.refresh(schedule)
    AutonomousScheduler.reschedule_business_pipeline_job(brand.id, schedule)
    return {"success": True, "message": "Automation scheduler paused.", "is_active": False}


@router.post("/run-now", response_model=RunNowResponse)
async def run_pipeline_now(
    payload: Optional[RunNowRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand_id = payload.brand_profile_id if payload else None
    brand = _get_user_brand(db, current_user, brand_id)
    force = payload.force if payload else True

    logger.info(f"[Automation API] Triggering immediate run for brand {brand.id} ({brand.name})...")
    result = await run_pipeline(business_id=brand.id, force=force)

    if result.get("status") == "success":
        return RunNowResponse(
            status="success",
            run_id=result.get("run_id"),
            post_id=result.get("post_id"),
            quality_score=result.get("quality_score"),
            auto_approved=result.get("auto_approved"),
            message="Autonomous content generation completed successfully."
        )
    elif result.get("status") == "skipped":
        return RunNowResponse(
            status="skipped",
            run_id=result.get("run_id"),
            message=f"Pipeline run was skipped: {result.get('reason')}"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Autonomous pipeline execution failed: {result.get('error', 'Unknown error')}"
        )


@router.get("/status", response_model=AutomationStatusResponse)
def get_automation_status(
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).first()

    if not schedule:
        schedule = BusinessSchedule(brand_profile_id=brand.id)
        db.add(schedule)
        db.commit()
        db.refresh(schedule)

    enriched = _enrich_schedule_response(schedule, brand.id, db)

    # Query latest run
    latest_run = db.query(PipelineRun).filter(
        PipelineRun.brand_profile_id == brand.id
    ).order_by(PipelineRun.created_at.desc()).first()

    current_stage = getattr(latest_run, "stage", None) if latest_run and latest_run.status == "running" else None
    current_progress = getattr(latest_run, "progress_percent", 0) if latest_run and latest_run.status == "running" else (100 if latest_run and latest_run.status == "completed" else 0)

    jobs = AutonomousScheduler.list_jobs_info(business_id=brand.id)

    return AutomationStatusResponse(
        is_active=schedule.is_active,
        engine_status="running" if (latest_run and latest_run.status == "running") else ("active" if schedule.is_active else "paused"),
        is_leader=getattr(AutonomousScheduler, '_is_leader', True),
        current_stage=current_stage,
        current_progress=current_progress,
        generation_date=schedule.generation_date or None,
        generation_time=schedule.generation_time,
        posting_date=schedule.posting_date or None,
        posting_time=schedule.posting_time,
        timezone=schedule.timezone,
        auto_approve=schedule.auto_approve,
        auto_schedule=getattr(schedule, "auto_schedule", True),
        next_generation_time=enriched.next_generation_time,
        next_posting_time=enriched.next_posting_time,
        last_run_date=latest_run.run_date if latest_run else None,
        last_run_status=latest_run.status if latest_run else None,
        last_run_error=latest_run.error if latest_run else None,
        last_post_id=enriched.today_post_id,
        last_post_status=enriched.today_post_status,
        last_post_url=enriched.today_post_url,
        active_jobs_count=len(jobs)
    )


@router.get("/jobs", response_model=List[ScheduledJobInfo])
def list_automation_jobs(
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    jobs = AutonomousScheduler.list_jobs_info(business_id=brand.id)
    return [ScheduledJobInfo(**j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=ScheduledJobInfo)
def get_automation_job(
    job_id: str,
    brand_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    jobs = AutonomousScheduler.list_jobs_info(business_id=brand.id)
    for j in jobs:
        if j["job_id"] == job_id:
            return ScheduledJobInfo(**j)
    raise HTTPException(status_code=404, detail=f"Scheduled job '{job_id}' not found.")


@router.get("/logs", response_model=List[AutomationLogEntry])
def list_automation_logs(
    brand_id: Optional[int] = None,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    brand = _get_user_brand(db, current_user, brand_id)
    runs = db.query(PipelineRun).filter(
        PipelineRun.brand_profile_id == brand.id
    ).order_by(PipelineRun.created_at.desc()).limit(limit).all()

    entries = []
    for r in runs:
        duration = None
        if r.started_at and r.finished_at:
            duration = (r.finished_at - r.started_at).total_seconds()
        topic = (r.stats or {}).get("topic") if isinstance(r.stats, dict) else None
        entries.append(AutomationLogEntry(
            id=r.id,
            brand_profile_id=r.brand_profile_id,
            run_date=r.run_date,
            status=r.status,
            stage=getattr(r, "stage", "Completed" if r.status == "completed" else "Failed" if r.status == "failed" else "Running"),
            progress_percent=getattr(r, "progress_percent", 100 if r.status == "completed" else 0),
            started_at=r.started_at,
            finished_at=r.finished_at,
            duration_seconds=duration,
            quality_score=r.quality_score,
            topic=topic,
            error=r.error,
            stats=r.stats or {}
        ))
    return entries


# =========================================================================
# 2. REVIEW QUEUE & CONTENT WORKFLOW APIS
# =========================================================================

@router.get("/review-queue", response_model=List[AutonomousPostResponse])
def get_review_queue(
    status: Optional[str] = Query(None, description="Filter by status: pending_approval, scheduled, posted, draft, all"),
    brand_id: Optional[int] = None,
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    try:
        brand = _get_user_brand(db, current_user, brand_id)
        if not brand:
            return []

        query = db.query(AutonomousPost).options(
            joinedload(AutonomousPost.brief).joinedload(ContentBrief.assets),
            joinedload(AutonomousPost.variants)
        ).filter(AutonomousPost.brand_profile_id == brand.id)

        if status and status != "all":
            if status == "pending":
                query = query.filter(AutonomousPost.status.in_(["pending_approval", "pending"]))
            else:
                query = query.filter(AutonomousPost.status == status)

        posts = query.order_by(AutonomousPost.created_at.desc()).limit(limit).all()
        return posts or []
    except Exception as queue_err:
        logger.error(f"[ReviewQueue] Error loading review queue: {queue_err}", exc_info=True)
        return []


@router.post("/content/{content_id}/approve", response_model=AutonomousPostResponse)
def approve_content(
    content_id: int,
    payload: Optional[PostApproveRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    post = db.query(AutonomousPost).filter(AutonomousPost.id == content_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Content item not found.")

    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == post.brand_profile_id).first()
    auto_sched = getattr(schedule, "auto_schedule", True) if schedule else True
    now_utc = datetime.utcnow()

    if payload and payload.custom_scheduled_time:
        post.scheduled_time = payload.custom_scheduled_time
    elif not post.scheduled_time or post.scheduled_time <= now_utc:
        # If no scheduled time or scheduled time has already elapsed, schedule for immediate dispatch
        post.scheduled_time = now_utc

    if payload and payload.notes:
        post.editor_notes = payload.notes

    if auto_sched:
        post.status = "scheduled"

        # If post variants list is empty, synthesize variants for all connected accounts
        if not post.variants:
            connected_accounts = db.query(SocialAccount).filter(
                SocialAccount.brand_profile_id == post.brand_profile_id,
                SocialAccount.is_connected == True
            ).all()
            brief = post.brief
            raw_json = brief.raw_json if brief and brief.raw_json else {}
            topic_str = brief.topic if brief else "Featured Update"
            captions_map = raw_json.get("captions") or {}
            
            # Find any generated media assets
            from app.models.autonomous_pipeline import GeneratedAsset
            assets = db.query(GeneratedAsset).filter(GeneratedAsset.brief_id == post.brief_id).all()
            first_img = next((a for a in assets if a.type == "image"), None)
            first_vid = next((a for a in assets if a.type == "video"), None)

            for sa in connected_accounts:
                m_path = first_vid.file_path if (sa.platform in {"youtube", "instagram", "tiktok"} and first_vid) else (first_img.file_path if first_img else None)
                m_type = "video" if (sa.platform in {"youtube", "instagram", "tiktok"} and first_vid) else "image"
                var = PostVariant(
                    post_id=post.id,
                    platform=sa.platform,
                    caption=captions_map.get(sa.platform, topic_str),
                    title=topic_str if sa.platform == "youtube" else None,
                    hashtags=brief.hashtags if brief else [],
                    media_path=m_path,
                    media_type=m_type,
                    aspect_ratio="9:16" if m_type == "video" else "1:1",
                    scheduled_time=post.scheduled_time,
                    status="scheduled"
                )
                db.add(var)
            db.commit()
            db.refresh(post)

        for variant in post.variants:
            variant.status = "scheduled"
            variant.scheduled_time = post.scheduled_time

        db.commit()
        db.refresh(post)
        AutonomousScheduler.schedule_post_publication(post.id, post.scheduled_time)
        logger.info(f"[Automation API] Content {content_id} approved and scheduled for {post.scheduled_time}.")
    else:
        post.status = "approved"
        for variant in post.variants:
            variant.status = "approved"
        db.commit()
        db.refresh(post)
        logger.info(f"[Automation API] Content {content_id} approved without auto-schedule.")

    return post


@router.post("/content/{content_id}/reject", response_model=AutonomousPostResponse)
def reject_content(
    content_id: int,
    payload: PostRejectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    post = db.query(AutonomousPost).filter(AutonomousPost.id == content_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Content item not found.")

    post.status = "rejected"
    post.rejection_reason = payload.reason
    for variant in post.variants:
        variant.status = "skipped"

    db.commit()
    db.refresh(post)
    AutonomousScheduler.cancel_post_publication(post.id)
    logger.info(f"[Automation API] Content {content_id} rejected. Reason: {payload.reason}")
    return post


@router.patch("/content/{content_id}", response_model=AutonomousPostResponse)
def edit_content(
    content_id: int,
    payload: ContentEditRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    post = db.query(AutonomousPost).filter(AutonomousPost.id == content_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Content item not found.")

    if payload.editor_notes is not None:
        post.editor_notes = payload.editor_notes
    if payload.scheduled_time is not None:
        post.scheduled_time = payload.scheduled_time
        for v in post.variants:
            v.scheduled_time = payload.scheduled_time

    for v in post.variants:
        if payload.caption is not None:
            v.caption = payload.caption
        if payload.title is not None:
            v.title = payload.title
        if payload.hashtags is not None:
            v.hashtags = payload.hashtags

    db.commit()
    db.refresh(post)

    if post.status == "scheduled" and payload.scheduled_time is not None:
        AutonomousScheduler.schedule_post_publication(post.id, post.scheduled_time)

    return post


@router.post("/content/{content_id}/reschedule", response_model=AutonomousPostResponse)
def reschedule_content(
    content_id: int,
    payload: ContentRescheduleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    post = db.query(AutonomousPost).filter(AutonomousPost.id == content_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Content item not found.")

    post.scheduled_time = payload.scheduled_time
    post.status = "scheduled"
    for v in post.variants:
        v.scheduled_time = payload.scheduled_time
        v.status = "scheduled"

    db.commit()
    db.refresh(post)
    AutonomousScheduler.schedule_post_publication(post.id, post.scheduled_time)
    logger.info(f"[Automation API] Content {content_id} rescheduled to {post.scheduled_time}.")
    return post


@router.post("/content/{content_id}/retry", response_model=Dict[str, Any])
async def retry_content_publication(
    content_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    post = db.query(AutonomousPost).filter(AutonomousPost.id == content_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Content item not found.")

    failed_variants = [v for v in post.variants if v.status == "failed"]
    if not failed_variants:
        # If none marked failed, re-mark variants that haven't posted yet
        failed_variants = [v for v in post.variants if v.status != "posted"]

    for v in failed_variants:
        v.status = "scheduled"
        v.scheduled_time = datetime.utcnow()
    post.status = "scheduled"
    db.commit()

    # Trigger publisher directly
    res = await publish_due_posts()
    db.refresh(post)
    return {
        "success": True,
        "message": f"Retry execution initiated for content {content_id}.",
        "status": post.status,
        "post_status": post.status,
        "publisher_result": res
    }
