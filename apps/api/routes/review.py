"""
Review Queue endpoints (§8).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import ReviewQueueItem
from schemas import ReviewAction, ReviewQueueOut

router = APIRouter(prefix="/review-queue", tags=["review"])


@router.get("", response_model=list[ReviewQueueOut])
async def list_queue(
    status: str = "open",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ReviewQueueItem)
        .where(ReviewQueueItem.status == status)
        .order_by(ReviewQueueItem.created_at.desc())
    )
    return result.scalars().all()


@router.post("/{item_id}/approve", response_model=ReviewQueueOut)
async def approve_item(item_id: uuid.UUID, body: ReviewAction = ReviewAction(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ReviewQueueItem).where(ReviewQueueItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(404, "Queue item not found")
    item.status = "approved"
    await db.commit()
    await db.refresh(item)
    return item


@router.post("/{item_id}/reject", response_model=ReviewQueueOut)
async def reject_item(item_id: uuid.UUID, body: ReviewAction = ReviewAction(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ReviewQueueItem).where(ReviewQueueItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(404, "Queue item not found")
    item.status = "rejected"
    await db.commit()
    await db.refresh(item)
    return item


@router.post("/{item_id}/edit", response_model=ReviewQueueOut)
async def edit_item(item_id: uuid.UUID, body: ReviewAction = ReviewAction(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ReviewQueueItem).where(ReviewQueueItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(404, "Queue item not found")
    item.status = "edited"
    await db.commit()
    await db.refresh(item)
    return item
