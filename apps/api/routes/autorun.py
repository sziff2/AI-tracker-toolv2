"""
AutoRun API Routes — two-pipeline autonomous prompt optimisation.

POST /autorun/start   {"hours": 8, "prompt_types": [...], "pipeline": "extraction"|"output"|"both"}
POST /autorun/stop    {"pipeline": "extraction"|"output"|"both"}
GET  /autorun/status  → full state for both pipelines + combined log
"""
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["autorun"])

try:
    from scripts.autorun import (
        run_autorun_loop, get_state, request_stop,
        EXTRACTION_PROMPT_TYPES, OUTPUT_PROMPT_TYPES,
    )
    _available = True
except ImportError as e:
    logger.warning("AutoRun module not available: %s", e)
    _available = False
    EXTRACTION_PROMPT_TYPES = []
    OUTPUT_PROMPT_TYPES = []


class AutoRunRequest(BaseModel):
    hours: float = 8.0
    prompt_types: Optional[list[str]] = None
    pipeline: str = "extraction"   # "extraction" | "output" | "both"
    dry_run: bool = False


class StopRequest(BaseModel):
    pipeline: str = "both"


@router.post("/autorun/start")
async def start_autorun(body: AutoRunRequest, background_tasks: BackgroundTasks):
    if not _available:
        raise HTTPException(503, "AutoRun module not loaded. Check scripts/autorun.py exists.")
    if body.hours <= 0 or body.hours > 24:
        raise HTTPException(400, "hours must be between 0 and 24")

    state = get_state()
    pipelines = ["extraction", "output"] if body.pipeline == "both" else [body.pipeline]
    for p in pipelines:
        if state.get(p, {}).get("running"):
            raise HTTPException(409, f"{p} pipeline already running (job {state[p].get('job_id')}). Stop it first.")

    background_tasks.add_task(
        run_autorun_loop,
        hours=body.hours,
        prompt_types=body.prompt_types,
        pipeline=body.pipeline,
        dry_run=body.dry_run,
    )
    return {
        "status": "started",
        "pipeline": body.pipeline,
        "hours": body.hours,
        "prompt_types": body.prompt_types,
    }


@router.post("/autorun/stop")
async def stop_autorun(body: StopRequest):
    if not _available:
        raise HTTPException(503, "AutoRun module not loaded.")
    request_stop(body.pipeline)
    return {"status": "stop_requested", "pipeline": body.pipeline}


@router.get("/autorun/status")
async def autorun_status():
    if not _available:
        return {"available": False, "extraction": {}, "output": {}, "log": []}
    state = get_state()
    state["available"] = True
    state["extraction_prompt_types"] = EXTRACTION_PROMPT_TYPES
    state["output_prompt_types"] = OUTPUT_PROMPT_TYPES
    return state
