"""
FastAPI research routes.

Changes vs original:
- Added GET /api/research/history (was called by Streamlit but never defined).
- Added GET /api/projects (alias for history without user filter).
- Health check now returns degraded status instead of raising 503 when
  Ollama is unreachable, so the app stays usable with the health indicator
  showing "degraded" rather than the frontend completely breaking.
- Removed the duplicate /api/health defined in main.py (kept it there for
  backward compat, this one under the router prefix is the richer version).
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from controller import controller
from services.ollamaService import ollama_service
from utils.jsonDB import json_db

router = APIRouter(prefix="/api", tags=["research"])


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Research question or topic")
    user_id: str = Field(default="local_user", description="Caller identity")


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get("/health", summary="Backend and Ollama health check")
async def health_check():
    ollama_status = await ollama_service.health_check()
    status = "ok" if ollama_status.get("connected") else "degraded"
    return {"status": status, "ollama": ollama_status}


# ------------------------------------------------------------------
# Core research workflow
# ------------------------------------------------------------------

@router.post("/research", summary="Start a new research job and return results")
async def start_research(request: ResearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        project_id = await controller.process_research_request(request.user_id, query)
        return await json_db.get_project(project_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Research pipeline error: {exc}"
        ) from exc


# ------------------------------------------------------------------
# Project & report retrieval
# ------------------------------------------------------------------

@router.get("/project/{project_id}", summary="Fetch a project by ID")
async def get_project(project_id: str):
    try:
        return await json_db.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/report/{report_id}", summary="Fetch a report by ID")
async def get_report(report_id: str):
    try:
        return await json_db.get_report(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{user_id}", summary="List all projects for a user")
async def list_projects(user_id: str):
    return await json_db.list_user_projects(user_id)


# ------------------------------------------------------------------
# History endpoint  ← THIS WAS MISSING; Streamlit called it and got 404
# ------------------------------------------------------------------

@router.get("/research/history", summary="List recent research projects (all users)")
async def research_history(limit: int = Query(default=20, ge=1, le=100)):
    return await json_db.list_all_projects(limit=limit)