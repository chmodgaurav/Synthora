"""
Synthora FastAPI application entry point.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.researchRoutes import router

app = FastAPI(
    title="Synthora AI Research Agent",
    version="2.1.0",
    description="Multi-agent research pipeline: search → summarize → fact-check → report",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your Streamlit domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "Synthora AI Research Agent",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)