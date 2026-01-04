from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.feedback import router as feedback_router
from .api.datasets import router as datasets_router
from .api.jobs import router as jobs_router
from .api.results import router as results_router
from .api.reports import router as reports_router
from .api.runs import router as runs_router
from .api.comparisons import router as comparisons_router
from .api.progress import router as progress_router
import uvicorn
from .utils.errors import add_exception_handlers
from .settings import load_settings


def create_app() -> FastAPI:
    tags_metadata = [
        {"name": "datasets", "description": "Browse datasets and conversations."},
        {"name": "runs", "description": "Start runs, retrieve metrics, artifacts, and comparisons."},
        {"name": "progress", "description": "Run progress (HTTP + WebSocket)."},
        {"name": "jobs", "description": "Queue jobs and manage cancellation."},
        {"name": "results", "description": "Fetch summary and consolidated results."},
        {"name": "reports", "description": "Generate HTML/Markdown/comparison reports."},
        {"name": "feedback", "description": "Submit and retrieve evaluator feedback."},
    ]
    app = FastAPI(
        title="Eval Server",
        version="0.1.0",
        description="API for running, tracking, and reporting LLM evaluation runs. Provides dataset browsing, run management, progress, results, reports, feedback, comparisons, and artifacts.",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=tags_metadata,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load settings early to fail fast if secrets are missing
    try:
        load_settings()
    except Exception:
        # Defer raising to endpoints; still construct app for OpenAPI and tests.
        pass

    @app.get("/health", summary="Health check", description="Simple health probe endpoint.")
    def health():
        return {"status": "ok"}

    # API routes under /api/v1
    add_exception_handlers(app)
    app.include_router(datasets_router, prefix="/api/v1")
    app.include_router(runs_router, prefix="/api/v1")
    app.include_router(progress_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(results_router, prefix="/api/v1")
    app.include_router(reports_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(comparisons_router, prefix="/api/v1")

    return app


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
