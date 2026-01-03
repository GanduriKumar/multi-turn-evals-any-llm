from fastapi import FastAPI
from .api.feedback import router as feedback_router
import uvicorn


def create_app() -> FastAPI:
    app = FastAPI(title="Eval Server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # API routes
    app.include_router(feedback_router, prefix="/api")

    return app


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
