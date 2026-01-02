from fastapi import FastAPI
import uvicorn


def create_app() -> FastAPI:
    app = FastAPI(title="Eval Server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
