"""Application entrypoint for the AIgentEgo runtime API."""

import uvicorn
from fastapi import FastAPI

from aigentego.api.routes import router
from aigentego.settings import Settings


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="AIgentEgo")
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Run the API server with Uvicorn."""
    settings = Settings()
    uvicorn.run(
        "aigentego.main:app",
        host=settings.agent_host,
        port=settings.agent_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
