from fastapi import FastAPI

from app.config import settings
from app.exceptions import register_exception_handlers
from app.routers.chat import router as chat_router
from app.routers.health import router as health_router


def create_app() -> FastAPI:
    # Create the FastAPI application object.
    app = FastAPI(
        title=settings.app_name,
        description="Day 1 and Day 2 learning project for AI application development.",
        version=settings.app_version,
    )

    # Register routes from the router module.
    app.include_router(health_router)
    app.include_router(chat_router)
    register_exception_handlers(app)
    return app


# Uvicorn starts the project from this app instance.
app = create_app()
