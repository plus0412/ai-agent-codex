from fastapi import FastAPI

from app.config import settings
from app.database import init_db
from app.exceptions import register_exception_handlers
from app.routers.agent import router as agent_router
from app.routers.health import router as health_router
from app.routers.knowledge import router as knowledge_router


def create_app() -> FastAPI:
    # 创建 FastAPI 应用对象。
    app = FastAPI(
        title=settings.app_name,
        description="面向求职与学习场景的 Agent 项目骨架。",
        version=settings.app_version,
    )

    # 应用启动时先确保数据库表存在。
    init_db()

    # 注册路由。
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(knowledge_router)
    register_exception_handlers(app)
    return app


# uvicorn 启动时会读取这个 app 对象。
app = create_app()
