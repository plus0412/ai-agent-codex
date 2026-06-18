from fastapi import APIRouter

from app.schemas.health import HealthResponse

# health 路由一般用来做服务存活检查。
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    # 返回一个固定结果，表示服务已经正常启动。
    return HealthResponse(status="ok", message="job-study-agent is running")

