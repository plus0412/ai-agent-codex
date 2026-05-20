from fastapi import APIRouter

from app.schemas.health import HealthResponse, HelloRequest, HelloResponse

# A router groups related endpoints together.
router = APIRouter(tags=["demo"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    # Return a fixed response to confirm the service is running.
    return HealthResponse(status="ok", message="service is running")


@router.post("/hello", response_model=HelloResponse)
def say_hello(request: HelloRequest) -> HelloResponse:
    # Read the input name and return a greeting message.
    return HelloResponse(message=f"Hello, {request.name}. Welcome to FastAPI.")
