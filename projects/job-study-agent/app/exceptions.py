from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # 业务层抛出的 ValueError 统一转成结构化错误响应。
        return JSONResponse(
            status_code=400,
            content={
                "error_type": "ValueError",
                "message": str(exc),
                "path": str(request.url.path),
            },
        )
