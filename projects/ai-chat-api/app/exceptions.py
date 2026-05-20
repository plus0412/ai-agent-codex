from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # 当业务层抛出 ValueError 时，统一返回结构化错误信息。
        return JSONResponse(
            status_code=400,
            content={
                "error_type": "ValueError",
                "message": str(exc),
                "path": str(request.url.path),
            },
        )
