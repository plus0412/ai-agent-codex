from pydantic import BaseModel


class HealthResponse(BaseModel):
    # 服务状态，例如 ok。
    status: str
    # 对状态的说明文字。
    message: str

