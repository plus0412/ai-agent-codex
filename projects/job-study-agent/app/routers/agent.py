from fastapi import APIRouter

from app.schemas.agent import AgentChatRequest, AgentChatResponse
from app.services.agent_service import build_agent_chat_reply

# 这一组路由专门负责综合项目里的 Agent 相关接口。
router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=AgentChatResponse, response_model_exclude_none=True)
def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    # 先调用 service 层，后续真正的 Agent 主流程会逐步写在这里面。
    return build_agent_chat_reply(request)
