from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    # 当前用户输入的消息。
    message: str = Field(..., min_length=1, max_length=2000, description="用户输入的消息")
    # 会话 id，后续接入多轮记忆时会用到。
    session_id: str = Field("demo-session", description="当前会话 id")
    # 可选知识库名称；当用户想基于资料回答时使用。
    index_name: str = Field("", max_length=100, description="可选知识库名称")
    # 是否返回调试信息，方便我们学习和查看底层调用过程。
    debug: bool = Field(False, description="是否返回调试信息")


class AgentMessageItem(BaseModel):
    # 消息角色，例如 system / user / assistant。
    role: str
    # 这一条消息的文本内容。
    content: str


class AgentDebugInfo(BaseModel):
    # 当前实际调用的大模型名称。
    used_model: str
    # 当前会话消息保存在哪里。
    session_storage: str
    # 当前会话历史本次是从哪里读取到的。
    session_history_source: str
    # 发给模型前，历史会话里已经有多少条消息。
    session_message_count_before: int
    # 当前回答保存后，会话里一共有多少条消息。
    session_message_count_after: int
    # 当前回答保存后，会话里一共有多少轮对话。
    session_turn_count_after: int
    # 这一轮真正发给模型的消息链。
    messages_sent_to_llm: list[AgentMessageItem]
    # 当前这次图执行记录下来的步骤。
    graph_steps: list[str]
    # LangGraph 路由节点判断出的意图。
    intent: str
    # LangGraph 最终所处阶段。
    final_stage: str
    # 当前使用的知识库名称。
    index_name: str
    # 当前检索到的知识片段数量。
    retrieved_chunk_count: int
    # 当前检索命中的来源名称。
    retrieved_source_names: list[str]


class AgentChatResponse(BaseModel):
    # 返回给用户的回复。
    reply: str
    # 当前返回来自哪个模块或阶段，便于学习和调试。
    stage: str
    # 当前使用的会话 id。
    session_id: str
    # 当前实际使用的大模型名称。
    model: str
    # 当前会话累计多少轮对话。
    session_turn_count: int
    # 调试信息；只有 debug=true 时才会返回。
    debug: AgentDebugInfo | None = None
