from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage


class AgentGraphState(TypedDict):
    # 当前这一轮用户输入。
    message: str
    # 当前会话 id。
    session_id: str
    # 发给 LangGraph 和模型使用的完整消息链。
    messages: list[BaseMessage]
    # 当前使用的知识库名称。
    index_name: str
    # 当前任务意图，会在路由节点中写入。
    intent: Literal["chat", "knowledge_qa", "job_analysis", "study_plan", "fallback"]
    # 最终返回给用户的回复。
    reply: str
    # 当前图执行到了哪个阶段。
    stage: str
    # 是否需要追问用户补充信息。
    need_followup: bool
    # 追问内容。
    followup_question: str
    # 当前错误信息。
    error_message: str
    # 当前检索到的知识片段。
    retrieved_chunks: list[str]
    # 当前检索命中的来源名称。
    retrieved_source_names: list[str]
    # 当前图的调试步骤。
    debug_steps: list[str]
