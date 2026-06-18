from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db_session
from app.graph.state import AgentGraphState
from app.models import ChatMessage
from app.redis_client import (
    load_session_messages_from_cache,
    save_session_messages_to_cache,
)
from app.schemas.agent import AgentChatRequest, AgentChatResponse, AgentDebugInfo, AgentMessageItem
from app.schemas.knowledge import KnowledgeSearchRequest
from app.services.knowledge_service import build_knowledge_search_reply

# 当前项目的统一系统提示词。
SYSTEM_PROMPT = (
    "你是一名面向求职与学习场景的 AI 助手。"
    "请用清晰、自然、通俗的中文回答。"
    "如果用户的问题和学习规划、岗位理解、技术概念有关，请尽量解释得具体一些。"
)


def ensure_llm_config() -> None:
    # 在真正调用模型前，先检查关键配置是否已经准备好。
    if not settings.llm_api_key:
        raise ValueError("缺少 LLM_API_KEY，请先在 .env 中配置大模型 API Key")

    if not settings.llm_base_url:
        raise ValueError("缺少 LLM_BASE_URL，请先在 .env 中配置大模型接口地址")


def get_llm_client() -> OpenAI:
    # 创建一个 OpenAI 兼容客户端。
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def get_langchain_llm() -> ChatOpenAI:
    # 创建 LangChain 的聊天模型对象，后续 LangGraph 节点会复用它。
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def load_history_messages_from_mysql(session_id: str) -> list[AgentMessageItem]:
    # 从 MySQL 中读取当前会话的历史消息。
    with get_db_session() as db:
        rows = db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
        ).scalars()

        return [
            AgentMessageItem(role=row.role, content=row.content)
            for row in rows
        ]


def build_history_messages(session_id: str) -> tuple[list[AgentMessageItem], str]:
    # 优先从 Redis 读取会话历史；如果没命中，再回源 MySQL。
    cached_history = load_session_messages_from_cache(session_id)
    if cached_history is not None:
        return cached_history, "redis"

    mysql_history = load_history_messages_from_mysql(session_id)
    if mysql_history:
        save_session_messages_to_cache(session_id, mysql_history)
        return mysql_history, "mysql"

    return [], "mysql"


def build_langchain_history_messages(history: list[AgentMessageItem]) -> list[BaseMessage]:
    # 把历史消息转成 LangChain / LangGraph 能识别的消息对象。
    history_messages: list[BaseMessage] = []
    for item in history:
        if item.role == "user":
            history_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            history_messages.append(AIMessage(content=item.content))
    return history_messages


def trim_session_history(db: Session, session_id: str) -> None:
    # 只保留最近 N 条消息，避免数据库里的单个会话无限增长。
    stale_ids = list(
        db.execute(
            select(ChatMessage.id)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .offset(settings.session_max_messages)
        ).scalars()
    )

    if stale_ids:
        db.execute(delete(ChatMessage).where(ChatMessage.id.in_(stale_ids)))


def save_session_turn(session_id: str, user_message: str, assistant_reply: str) -> None:
    # 把这一轮的用户消息和助手回复写入 MySQL。
    with get_db_session() as db:
        db.add(ChatMessage(session_id=session_id, role="user", content=user_message))
        db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_reply))
        db.flush()
        trim_session_history(db, session_id)

    latest_history = load_history_messages_from_mysql(session_id)
    save_session_messages_to_cache(session_id, latest_history)


def get_session_message_count(session_id: str) -> int:
    # 统计当前会话里一共有多少条消息。
    history, _ = build_history_messages(session_id)
    return len(history)


def build_debug_messages(messages: list[BaseMessage]) -> list[AgentMessageItem]:
    # 把 LangChain 消息对象整理成更适合前端查看的调试结构。
    debug_messages: list[AgentMessageItem] = []
    for message in messages:
        role = "unknown"
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"

        content = message.content if isinstance(message.content, str) else str(message.content)
        debug_messages.append(AgentMessageItem(role=role, content=content))
    return debug_messages


def detect_intent_by_rules(message: str) -> str:
    # 先用最小规则路由，后面再升级成更完整的分类方式。
    lowered_message = message.lower()

    if any(keyword in lowered_message for keyword in ["计划", "学习路线", "学习建议", "7天", "七天"]):
        return "study_plan"

    if any(keyword in lowered_message for keyword in ["岗位", "jd", "招聘要求", "岗位要求", "分析一下岗位"]):
        return "job_analysis"

    if any(keyword in lowered_message for keyword in ["资料", "文档", "知识库", "笔记", "上传的内容"]):
        return "knowledge_qa"

    return "chat"


def route_intent_node(state: AgentGraphState) -> AgentGraphState:
    # 第一个节点：先判断当前问题属于哪一类。
    intent = detect_intent_by_rules(state["message"])
    debug_steps = [*state["debug_steps"], f"route_intent: intent={intent}"]
    return {
        **state,
        "intent": intent,
        "stage": "intent_routed",
        "debug_steps": debug_steps,
    }


def check_context_node(state: AgentGraphState) -> AgentGraphState:
    # 第二个节点：检查当前问题是否具备足够上下文。
    intent = state["intent"]
    debug_steps = [*state["debug_steps"], f"check_context: intent={intent}"]

    if intent in {"knowledge_qa", "job_analysis"} and not state["index_name"]:
        return {
            **state,
            "need_followup": True,
            "followup_question": "这类问题需要指定知识库。请先上传资料建库，并在提问时传入 index_name。",
            "stage": "context_checked_need_followup",
            "debug_steps": debug_steps,
        }

    cleaned_message = state["message"].strip()
    if intent == "study_plan" and len(cleaned_message) < 6:
        return {
            **state,
            "need_followup": True,
            "followup_question": "你可以再说具体一点，例如你的目标岗位、当前基础，或者希望我给几天的学习计划。",
            "stage": "context_checked_need_followup",
            "debug_steps": debug_steps,
        }

    return {
        **state,
        "need_followup": False,
        "followup_question": "",
        "stage": "context_checked_ok",
        "debug_steps": debug_steps,
    }


def retrieve_knowledge_node(state: AgentGraphState) -> AgentGraphState:
    # 检索节点：从指定知识库中找相关片段。
    debug_steps = [*state["debug_steps"], f"retrieve_knowledge: index_name={state['index_name']}"]
    search_result = build_knowledge_search_reply(
        KnowledgeSearchRequest(
            index_name=state["index_name"],
            question=state["message"],
            top_k=3,
            score_threshold=0.0,
            source_name_filter="",
        )
    )

    return {
        **state,
        "retrieved_chunks": search_result.selected_chunks,
        "retrieved_source_names": search_result.source_names,
        "stage": "knowledge_retrieved",
        "debug_steps": debug_steps,
    }


def job_analysis_node(state: AgentGraphState) -> AgentGraphState:
    # 岗位分析节点：基于检索到的资料提炼岗位重点。
    llm = get_langchain_llm()
    debug_steps = [*state["debug_steps"], f"job_analysis: index_name={state['index_name']}"]
    joined_chunks = "\n\n".join(state["retrieved_chunks"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "你现在的任务是做岗位分析。"
                + "请优先提炼岗位要求中的关键技能、优先级最高的能力、以及当前求职者最该补的方向。"
                + "回答时请尽量按“岗位重点 / 优先补强 / 学习建议 / 总结”四部分输出。",
            ),
            MessagesPlaceholder("messages"),
            (
                "human",
                "岗位资料片段如下：\n{knowledge_chunks}\n\n请结合这些片段做岗位分析，并给出适合学习者的行动建议。",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    reply = chain.invoke(
        {
            "messages": state["messages"],
            "knowledge_chunks": joined_chunks or "当前没有检索到岗位资料片段。",
        }
    )

    return {
        **state,
        "reply": reply,
        "stage": "job_analysis_completed",
        "debug_steps": debug_steps,
    }


def study_plan_node(state: AgentGraphState) -> AgentGraphState:
    # 学习计划节点：把问题变成可执行的学习安排。
    llm = get_langchain_llm()
    debug_steps = [*state["debug_steps"], f"study_plan: message={state['message']}"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "你现在的任务是生成一个可执行的学习计划。"
                + "请结合用户当前基础、目标方向和时间限制，输出简洁清晰的计划。"
                + "回答时请尽量按“目标 / 分阶段安排 / 每天要做什么 / 注意事项”四部分输出。",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    reply = chain.invoke({"messages": state["messages"]})

    return {
        **state,
        "reply": reply,
        "stage": "study_plan_completed",
        "debug_steps": debug_steps,
    }


def answer_with_knowledge_node(state: AgentGraphState) -> AgentGraphState:
    # 基于知识片段生成回答。
    llm = get_langchain_llm()
    debug_steps = [*state["debug_steps"], f"answer_with_knowledge: intent={state['intent']}"]
    joined_chunks = "\n\n".join(state["retrieved_chunks"])

    system_prompt = (
        SYSTEM_PROMPT
        + "请优先依据我提供的知识片段回答。"
        + "如果知识片段里没有足够信息，就明确说明“知识库片段中没有直接提到”。"
    )
    if state["intent"] == "job_analysis":
        system_prompt += "如果这是岗位分析问题，请总结岗位重点能力、技能要求和建议优先补的方向。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
            (
                "human",
                "补充知识片段如下：\n{knowledge_chunks}\n\n请结合上面的知识片段回答我这次的问题。",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    reply = chain.invoke(
        {
            "messages": state["messages"],
            "knowledge_chunks": joined_chunks or "当前没有检索到知识片段。",
        }
    )

    return {
        **state,
        "reply": reply,
        "stage": "knowledge_answer_completed",
        "debug_steps": debug_steps,
    }


def direct_answer_node(state: AgentGraphState) -> AgentGraphState:
    # 直接回答节点：对当前已经可以回答的问题直接生成结果。
    llm = get_langchain_llm()
    debug_steps = [*state["debug_steps"], f"direct_answer: intent={state['intent']}"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "当前你处于综合项目的主干流程中。"
                + "如果问题属于学习计划类，可以给出简洁可执行建议；如果是普通聊天类，就直接自然回答。",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    reply = chain.invoke({"messages": state["messages"]})

    return {
        **state,
        "reply": reply,
        "stage": "direct_answer_completed",
        "debug_steps": debug_steps,
    }


def fallback_node(state: AgentGraphState) -> AgentGraphState:
    # 兜底节点：当追问或异常发生时，统一从这里返回可读结果。
    debug_steps = [*state["debug_steps"], "fallback: building fallback reply"]

    if state["need_followup"]:
        reply = state["followup_question"]
        stage = "fallback_followup_returned"
    elif state["error_message"]:
        reply = f"这次处理没有成功，原因是：{state['error_message']}"
        stage = "fallback_error_returned"
    else:
        reply = "当前这次请求没有命中可执行路径，我们后续会继续补全这个分支。"
        stage = "fallback_default_returned"

    return {
        **state,
        "reply": reply,
        "stage": stage,
        "debug_steps": debug_steps,
    }


def save_session_node(state: AgentGraphState) -> AgentGraphState:
    # 最后一个节点：统一把当前轮次结果写入会话历史。
    debug_steps = [*state["debug_steps"], "save_session: session saved to mysql"]
    save_session_turn(
        state["session_id"],
        state["message"],
        state["reply"],
    )
    return {
        **state,
        "stage": "session_saved",
        "debug_steps": debug_steps,
    }


def choose_after_check_context(state: AgentGraphState) -> str:
    # 根据上下文检查结果，决定去直接回答还是去 fallback。
    if state["need_followup"]:
        return "fallback"
    if state["intent"] in {"knowledge_qa", "job_analysis"}:
        return "retrieve_knowledge"
    if state["intent"] == "study_plan":
        return "study_plan"
    return "direct_answer"


def choose_after_retrieve_knowledge(state: AgentGraphState) -> str:
    # 检索到知识后，根据任务类型决定走哪个任务节点。
    if state["intent"] == "job_analysis":
        return "job_analysis"
    return "answer_with_knowledge"


def build_agent_graph() -> StateGraph:
    # 组装当前综合项目的 LangGraph 主图。
    graph_builder = StateGraph(AgentGraphState)
    graph_builder.add_node("route_intent", route_intent_node)
    graph_builder.add_node("check_context", check_context_node)
    graph_builder.add_node("retrieve_knowledge", retrieve_knowledge_node)
    graph_builder.add_node("answer_with_knowledge", answer_with_knowledge_node)
    graph_builder.add_node("job_analysis", job_analysis_node)
    graph_builder.add_node("study_plan", study_plan_node)
    graph_builder.add_node("direct_answer", direct_answer_node)
    graph_builder.add_node("fallback", fallback_node)
    graph_builder.add_node("save_session", save_session_node)

    graph_builder.add_edge(START, "route_intent")
    graph_builder.add_edge("route_intent", "check_context")
    graph_builder.add_conditional_edges(
        "check_context",
        choose_after_check_context,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "study_plan": "study_plan",
            "direct_answer": "direct_answer",
            "fallback": "fallback",
        },
    )
    graph_builder.add_conditional_edges(
        "retrieve_knowledge",
        choose_after_retrieve_knowledge,
        {
            "answer_with_knowledge": "answer_with_knowledge",
            "job_analysis": "job_analysis",
        },
    )
    graph_builder.add_edge("answer_with_knowledge", "save_session")
    graph_builder.add_edge("job_analysis", "save_session")
    graph_builder.add_edge("study_plan", "save_session")
    graph_builder.add_edge("direct_answer", "save_session")
    graph_builder.add_edge("fallback", "save_session")
    graph_builder.add_edge("save_session", END)
    return graph_builder.compile()


def build_agent_chat_reply(request: AgentChatRequest) -> AgentChatResponse:
    # 综合项目的主入口：先准备状态，再执行 LangGraph 主图。
    cleaned_message = request.message.strip()
    if not cleaned_message:
        raise ValueError("message 不能为空")

    cleaned_session_id = request.session_id.strip()
    if not cleaned_session_id:
        raise ValueError("session_id 不能为空")

    ensure_llm_config()

    history, session_history_source = build_history_messages(cleaned_session_id)
    session_message_count_before = len(history)
    history_messages = build_langchain_history_messages(history)
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    current_user_message = HumanMessage(content=cleaned_message)

    graph = build_agent_graph()
    result = graph.invoke(
        {
            "message": cleaned_message,
            "session_id": cleaned_session_id,
            "messages": [system_message, *history_messages, current_user_message],
            "index_name": request.index_name.strip(),
            "intent": "fallback",
            "reply": "",
            "stage": "graph_initialized",
            "need_followup": False,
            "followup_question": "",
            "error_message": "",
            "retrieved_chunks": [],
            "retrieved_source_names": [],
            "debug_steps": ["graph_start"],
        }
    )

    session_message_count_after = get_session_message_count(cleaned_session_id)
    session_turn_count_after = session_message_count_after // 2

    debug_info = None
    if request.debug:
        debug_info = AgentDebugInfo(
            used_model=settings.llm_model,
            session_storage="mysql + redis",
            session_history_source=session_history_source,
            session_message_count_before=session_message_count_before,
            session_message_count_after=session_message_count_after,
            session_turn_count_after=session_turn_count_after,
            messages_sent_to_llm=build_debug_messages(result["messages"]),
            graph_steps=result["debug_steps"],
            intent=result["intent"],
            final_stage=result["stage"],
            index_name=result["index_name"],
            retrieved_chunk_count=len(result["retrieved_chunks"]),
            retrieved_source_names=result["retrieved_source_names"],
        )

    return AgentChatResponse(
        reply=result["reply"],
        stage=result["stage"],
        session_id=cleaned_session_id,
        model=settings.llm_model,
        session_turn_count=session_turn_count_after,
        debug=debug_info,
    )
