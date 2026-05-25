import json
import re
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from math import sqrt
from pathlib import Path

import faiss
import numpy as np
from fastapi import UploadFile
from openai import OpenAI
from pypdf import PdfReader

from app.config import settings
from app.schemas.chat import (
    ChunkDemoRequest,
    ChunkDemoResponse,
    ChatCitationItem,
    ChatDemoRequest,
    ChatFaissIndexBuildRequest,
    ChatFaissIndexBuildResponse,
    ChatFaissSearchRequest,
    ChatFaissSearchResponse,
    ChatRagEmbeddingRequest,
    ChatRagEmbeddingResponse,
    ChatRagRequest,
    ChatRagResponse,
    ChatRagSearchRequest,
    ChatRagSearchResponse,
    ChatAgentLoopDemoRequest,
    ChatAgentLoopDemoResponse,
    ChatAgentLoopStepItem,
    ChatAgentRagDemoRequest,
    ChatAgentRagDemoResponse,
    ChatAgentRouteDemoRequest,
    ChatAgentRouteDemoResponse,
    ChatAgentSessionDemoRequest,
    ChatAgentSessionDemoResponse,
    ChatRealRequest,
    ChatSessionRequest,
    ChatSessionResponse,
    ChatSummaryRequest,
    ChatSummaryResponse,
    ChatToolCallItem,
    ChatToolDemoRequest,
    ChatToolDemoResponse,
    ChatTurn,
    ChatUploadIndexResponse,
    ChatVectorIndexBuildRequest,
    ChatVectorIndexBuildResponse,
    ChatVectorIndexListResponse,
    ChatVectorIndexLoadRequest,
    ChatVectorIndexLoadResponse,
    ChatVectorIndexSaveRequest,
    ChatVectorIndexSaveResponse,
    ChatVectorSearchRequest,
    ChatVectorSearchResponse,
    ChatVectorSearchWithCitationsResponse,
    KnowledgeDocumentInput,
    SavedVectorIndexInfo,
)


@dataclass
class VectorChunkItem:
    # 一个 chunk 对应一段文本、它的 embedding 向量，以及来源信息。
    text: str
    embedding: list[float]
    source_name: str
    chunk_index: int


@dataclass
class LocalVectorIndex:
    # 一个本地向量索引里，保存切分参数、来源信息和所有向量条目。
    index_name: str
    source_text: str
    source_name: str
    source_count: int
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    items: list[VectorChunkItem]


@dataclass
class LocalFaissIndex:
    # 一个最小 FAISS 索引对象：前处理结果和 FAISS 检索结构放在一起。
    index_name: str
    source_text: str
    source_count: int
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    items: list[VectorChunkItem]
    vector_dimension: int
    faiss_index: faiss.IndexFlatIP


# 用内存字典保存会话历史。
session_store: dict[str, list[ChatTurn]] = defaultdict(list)

# 用内存字典保存本地向量索引。
vector_index_store: dict[str, LocalVectorIndex] = {}

# 用内存字典保存 FAISS 索引。
faiss_index_store: dict[str, LocalFaissIndex] = {}


def load_demo_knowledge() -> str:
    # 读取一份本地知识文本，作为演示用知识库。
    knowledge_file = Path(__file__).resolve().parent.parent / "knowledge" / "fastapi_intro.txt"
    return knowledge_file.read_text(encoding="utf-8").strip()


def normalize_chunk_strategy(chunk_strategy: str) -> str:
    # 统一处理切分策略，避免大小写或空格带来的干扰。
    normalized = chunk_strategy.strip().lower()
    if normalized not in {"fixed", "paragraph"}:
        raise ValueError("chunk_strategy 只支持 fixed 或 paragraph")
    return normalized


def split_text_fixed(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    # 固定长度切分：按字符长度直接切。
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    step = chunk_size - chunk_overlap
    chunks: list[str] = []

    for start in range(0, len(cleaned_text), step):
        chunk = cleaned_text[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(chunk)
        if start + chunk_size >= len(cleaned_text):
            break

    return chunks


def split_paragraph_to_fixed_chunks(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # 如果单个段落太长，再退化成固定长度切分。
    return split_text_fixed(paragraph, chunk_size, chunk_overlap)


def split_text_by_paragraphs(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    # 按段落优先切分。
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned_text) if paragraph.strip()]
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(split_paragraph_to_fixed_chunks(paragraph, chunk_size, chunk_overlap))
            continue

        if not current_chunk:
            current_chunk = paragraph
            continue

        candidate = f"{current_chunk}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    chunk_strategy: str = "fixed",
) -> list[str]:
    # 统一切分入口：根据策略决定走哪一种切分方式。
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)
    if normalized_strategy == "paragraph":
        return split_text_by_paragraphs(text, chunk_size, chunk_overlap)
    return split_text_fixed(text, chunk_size, chunk_overlap)


def normalize_documents(documents: list[KnowledgeDocumentInput]) -> list[KnowledgeDocumentInput]:
    # 清洗多文档输入，过滤空白文本，并统一去掉前后空格。
    normalized_documents: list[KnowledgeDocumentInput] = []

    for document in documents:
        cleaned_source_name = document.source_name.strip()
        cleaned_text = document.text.strip()
        if not cleaned_source_name:
            raise ValueError("source_documents 里的 source_name 不能为空")
        if not cleaned_text:
            raise ValueError(f"文档 {cleaned_source_name} 的 text 不能为空")

        normalized_documents.append(
            KnowledgeDocumentInput(
                source_name=cleaned_source_name,
                text=cleaned_text,
            )
        )

    return normalized_documents


def normalize_source_name_filter(source_name_filter: str) -> str:
    # 统一处理来源文档过滤条件；空字符串表示不过滤。
    return source_name_filter.strip()


def build_float32_matrix(vectors: list[list[float]]) -> np.ndarray:
    # 把 Python 的二维向量列表转换成 FAISS 需要的 float32 矩阵。
    if not vectors:
        raise ValueError("向量列表不能为空")

    matrix = np.array(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("向量矩阵必须是二维结构")
    return matrix


def build_faiss_index(vectors: list[list[float]]) -> tuple[faiss.IndexFlatIP, int]:
    # 用最小可学版的 IndexFlatIP 构建 FAISS 索引。
    # 这里先做 L2 归一化，再用内积近似余弦相似度。
    matrix = build_float32_matrix(vectors)
    vector_dimension = matrix.shape[1]
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(vector_dimension)
    index.add(matrix)
    return index, vector_dimension


def select_relevant_chunks(question: str, chunks: list[str], top_k: int) -> list[str]:
    # 最小版关键词检索：统计问题中的关键词在每个 chunk 里命中了多少次。
    question_keywords = [word for word in question.strip().split() if word]
    scored_chunks: list[tuple[int, str]] = []

    for chunk in chunks:
        score = 0
        lower_chunk = chunk.lower()
        for word in question_keywords:
            if word.lower() in lower_chunk:
                score += 1
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for score, chunk in scored_chunks if score > 0]

    if selected:
        return selected[:top_k]

    return [chunk for _, chunk in scored_chunks[:top_k]]


def get_llm_client() -> OpenAI:
    # 创建一个 OpenAI 兼容客户端，这里实际连的是百炼兼容接口。
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
    )


def get_embedding_client() -> OpenAI:
    # embedding 和聊天模型走同一个兼容地址，只是模型名称不同。
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
    )


def ensure_api_key() -> None:
    # 统一检查 API Key。
    if not settings.dashscope_api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，请先在 .env 中配置百炼 API Key")


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当用户询问现在几点、今天日期、当前时间时调用",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mock_weather",
            "description": "当用户询问某个城市的天气、气温、是否带伞时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询天气的城市，例如杭州、上海、北京",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_demo_knowledge",
            "description": "当用户询问 FastAPI、知识库内容、课程笔记或项目文档相关问题时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要在演示知识库里检索的问题",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


def get_current_time_tool() -> str:
    # 本地时间工具：返回当前机器时间。
    now = datetime.now()
    result = {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": f"星期{'一二三四五六日'[now.weekday()]}",
    }
    return json.dumps(result, ensure_ascii=False)


def get_mock_weather_tool(city: str) -> str:
    # 本地天气工具：先用固定假数据演示工具调用流程。
    normalized_city = city.strip()
    if not normalized_city:
        raise ValueError("get_mock_weather 工具缺少 city 参数")

    weather_map = {
        "杭州": {"weather": "小雨", "temperature": "22C", "advice": "建议带伞"},
        "上海": {"weather": "多云", "temperature": "26C", "advice": "可以正常出行"},
        "北京": {"weather": "晴", "temperature": "28C", "advice": "注意防晒"},
        "深圳": {"weather": "阵雨", "temperature": "29C", "advice": "最好带伞"},
    }
    city_weather = weather_map.get(
        normalized_city,
        {
            "weather": "未知",
            "temperature": "未知",
            "advice": "当前演示工具暂不支持这个城市，请改问杭州、上海、北京或深圳",
        },
    )

    result = {
        "city": normalized_city,
        **city_weather,
    }
    return json.dumps(result, ensure_ascii=False)


def search_demo_knowledge_tool(question: str) -> str:
    # 最小知识库检索工具：复用之前的 RAG 切分和关键词检索逻辑，返回命中的片段。
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("search_demo_knowledge 工具缺少 question 参数")

    knowledge = load_demo_knowledge()
    chunk_size = 120
    chunk_overlap = 20
    chunk_strategy = "paragraph"
    chunks = split_text_into_chunks(
        knowledge,
        chunk_size,
        chunk_overlap,
        chunk_strategy,
    )
    selected_chunks = select_relevant_chunks(cleaned_question, chunks, top_k=2)

    result = {
        "question": cleaned_question,
        "chunk_strategy": chunk_strategy,
        "selected_chunks": selected_chunks,
    }
    return json.dumps(result, ensure_ascii=False)


def execute_demo_tool(tool_name: str, tool_args: dict) -> str:
    # 工具执行器：根据工具名真正调用本地函数。
    if tool_name == "get_current_time":
        return get_current_time_tool()
    if tool_name == "get_mock_weather":
        city = str(tool_args.get("city", "") or "").strip()
        return get_mock_weather_tool(city)
    if tool_name == "search_demo_knowledge":
        question = str(tool_args.get("question", "") or "").strip()
        return search_demo_knowledge_tool(question)
    raise ValueError(f"暂不支持的工具：{tool_name}")


def serialize_tool_calls(tool_calls) -> list[dict]:
    # 把模型返回的 tool_calls 转成普通字典，方便重新塞回 messages。
    serialized_tool_calls: list[dict] = []
    for tool_call in tool_calls:
        serialized_tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    return serialized_tool_calls


def parse_tool_args(raw_arguments: str) -> dict:
    # 把模型返回的 JSON 字符串参数解析成 Python 字典。
    try:
        tool_args = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("模型返回的工具参数不是合法 JSON") from exc

    if not isinstance(tool_args, dict):
        raise ValueError("模型返回的工具参数必须是 JSON 对象")
    return tool_args


def normalize_tool_args_for_response(tool_args: dict) -> dict[str, str | None]:
    # 把工具参数整理成更适合返回给前端查看的格式。
    return {key: None if value is None else str(value) for key, value in tool_args.items()}


def build_messages_from_history(history: list[ChatTurn], system_prompt: str) -> list[dict[str, str]]:
    # 把 session_store 里的历史消息恢复成发给模型的 messages 结构。
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    return messages


def build_demo_reply(request: ChatDemoRequest) -> str:
    # 演示接口：简单拼接回复。
    message = request.message.strip()
    if message.lower() == "error":
        raise ValueError("演示异常：你输入了 error")

    return f"{settings.chat_reply_prefix}{message}"


def build_real_reply(request: ChatRealRequest) -> tuple[str, str]:
    # 真实模型调用前，先做最基本的输入清洗。
    message = request.message.strip()
    ensure_api_key()

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "你是一名耐心、清晰、适合初学者的 AI 编程老师。"},
            {"role": "user", "content": message},
        ],
        reasoning_effort="high",
    )

    reply_text = completion.choices[0].message.content or ""
    return reply_text, settings.llm_model


def build_tool_demo_reply(request: ChatToolDemoRequest) -> ChatToolDemoResponse:
    # 最小 Tool Calling 演示：
    # 先让模型决定要不要调工具；
    # 如果调了，后端执行工具；
    # 再把工具结果交回模型，让模型组织最终回答。
    message = request.message.strip()
    ensure_api_key()

    client = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名 AI 助手。"
                "如果用户在问时间，就调用 get_current_time。"
                "如果用户在问天气、气温、是否带伞，就调用 get_mock_weather。"
                "如果问题不需要工具，就直接回答。"
            ),
        },
        {"role": "user", "content": message},
    ]

    first_completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    assistant_message = first_completion.choices[0].message
    tool_calls = assistant_message.tool_calls or []
    if not tool_calls:
        return ChatToolDemoResponse(
            reply=assistant_message.content or "",
            model=settings.llm_model,
            used_tool=False,
            tool_calls=[],
        )

    serialized_tool_calls = serialize_tool_calls(tool_calls)
    executed_tool_calls: list[ChatToolCallItem] = []

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": serialized_tool_calls,
        }
    )

    for tool_call in tool_calls:
        raw_arguments = tool_call.function.arguments or "{}"
        tool_args = parse_tool_args(raw_arguments)
        tool_result = execute_demo_tool(tool_call.function.name, tool_args)
        executed_tool_calls.append(
            ChatToolCallItem(
                tool_name=tool_call.function.name,
                tool_args=normalize_tool_args_for_response(tool_args),
                tool_result=tool_result,
            )
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

    second_completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    reply_text = second_completion.choices[0].message.content or ""

    return ChatToolDemoResponse(
        reply=reply_text,
        model=settings.llm_model,
        used_tool=True,
        tool_calls=executed_tool_calls,
    )


def build_agent_loop_demo_reply(request: ChatAgentLoopDemoRequest) -> ChatAgentLoopDemoResponse:
    # 最小 Agent 循环演示：
    # 不再固定“最多只调用一轮工具”，而是允许模型在一个循环里多次决定是否继续调工具。
    message = request.message.strip()
    ensure_api_key()

    client = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名会分步调用工具的 AI 助手。"
                "如果需要先查时间，再结合时间判断是否查天气，可以分多步调用工具。"
                "如果信息已经足够，就直接给出最终答案，不要继续调用工具。"
            ),
        },
        {"role": "user", "content": message},
    ]

    executed_steps: list[ChatAgentLoopStepItem] = []

    for step in range(1, request.max_steps + 1):
        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return ChatAgentLoopDemoResponse(
                reply=assistant_message.content or "",
                model=settings.llm_model,
                used_tool=bool(executed_steps),
                total_steps=len(executed_steps),
                stopped_by_max_steps=False,
                steps=executed_steps,
            )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": serialize_tool_calls(tool_calls),
            }
        )

        for tool_call in tool_calls:
            raw_arguments = tool_call.function.arguments or "{}"
            tool_args = parse_tool_args(raw_arguments)
            tool_result = execute_demo_tool(tool_call.function.name, tool_args)

            executed_steps.append(
                ChatAgentLoopStepItem(
                    step=step,
                    tool_name=tool_call.function.name,
                    tool_args=normalize_tool_args_for_response(tool_args),
                    tool_result=tool_result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    final_completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    final_reply = final_completion.choices[0].message.content or ""

    return ChatAgentLoopDemoResponse(
        reply=final_reply,
        model=settings.llm_model,
        used_tool=bool(executed_steps),
        total_steps=len(executed_steps),
        stopped_by_max_steps=True,
        steps=executed_steps,
    )


def build_agent_session_demo_reply(request: ChatAgentSessionDemoRequest) -> ChatAgentSessionDemoResponse:
    # 带会话记忆的最小 Agent：
    # 在同一个 session_id 下，把之前的用户问题和助手回答一起带上，
    # 让模型能结合历史上下文继续决定是否调用工具。
    cleaned_message = request.message.strip()
    ensure_api_key()

    system_prompt = (
        "你是一名带会话记忆、会分步调用工具的 AI 助手。"
        "你要结合当前 session 里的历史消息理解用户的省略表达。"
        "如果历史里已经提到城市、任务目标等信息，本轮可以继续沿用。"
        "如果需要更多信息，就调用工具；如果信息已经足够，就直接回答。"
    )

    history = session_store[request.session_id]
    messages = build_messages_from_history(history, system_prompt)
    messages.append({"role": "user", "content": cleaned_message})

    client = get_llm_client()
    executed_steps: list[ChatAgentLoopStepItem] = []

    for step in range(1, request.max_steps + 1):
        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            final_reply = assistant_message.content or ""
            history.append(ChatTurn(role="user", content=cleaned_message))
            history.append(ChatTurn(role="assistant", content=final_reply))

            return ChatAgentSessionDemoResponse(
                session_id=request.session_id,
                reply=final_reply,
                model=settings.llm_model,
                used_tool=bool(executed_steps),
                total_steps=len(executed_steps),
                stopped_by_max_steps=False,
                history_count=len(history),
                steps=executed_steps,
            )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": serialize_tool_calls(tool_calls),
            }
        )

        for tool_call in tool_calls:
            raw_arguments = tool_call.function.arguments or "{}"
            tool_args = parse_tool_args(raw_arguments)
            tool_result = execute_demo_tool(tool_call.function.name, tool_args)

            executed_steps.append(
                ChatAgentLoopStepItem(
                    step=step,
                    tool_name=tool_call.function.name,
                    tool_args=normalize_tool_args_for_response(tool_args),
                    tool_result=tool_result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    final_completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    final_reply = final_completion.choices[0].message.content or ""
    history.append(ChatTurn(role="user", content=cleaned_message))
    history.append(ChatTurn(role="assistant", content=final_reply))

    return ChatAgentSessionDemoResponse(
        session_id=request.session_id,
        reply=final_reply,
        model=settings.llm_model,
        used_tool=bool(executed_steps),
        total_steps=len(executed_steps),
        stopped_by_max_steps=True,
        history_count=len(history),
        steps=executed_steps,
    )


def build_agent_rag_demo_reply(request: ChatAgentRagDemoRequest) -> ChatAgentRagDemoResponse:
    # Agent + RAG 最小演示：
    # 给 Agent 增加一个“知识库检索工具”，让模型自己判断什么时候需要查知识库。
    cleaned_message = request.message.strip()
    ensure_api_key()

    client = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名会调用工具的 AI 助手。"
                "如果用户询问时间，调用 get_current_time。"
                "如果用户询问天气、气温、是否带伞，调用 get_mock_weather。"
                "如果用户询问 FastAPI、演示知识库、课程知识点或项目文档内容，调用 search_demo_knowledge。"
                "如果问题不需要工具，就直接回答。"
                "如果你调用了 search_demo_knowledge，请优先基于检索结果回答。"
            ),
        },
        {"role": "user", "content": cleaned_message},
    ]

    executed_steps: list[ChatAgentLoopStepItem] = []

    for step in range(1, request.max_steps + 1):
        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return ChatAgentRagDemoResponse(
                reply=assistant_message.content or "",
                model=settings.llm_model,
                used_tool=bool(executed_steps),
                total_steps=len(executed_steps),
                stopped_by_max_steps=False,
                steps=executed_steps,
            )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": serialize_tool_calls(tool_calls),
            }
        )

        for tool_call in tool_calls:
            raw_arguments = tool_call.function.arguments or "{}"
            tool_args = parse_tool_args(raw_arguments)
            tool_result = execute_demo_tool(tool_call.function.name, tool_args)

            executed_steps.append(
                ChatAgentLoopStepItem(
                    step=step,
                    tool_name=tool_call.function.name,
                    tool_args=normalize_tool_args_for_response(tool_args),
                    tool_result=tool_result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    final_completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    final_reply = final_completion.choices[0].message.content or ""

    return ChatAgentRagDemoResponse(
        reply=final_reply,
        model=settings.llm_model,
        used_tool=bool(executed_steps),
        total_steps=len(executed_steps),
        stopped_by_max_steps=True,
        steps=executed_steps,
    )


def build_agent_route_demo_reply(request: ChatAgentRouteDemoRequest) -> ChatAgentRouteDemoResponse:
    # 多工具路由 + 缺参追问演示：
    # 如果用户信息不足，就先追问，不要盲目调用工具；
    # 如果信息已经足够，再选择合适工具执行。
    cleaned_message = request.message.strip()
    ensure_api_key()

    client = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名会调用工具的 AI 助手。"
                "你可以调用 get_current_time、get_mock_weather、search_demo_knowledge。"
                "如果用户询问天气，但没有明确城市，你必须先追问用户城市，不能直接调用 get_mock_weather。"
                "如果用户询问知识库内容不明确，你可以先追问用户想了解哪方面。"
                "只有在参数足够时，才调用工具。"
                "如果问题不需要工具，就直接回答。"
            ),
        },
        {"role": "user", "content": cleaned_message},
    ]

    executed_steps: list[ChatAgentLoopStepItem] = []

    for step in range(1, request.max_steps + 1):
        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return ChatAgentRouteDemoResponse(
                reply=assistant_message.content or "",
                model=settings.llm_model,
                used_tool=bool(executed_steps),
                total_steps=len(executed_steps),
                stopped_by_max_steps=False,
                steps=executed_steps,
            )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": serialize_tool_calls(tool_calls),
            }
        )

        for tool_call in tool_calls:
            raw_arguments = tool_call.function.arguments or "{}"
            tool_args = parse_tool_args(raw_arguments)
            tool_result = execute_demo_tool(tool_call.function.name, tool_args)

            executed_steps.append(
                ChatAgentLoopStepItem(
                    step=step,
                    tool_name=tool_call.function.name,
                    tool_args=normalize_tool_args_for_response(tool_args),
                    tool_result=tool_result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    final_completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    final_reply = final_completion.choices[0].message.content or ""

    return ChatAgentRouteDemoResponse(
        reply=final_reply,
        model=settings.llm_model,
        used_tool=bool(executed_steps),
        total_steps=len(executed_steps),
        stopped_by_max_steps=True,
        steps=executed_steps,
    )


def build_summary_reply(request: ChatSummaryRequest) -> ChatSummaryResponse:
    # 结构化输出示例：让模型按 JSON 结构返回总结结果。
    message = request.message.strip()
    ensure_api_key()

    client = get_llm_client()
    completion = client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": "你是一名擅长总结信息的 AI 助手。请严格按给定结构输出。",
            },
            {"role": "user", "content": message},
        ],
        response_format=ChatSummaryResponse,
    )

    parsed_result = completion.choices[0].message.parsed
    if parsed_result is None:
        raise ValueError("模型没有返回可解析的结构化结果")

    return ChatSummaryResponse(
        title=parsed_result.title,
        summary=parsed_result.summary,
        keywords=parsed_result.keywords,
        model=settings.llm_model,
    )


def build_stream_reply(message: str) -> Generator[str, None, None]:
    # 流式输出示例：模型生成一点，就往外返回一点。
    cleaned_message = message.strip()
    ensure_api_key()

    client = get_llm_client()
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "你是一名耐心、清晰、适合初学者的 AI 编程老师。"},
            {"role": "user", "content": cleaned_message},
        ],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        content = delta.content or ""
        if content:
            yield content


def build_session_reply(request: ChatSessionRequest) -> ChatSessionResponse:
    # 多轮对话示例：把当前会话之前的消息一起传给模型。
    cleaned_message = request.message.strip()
    ensure_api_key()

    history = session_store[request.session_id]
    history.append(ChatTurn(role="user", content=cleaned_message))

    messages = [
        {"role": "system", "content": "你是一名耐心、清晰、适合初学者的 AI 编程老师。"},
    ]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )

    reply_text = completion.choices[0].message.content or ""
    history.append(ChatTurn(role="assistant", content=reply_text))

    return ChatSessionResponse(
        reply=reply_text,
        history_count=len(history),
        model=settings.llm_model,
    )


def build_rag_reply(request: ChatRagRequest) -> ChatRagResponse:
    # 最小版 RAG：先读取本地知识，再让模型基于知识回答问题。
    question = request.question.strip()
    ensure_api_key()

    knowledge = load_demo_knowledge()
    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名知识库问答助手。"
                    "请优先依据我提供的知识内容回答。"
                    "如果知识内容中没有明确答案，就明确说明“提供的知识中没有直接提到”。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"知识内容如下：\n{knowledge}\n\n"
                    f"用户问题如下：\n{question}"
                ),
            },
        ],
    )

    answer_text = completion.choices[0].message.content or ""
    return ChatRagResponse(
        answer=answer_text,
        knowledge=knowledge,
        model=settings.llm_model,
    )


def build_chunk_demo_reply(request: ChunkDemoRequest) -> ChunkDemoResponse:
    # 文本切分演示。
    cleaned_text = request.text.strip()
    chunk_strategy = normalize_chunk_strategy(request.chunk_strategy)
    chunks = split_text_into_chunks(
        cleaned_text,
        request.chunk_size,
        request.chunk_overlap,
        chunk_strategy,
    )

    return ChunkDemoResponse(
        original_length=len(cleaned_text),
        chunk_count=len(chunks),
        chunk_strategy=chunk_strategy,
        chunks=chunks,
    )


def build_rag_search_reply(request: ChatRagSearchRequest) -> ChatRagSearchResponse:
    # 关键词检索版 RAG。
    question = request.question.strip()
    ensure_api_key()

    knowledge = load_demo_knowledge()
    chunk_strategy = normalize_chunk_strategy(request.chunk_strategy)
    chunks = split_text_into_chunks(
        knowledge,
        request.chunk_size,
        request.chunk_overlap,
        chunk_strategy,
    )
    selected_chunks = select_relevant_chunks(question, chunks, request.top_k)
    joined_chunks = "\n\n".join(selected_chunks)

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名知识库问答助手。"
                    "请优先依据我提供的检索片段回答。"
                    "如果检索片段中没有明确答案，就明确说明“检索片段中没有直接提到”。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"检索到的相关片段如下：\n{joined_chunks}\n\n"
                    f"用户问题如下：\n{question}"
                ),
            },
        ],
    )

    answer_text = completion.choices[0].message.content or ""
    return ChatRagSearchResponse(
        answer=answer_text,
        selected_chunks=selected_chunks,
        chunk_strategy=chunk_strategy,
        model=settings.llm_model,
    )


def get_embeddings(texts: list[str]) -> list[list[float]]:
    # 批量把文本转成 embedding 向量。
    client = get_embedding_client()
    batch_size = 10
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch_texts,
        )
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    # 计算两个向量的余弦相似度。
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sqrt(sum(a * a for a in vector_a))
    norm_b = sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def select_relevant_chunks_by_embedding(
    question: str,
    chunks: list[str],
    top_k: int,
) -> list[tuple[float, str]]:
    # 语义检索教学版：把问题和所有 chunk 一起转成向量，再比较相似度。
    embeddings = get_embeddings([question, *chunks])
    question_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    scored_chunks: list[tuple[float, str]] = []
    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_embedding)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return scored_chunks[:top_k]


def build_rag_embedding_reply(request: ChatRagEmbeddingRequest) -> ChatRagEmbeddingResponse:
    # 语义检索版 RAG：每次请求都重新切分、重算 chunk embedding。
    question = request.question.strip()
    ensure_api_key()

    knowledge = load_demo_knowledge()
    chunk_strategy = normalize_chunk_strategy(request.chunk_strategy)
    chunks = split_text_into_chunks(
        knowledge,
        request.chunk_size,
        request.chunk_overlap,
        chunk_strategy,
    )
    scored_chunks = select_relevant_chunks_by_embedding(question, chunks, request.top_k)
    selected_chunks = [chunk for _, chunk in scored_chunks]
    scores = [round(score, 4) for score, _ in scored_chunks]
    joined_chunks = "\n\n".join(selected_chunks)

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名知识库问答助手。"
                    "请优先依据我提供的语义检索片段回答。"
                    "如果检索片段中没有明确答案，就明确说明“检索片段中没有直接提到”。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"语义检索到的相关片段如下：\n{joined_chunks}\n\n"
                    f"用户问题如下：\n{question}"
                ),
            },
        ],
    )

    answer_text = completion.choices[0].message.content or ""
    return ChatRagEmbeddingResponse(
        answer=answer_text,
        selected_chunks=selected_chunks,
        scores=scores,
        chunk_strategy=chunk_strategy,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


def build_local_vector_index(
    index_name: str,
    source_text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    chunk_strategy: str = "fixed",
    source_name: str = "inline_text",
    source_documents: list[KnowledgeDocumentInput] | None = None,
) -> LocalVectorIndex:
    # 建索引阶段：既支持单文档，也支持多文档共同组成一个知识库索引。
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)
    normalized_input_documents = normalize_documents(source_documents or [])
    prepared_documents = normalized_input_documents
    cleaned_source_text = source_text.strip()
    cleaned_source_name = source_name.strip() or "inline_text"

    if prepared_documents:
        combined_source_text = "\n\n".join(document.text for document in prepared_documents)
        source_count = len(prepared_documents)
    else:
        if not cleaned_source_text:
            raise ValueError("索引文本不能为空，无法构建向量索引")
        prepared_documents = [
            KnowledgeDocumentInput(
                source_name=cleaned_source_name,
                text=cleaned_source_text,
            )
        ]
        combined_source_text = cleaned_source_text
        source_count = 1

    all_chunks_with_source: list[tuple[str, str, int]] = []
    for document in prepared_documents:
        document_chunks = split_text_into_chunks(
            document.text,
            chunk_size,
            chunk_overlap,
            normalized_strategy,
        )
        for chunk_index, chunk_text in enumerate(document_chunks, start=1):
            all_chunks_with_source.append((chunk_text, document.source_name, chunk_index))

    if not all_chunks_with_source:
        raise ValueError("没有可用于构建索引的 chunk")

    chunk_texts = [chunk_text for chunk_text, _, _ in all_chunks_with_source]
    chunk_embeddings = get_embeddings(chunk_texts)
    items: list[VectorChunkItem] = []

    for (chunk_text, item_source_name, chunk_index), chunk_embedding in zip(
        all_chunks_with_source,
        chunk_embeddings,
    ):
        items.append(
            VectorChunkItem(
                text=chunk_text,
                embedding=chunk_embedding,
                source_name=item_source_name,
                chunk_index=chunk_index,
            )
        )

    index = LocalVectorIndex(
        index_name=index_name,
        source_text=combined_source_text,
        source_name=cleaned_source_name,
        source_count=source_count,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=normalized_strategy,
        items=items,
    )
    vector_index_store[index_name] = index
    return index


def build_local_faiss_index(request: ChatFaissIndexBuildRequest) -> LocalFaissIndex:
    # FAISS 版建索引：
    # 前处理仍然复用现有的切分、多文档、embedding 逻辑，
    # 只把“检索结构”替换成 FAISS。
    source_text = request.source_text.strip()
    if not source_text and not request.source_documents:
        source_text = load_demo_knowledge()

    vector_index = build_local_vector_index(
        index_name=request.index_name.strip(),
        source_text=source_text,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        chunk_strategy=request.chunk_strategy,
        source_name="inline_text",
        source_documents=request.source_documents,
    )

    item_embeddings = [item.embedding for item in vector_index.items]
    faiss_index, vector_dimension = build_faiss_index(item_embeddings)

    index = LocalFaissIndex(
        index_name=vector_index.index_name,
        source_text=vector_index.source_text,
        source_count=vector_index.source_count,
        chunk_size=vector_index.chunk_size,
        chunk_overlap=vector_index.chunk_overlap,
        chunk_strategy=vector_index.chunk_strategy,
        items=vector_index.items,
        vector_dimension=vector_dimension,
        faiss_index=faiss_index,
    )
    faiss_index_store[index.index_name] = index
    return index


def search_local_vector_index(
    question: str,
    index: LocalVectorIndex,
    top_k: int,
    score_threshold: float = 0.0,
    source_name_filter: str = "",
) -> list[tuple[float, VectorChunkItem]]:
    # 查询阶段：只计算问题向量，然后和已保存的 chunk 向量逐个比较。
    # 这里额外支持：
    # 1. score_threshold：过滤掉相似度太低的结果
    # 2. source_name_filter：只在指定来源文档中检索
    normalized_source_name_filter = normalize_source_name_filter(source_name_filter)
    question_embedding = get_embeddings([question])[0]
    scored_chunks: list[tuple[float, VectorChunkItem]] = []

    for item in index.items:
        if normalized_source_name_filter and item.source_name != normalized_source_name_filter:
            continue
        score = cosine_similarity(question_embedding, item.embedding)
        if score < score_threshold:
            continue
        scored_chunks.append((score, item))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return scored_chunks[:top_k]


def search_local_faiss_index(
    question: str,
    index: LocalFaissIndex,
    top_k: int,
    score_threshold: float = 0.0,
    source_name_filter: str = "",
) -> list[tuple[float, VectorChunkItem]]:
    # FAISS 版检索：
    # 先让 FAISS 做近邻搜索，再结合教学版过滤逻辑做后处理。
    normalized_source_name_filter = normalize_source_name_filter(source_name_filter)
    question_embedding = get_embeddings([question])[0]
    query_matrix = build_float32_matrix([question_embedding])
    faiss.normalize_L2(query_matrix)

    search_limit = len(index.items)
    scores_matrix, positions_matrix = index.faiss_index.search(query_matrix, search_limit)
    scores = scores_matrix[0]
    positions = positions_matrix[0]

    scored_chunks: list[tuple[float, VectorChunkItem]] = []
    for score, position in zip(scores, positions):
        if position < 0:
            continue

        item = index.items[position]
        if normalized_source_name_filter and item.source_name != normalized_source_name_filter:
            continue
        if score < score_threshold:
            continue

        scored_chunks.append((float(score), item))
        if len(scored_chunks) >= top_k:
            break

    return scored_chunks


def get_project_root() -> Path:
    # 取到项目根目录。
    return Path(__file__).resolve().parents[2]


def get_vector_index_storage_dir() -> Path:
    # 所有本地索引文件都统一放到 storage/vector_indexes 目录下。
    storage_dir = get_project_root() / "storage" / "vector_indexes"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def sanitize_index_name(index_name: str) -> str:
    # Windows 文件名不能包含某些特殊字符，这里统一替换成下划线。
    sanitized_name = re.sub(r'[<>:"/\\|?*]', "_", index_name.strip())
    if not sanitized_name:
        raise ValueError("索引名称不能为空")
    return sanitized_name


def get_vector_index_file_path(index_name: str) -> Path:
    # 根据索引名计算本地 JSON 文件路径。
    safe_index_name = sanitize_index_name(index_name)
    return get_vector_index_storage_dir() / f"{safe_index_name}.json"


def serialize_local_vector_index(index: LocalVectorIndex) -> dict:
    # 把内存中的 dataclass 对象转成可写入 JSON 的普通字典。
    return {
        "index_name": index.index_name,
        "source_text": index.source_text,
        "source_name": index.source_name,
        "source_count": index.source_count,
        "chunk_size": index.chunk_size,
        "chunk_overlap": index.chunk_overlap,
        "chunk_strategy": index.chunk_strategy,
        "items": [
            {
                "text": item.text,
                "embedding": item.embedding,
                "source_name": item.source_name,
                "chunk_index": item.chunk_index,
            }
            for item in index.items
        ],
    }


def deserialize_local_vector_index(data: dict) -> LocalVectorIndex:
    # 把 JSON 里的普通字典重新还原成 LocalVectorIndex 对象。
    items = [
        VectorChunkItem(
            text=item["text"],
            embedding=item["embedding"],
            source_name=item.get("source_name", data.get("source_name", "unknown")),
            chunk_index=item.get("chunk_index", 0),
        )
        for item in data["items"]
    ]
    return LocalVectorIndex(
        index_name=data["index_name"],
        source_text=data["source_text"],
        source_name=data.get("source_name", "unknown"),
        source_count=data.get("source_count", 1),
        chunk_size=data["chunk_size"],
        chunk_overlap=data.get("chunk_overlap", 0),
        chunk_strategy=data.get("chunk_strategy", "fixed"),
        items=items,
    )


def save_local_vector_index(index: LocalVectorIndex) -> Path:
    # 把内存索引写入本地 JSON 文件，实现最小版持久化。
    file_path = get_vector_index_file_path(index.index_name)
    payload = serialize_local_vector_index(index)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file_path


def load_local_vector_index(index_name: str) -> tuple[LocalVectorIndex, Path]:
    # 从本地 JSON 文件读取索引，并重新放回内存字典。
    file_path = get_vector_index_file_path(index_name)
    if not file_path.exists():
        raise ValueError(f"索引文件不存在，请先保存索引：{file_path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    index = deserialize_local_vector_index(data)
    vector_index_store[index.index_name] = index
    return index, file_path


def list_saved_vector_indexes() -> list[SavedVectorIndexInfo]:
    # 扫描本地目录下所有已保存的 JSON 索引文件。
    storage_dir = get_vector_index_storage_dir()
    indexes: list[SavedVectorIndexInfo] = []

    for file_path in storage_dir.glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        indexes.append(
            SavedVectorIndexInfo(
                index_name=data["index_name"],
                file_path=str(file_path),
                chunk_count=len(data["items"]),
                source_count=data.get("source_count", 1),
                chunk_size=data["chunk_size"],
                chunk_overlap=data.get("chunk_overlap", 0),
                chunk_strategy=data.get("chunk_strategy", "fixed"),
            )
        )

    indexes.sort(key=lambda item: item.index_name)
    return indexes


def parse_txt_or_md_content(file_name: str, file_bytes: bytes) -> str:
    # txt 和 md 都先按 utf-8 读取。
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"文件 {file_name} 不是有效的 UTF-8 文本，请先转换编码") from exc


def parse_pdf_content(file_name: str, file_bytes: bytes) -> str:
    # 读取 PDF 的每一页文本，再拼接成完整知识内容。
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"文件 {file_name} 不是有效的 PDF，无法解析") from exc

    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")

    content = "\n".join(page_texts).strip()
    if not content:
        raise ValueError(f"文件 {file_name} 没有可提取的文本内容")
    return content


def extract_text_from_upload(file_name: str, file_bytes: bytes) -> tuple[str, str]:
    # 根据文件后缀选择不同解析方式，当前支持 txt / md / pdf。
    extension = Path(file_name).suffix.lower()
    if extension == ".txt":
        return parse_txt_or_md_content(file_name, file_bytes), extension
    if extension == ".md":
        return parse_txt_or_md_content(file_name, file_bytes), extension
    if extension == ".pdf":
        return parse_pdf_content(file_name, file_bytes), extension

    raise ValueError("当前只支持上传 .txt、.md、.pdf 文件")


async def build_upload_index_reply(
    index_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
    file: UploadFile,
) -> ChatUploadIndexResponse:
    # 上传文件后，先解析文本，再直接建立本地向量索引。
    ensure_api_key()

    cleaned_index_name = index_name.strip()
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)
    if not cleaned_index_name:
        raise ValueError("索引名称不能为空")

    if chunk_size < 20 or chunk_size > 1000:
        raise ValueError("chunk_size 必须在 20 到 1000 之间")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    if not file.filename:
        raise ValueError("上传文件必须带有文件名")

    file_bytes = await file.read()
    if not file_bytes:
        raise ValueError("上传文件不能为空")

    source_text, extension = extract_text_from_upload(file.filename, file_bytes)
    index = build_local_vector_index(
        cleaned_index_name,
        source_text,
        chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=normalized_strategy,
        source_name=file.filename,
    )

    return ChatUploadIndexResponse(
        index_name=index.index_name,
        file_name=file.filename,
        file_extension=extension,
        source_length=len(index.source_text),
        source_count=index.source_count,
        chunk_count=len(index.items),
        chunk_size=index.chunk_size,
        chunk_overlap=index.chunk_overlap,
        chunk_strategy=index.chunk_strategy,
        embedding_model=settings.embedding_model,
    )


def build_faiss_index_reply(request: ChatFaissIndexBuildRequest) -> ChatFaissIndexBuildResponse:
    # FAISS 版建库入口。
    ensure_api_key()

    index = build_local_faiss_index(request)
    return ChatFaissIndexBuildResponse(
        index_name=index.index_name,
        source_length=len(index.source_text),
        source_count=index.source_count,
        chunk_count=len(index.items),
        chunk_size=index.chunk_size,
        chunk_overlap=index.chunk_overlap,
        chunk_strategy=index.chunk_strategy,
        vector_dimension=index.vector_dimension,
        embedding_model=settings.embedding_model,
    )


def build_vector_index_reply(request: ChatVectorIndexBuildRequest) -> ChatVectorIndexBuildResponse:
    # 预处理阶段入口：先把知识库变成可检索的本地向量索引。
    ensure_api_key()

    index_name = request.index_name.strip()
    source_text = request.source_text.strip()
    if not source_text and not request.source_documents:
        source_text = load_demo_knowledge()

    chunk_strategy = normalize_chunk_strategy(request.chunk_strategy)
    index = build_local_vector_index(
        index_name,
        source_text,
        request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        chunk_strategy=chunk_strategy,
        source_name="inline_text",
        source_documents=request.source_documents,
    )

    return ChatVectorIndexBuildResponse(
        index_name=index.index_name,
        source_length=len(index.source_text),
        source_count=index.source_count,
        chunk_count=len(index.items),
        chunk_size=index.chunk_size,
        chunk_overlap=index.chunk_overlap,
        chunk_strategy=index.chunk_strategy,
        embedding_model=settings.embedding_model,
    )


def build_faiss_search_reply(request: ChatFaissSearchRequest) -> ChatFaissSearchResponse:
    # FAISS 版查询入口。
    ensure_api_key()

    index_name = request.index_name.strip()
    question = request.question.strip()
    source_name_filter = normalize_source_name_filter(request.source_name_filter)
    index = faiss_index_store.get(index_name)
    if index is None:
        raise ValueError(
            f"FAISS 索引 {index_name} 不存在，请先调用 /chat/faiss-index/build 构建索引"
        )

    scored_chunks = search_local_faiss_index(
        question,
        index,
        request.top_k,
        score_threshold=request.score_threshold,
        source_name_filter=source_name_filter,
    )
    selected_chunks = [item.text for _, item in scored_chunks]
    scores = [round(score, 4) for score, _ in scored_chunks]
    source_names = [item.source_name for _, item in scored_chunks]

    if not selected_chunks:
        return ChatFaissSearchResponse(
            answer="当前筛选条件下，没有检索到足够相关的知识片段。",
            selected_chunks=[],
            scores=[],
            source_names=[],
            index_name=index.index_name,
            chunk_count=len(index.items),
            source_count=index.source_count,
            score_threshold=request.score_threshold,
            source_name_filter=source_name_filter,
            chunk_strategy=index.chunk_strategy,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
        )

    joined_chunks = "\n\n".join(selected_chunks)
    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名知识库问答助手。"
                    "请优先依据我提供的 FAISS 检索片段回答。"
                    "如果检索片段中没有明确答案，就明确说明“检索片段中没有直接提到”。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"FAISS 检索到的相关片段如下：\n{joined_chunks}\n\n"
                    f"用户问题如下：\n{question}"
                ),
            },
        ],
    )

    answer_text = completion.choices[0].message.content or ""
    return ChatFaissSearchResponse(
        answer=answer_text,
        selected_chunks=selected_chunks,
        scores=scores,
        source_names=source_names,
        index_name=index.index_name,
        chunk_count=len(index.items),
        source_count=index.source_count,
        score_threshold=request.score_threshold,
        source_name_filter=source_name_filter,
        chunk_strategy=index.chunk_strategy,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


def build_vector_search_reply(request: ChatVectorSearchRequest) -> ChatVectorSearchResponse:
    # 查询阶段入口：问题只和已有索引比较，不再重算所有 chunk 的 embedding。
    ensure_api_key()

    index_name = request.index_name.strip()
    question = request.question.strip()
    source_name_filter = normalize_source_name_filter(request.source_name_filter)
    index = vector_index_store.get(index_name)
    if index is None:
        raise ValueError(
            f"索引 {index_name} 不存在，请先调用 /chat/vector-index/build 构建内存索引，"
            "或者调用 /chat/vector-index/load 从本地文件加载索引"
        )

    scored_chunks = search_local_vector_index(
        question,
        index,
        request.top_k,
        score_threshold=request.score_threshold,
        source_name_filter=source_name_filter,
    )
    selected_chunks = [item.text for _, item in scored_chunks]
    scores = [round(score, 4) for score, _ in scored_chunks]

    if not selected_chunks:
        return ChatVectorSearchResponse(
            answer="当前筛选条件下，没有检索到足够相关的知识片段。",
            selected_chunks=[],
            scores=[],
            index_name=index.index_name,
            chunk_count=len(index.items),
            source_count=index.source_count,
            score_threshold=request.score_threshold,
            source_name_filter=source_name_filter,
            chunk_strategy=index.chunk_strategy,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
        )

    joined_chunks = "\n\n".join(selected_chunks)

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名知识库问答助手。"
                    "请优先依据我提供的本地向量检索片段回答。"
                    "如果检索片段中没有明确答案，就明确说明“检索片段中没有直接提到”。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"本地向量检索到的相关片段如下：\n{joined_chunks}\n\n"
                    f"用户问题如下：\n{question}"
                ),
            },
        ],
    )

    answer_text = completion.choices[0].message.content or ""
    return ChatVectorSearchResponse(
        answer=answer_text,
        selected_chunks=selected_chunks,
        scores=scores,
        index_name=index.index_name,
        chunk_count=len(index.items),
        source_count=index.source_count,
        score_threshold=request.score_threshold,
        source_name_filter=source_name_filter,
        chunk_strategy=index.chunk_strategy,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


def build_vector_search_with_citations_reply(
    request: ChatVectorSearchRequest,
) -> ChatVectorSearchWithCitationsResponse:
    # 在回答结果中同时返回命中片段的来源信息。
    ensure_api_key()

    index_name = request.index_name.strip()
    question = request.question.strip()
    source_name_filter = normalize_source_name_filter(request.source_name_filter)
    index = vector_index_store.get(index_name)
    if index is None:
        raise ValueError(
            f"索引 {index_name} 不存在，请先调用 /chat/upload-index 或 /chat/vector-index/build 建立索引"
        )

    scored_chunks = search_local_vector_index(
        question,
        index,
        request.top_k,
        score_threshold=request.score_threshold,
        source_name_filter=source_name_filter,
    )
    selected_chunks = [item.text for _, item in scored_chunks]

    if not selected_chunks:
        return ChatVectorSearchWithCitationsResponse(
            answer="当前筛选条件下，没有检索到足够相关的知识片段。",
            citations=[],
            index_name=index.index_name,
            chunk_count=len(index.items),
            source_count=index.source_count,
            score_threshold=request.score_threshold,
            source_name_filter=source_name_filter,
            chunk_strategy=index.chunk_strategy,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
        )

    joined_chunks = "\n\n".join(selected_chunks)

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名知识库问答助手。"
                    "请优先依据我提供的检索片段回答问题。"
                    "如果答案不能从片段中直接得到，就明确说明“检索片段中没有直接提到”。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"检索到的相关片段如下：\n{joined_chunks}\n\n"
                    f"用户问题如下：\n{question}"
                ),
            },
        ],
    )

    answer_text = completion.choices[0].message.content or ""
    citations = [
        ChatCitationItem(
            index_name=index.index_name,
            source_name=item.source_name,
            chunk_index=item.chunk_index,
            chunk_text=item.text,
            score=round(score, 4),
        )
        for score, item in scored_chunks
    ]

    return ChatVectorSearchWithCitationsResponse(
        answer=answer_text,
        citations=citations,
        index_name=index.index_name,
        chunk_count=len(index.items),
        source_count=index.source_count,
        score_threshold=request.score_threshold,
        source_name_filter=source_name_filter,
        chunk_strategy=index.chunk_strategy,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


def build_vector_index_save_reply(request: ChatVectorIndexSaveRequest) -> ChatVectorIndexSaveResponse:
    # 保存阶段：把内存中的索引写到本地文件。
    index_name = request.index_name.strip()
    index = vector_index_store.get(index_name)
    if index is None:
        raise ValueError(f"索引 {index_name} 不在内存中，请先构建索引再保存")

    file_path = save_local_vector_index(index)
    return ChatVectorIndexSaveResponse(
        index_name=index.index_name,
        file_path=str(file_path),
        chunk_count=len(index.items),
    )


def build_vector_index_load_reply(request: ChatVectorIndexLoadRequest) -> ChatVectorIndexLoadResponse:
    # 加载阶段：把本地文件里的索引恢复到内存。
    index, file_path = load_local_vector_index(request.index_name.strip())
    return ChatVectorIndexLoadResponse(
        index_name=index.index_name,
        file_path=str(file_path),
        chunk_count=len(index.items),
        source_count=index.source_count,
        chunk_size=index.chunk_size,
        chunk_overlap=index.chunk_overlap,
        chunk_strategy=index.chunk_strategy,
    )


def build_vector_index_list_reply() -> ChatVectorIndexListResponse:
    # 查看当前磁盘上有哪些已经保存的索引文件。
    indexes = list_saved_vector_indexes()
    return ChatVectorIndexListResponse(
        indexes=indexes,
        count=len(indexes),
    )
