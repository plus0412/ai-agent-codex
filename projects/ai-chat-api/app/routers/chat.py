from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.chat import (
    ChunkDemoRequest,
    ChunkDemoResponse,
    ChatDemoRequest,
    ChatDemoResponse,
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
    ChatRealRequest,
    ChatRealResponse,
    ChatSessionRequest,
    ChatSessionResponse,
    ChatSummaryRequest,
    ChatSummaryResponse,
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
)
from app.services.chat_service import (
    build_chunk_demo_reply,
    build_demo_reply,
    build_faiss_index_reply,
    build_faiss_search_reply,
    build_rag_embedding_reply,
    build_rag_reply,
    build_rag_search_reply,
    build_real_reply,
    build_session_reply,
    build_stream_reply,
    build_summary_reply,
    build_upload_index_reply,
    build_vector_index_reply,
    build_vector_index_list_reply,
    build_vector_index_load_reply,
    build_vector_index_save_reply,
    build_vector_search_reply,
    build_vector_search_with_citations_reply,
)

# 这一组路由专门放聊天相关的演示接口。
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/demo", response_model=ChatDemoResponse)
def chat_demo(request: ChatDemoRequest) -> ChatDemoResponse:
    # 路由函数负责接收请求，具体业务处理交给 service 层。
    reply = build_demo_reply(request)
    return ChatDemoResponse(reply=reply)


@router.post("/real", response_model=ChatRealResponse)
def chat_real(request: ChatRealRequest) -> ChatRealResponse:
    # 真实聊天接口：把用户消息交给大模型，再把模型回复返回给前端。
    reply_text, model_name = build_real_reply(request)
    return ChatRealResponse(reply=reply_text, model=model_name)


@router.post("/summary", response_model=ChatSummaryResponse)
def chat_summary(request: ChatSummaryRequest) -> ChatSummaryResponse:
    # 结构化输出接口：让模型返回标题、总结和关键词。
    return build_summary_reply(request)


@router.post("/stream")
def chat_stream(request: ChatRealRequest) -> StreamingResponse:
    # 流式接口：模型生成一段，后端就往前端推送一段。
    stream = build_stream_reply(request.message)
    return StreamingResponse(stream, media_type="text/plain; charset=utf-8")


@router.post("/session", response_model=ChatSessionResponse)
def chat_session(request: ChatSessionRequest) -> ChatSessionResponse:
    # 多轮对话接口：同一个 session_id 会自动带上历史消息。
    return build_session_reply(request)


@router.post("/rag-demo", response_model=ChatRagResponse)
def chat_rag_demo(request: ChatRagRequest) -> ChatRagResponse:
    # 最小版 RAG：把整份知识直接塞给模型。
    return build_rag_reply(request)


@router.post("/chunk-demo", response_model=ChunkDemoResponse)
def chat_chunk_demo(request: ChunkDemoRequest) -> ChunkDemoResponse:
    # 文本切分演示接口：观察长文本如何被切成多个 chunk。
    return build_chunk_demo_reply(request)


@router.post("/rag-search-demo", response_model=ChatRagSearchResponse)
def chat_rag_search_demo(request: ChatRagSearchRequest) -> ChatRagSearchResponse:
    # 关键词检索版 RAG：先切分，再选相关 chunk，再交给模型回答。
    return build_rag_search_reply(request)


@router.post("/rag-embedding-demo", response_model=ChatRagEmbeddingResponse)
def chat_rag_embedding_demo(request: ChatRagEmbeddingRequest) -> ChatRagEmbeddingResponse:
    # embedding 检索版 RAG：每次请求都现算 chunk 的 embedding。
    return build_rag_embedding_reply(request)


@router.post("/vector-index/build", response_model=ChatVectorIndexBuildResponse)
def chat_vector_index_build(
    request: ChatVectorIndexBuildRequest,
) -> ChatVectorIndexBuildResponse:
    # 先把文本切块、算 embedding，并存到本地内存索引里。
    return build_vector_index_reply(request)


@router.post("/faiss-index/build", response_model=ChatFaissIndexBuildResponse)
def chat_faiss_index_build(
    request: ChatFaissIndexBuildRequest,
) -> ChatFaissIndexBuildResponse:
    # 先完成常规 chunk 和 embedding 预处理，再用 FAISS 构建向量检索结构。
    return build_faiss_index_reply(request)


@router.post("/vector-search", response_model=ChatVectorSearchResponse)
def chat_vector_search(request: ChatVectorSearchRequest) -> ChatVectorSearchResponse:
    # 查询阶段只计算问题向量，然后去已经建好的本地向量索引里查找。
    return build_vector_search_reply(request)


@router.post("/faiss-search", response_model=ChatFaissSearchResponse)
def chat_faiss_search(request: ChatFaissSearchRequest) -> ChatFaissSearchResponse:
    # FAISS 查询阶段：让 FAISS 先做近邻检索，再把结果交给模型生成答案。
    return build_faiss_search_reply(request)


@router.post("/vector-search-with-citations", response_model=ChatVectorSearchWithCitationsResponse)
def chat_vector_search_with_citations(
    request: ChatVectorSearchRequest,
) -> ChatVectorSearchWithCitationsResponse:
    # 在检索问答结果中同时返回引用来源片段。
    return build_vector_search_with_citations_reply(request)


@router.post("/vector-index/save", response_model=ChatVectorIndexSaveResponse)
def chat_vector_index_save(
    request: ChatVectorIndexSaveRequest,
) -> ChatVectorIndexSaveResponse:
    # 把内存中的索引保存到本地 JSON 文件，避免服务重启后数据消失。
    return build_vector_index_save_reply(request)


@router.post("/vector-index/load", response_model=ChatVectorIndexLoadResponse)
def chat_vector_index_load(
    request: ChatVectorIndexLoadRequest,
) -> ChatVectorIndexLoadResponse:
    # 从本地 JSON 文件重新加载索引到内存中。
    return build_vector_index_load_reply(request)


@router.get("/vector-index/list", response_model=ChatVectorIndexListResponse)
def chat_vector_index_list() -> ChatVectorIndexListResponse:
    # 查看当前已经保存到本地文件的所有索引。
    return build_vector_index_list_reply()


@router.post("/upload-index", response_model=ChatUploadIndexResponse)
async def chat_upload_index(
    index_name: str = Form(..., description="新索引名称"),
    chunk_size: int = Form(100, description="每个片段的最大字符数"),
    chunk_overlap: int = Form(0, description="相邻片段的重叠字符数"),
    chunk_strategy: str = Form("fixed", description="切分策略：fixed 或 paragraph"),
    file: UploadFile = File(..., description="上传的知识文件"),
) -> ChatUploadIndexResponse:
    # 上传文档后，直接解析文本并建立向量索引。
    return await build_upload_index_reply(
        index_name,
        chunk_size,
        chunk_overlap,
        chunk_strategy,
        file,
    )
