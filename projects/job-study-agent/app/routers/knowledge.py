from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.knowledge import KnowledgeIndexListResponse, KnowledgeSearchRequest, KnowledgeSearchResponse, KnowledgeUploadResponse
from app.services.knowledge_service import build_knowledge_index_list_reply, build_knowledge_search_reply, build_upload_index_reply

# 这一组路由负责知识库上传、查看和检索。
router = APIRouter(prefix="/agent", tags=["knowledge"])


@router.post("/upload-index", response_model=KnowledgeUploadResponse)
def upload_index(
    index_name: str = Form(...),
    chunk_size: int = Form(100),
    chunk_overlap: int = Form(0),
    chunk_strategy: str = Form("fixed"),
    file: UploadFile = File(...),
) -> KnowledgeUploadResponse:
    # 上传文件并创建本地知识库索引。
    return build_upload_index_reply(
        index_name=index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        file=file,
    )


@router.get("/indexes", response_model=KnowledgeIndexListResponse)
def list_indexes() -> KnowledgeIndexListResponse:
    # 查看当前已经保存了哪些知识库。
    indexes = build_knowledge_index_list_reply()
    return KnowledgeIndexListResponse(indexes=indexes, count=len(indexes))


@router.post("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    # 单独测试知识库检索效果。
    return build_knowledge_search_reply(request)
