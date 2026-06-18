import json
import re
from dataclasses import dataclass
from io import BytesIO
from math import sqrt
from pathlib import Path

from fastapi import UploadFile
from openai import OpenAI
from pypdf import PdfReader

from app.config import settings
from app.schemas.knowledge import (
    KnowledgeIndexInfo,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeUploadResponse,
)


@dataclass
class VectorChunkItem:
    # 单个 chunk 的文本、向量和来源信息。
    text: str
    embedding: list[float]
    source_name: str
    chunk_index: int


@dataclass
class LocalVectorIndex:
    # 本地知识库索引的完整结构。
    index_name: str
    source_text: str
    source_count: int
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    items: list[VectorChunkItem]


# 内存缓存，避免每次都重新从磁盘加载。
vector_index_store: dict[str, LocalVectorIndex] = {}


def get_project_root() -> Path:
    # 返回项目根目录。
    return Path(__file__).resolve().parents[2]


def get_vector_index_storage_dir() -> Path:
    # 本地知识库 JSON 文件的存储目录。
    return get_project_root() / "storage" / "vector_indexes"


def ensure_vector_index_storage_dir() -> Path:
    # 确保存储目录存在。
    storage_dir = get_vector_index_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def normalize_chunk_strategy(chunk_strategy: str) -> str:
    # 统一切分策略。
    normalized = chunk_strategy.strip().lower()
    if normalized not in {"fixed", "paragraph"}:
        raise ValueError("chunk_strategy 只支持 fixed 或 paragraph")
    return normalized


def split_text_fixed(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    # 按固定长度切分文本。
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
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned_text):
            break
    return chunks


def split_text_by_paragraphs(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    # 优先按段落切分。
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
            chunks.extend(split_text_fixed(paragraph, chunk_size, chunk_overlap))
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


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int = 0, chunk_strategy: str = "fixed") -> list[str]:
    # 统一切分入口。
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)
    if normalized_strategy == "paragraph":
        return split_text_by_paragraphs(text, chunk_size, chunk_overlap)
    return split_text_fixed(text, chunk_size, chunk_overlap)


def get_embedding_client() -> OpenAI:
    # 复用百炼兼容 OpenAI 的 embedding 客户端。
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    # 批量获取 embedding。
    if not texts:
        return []

    client = get_embedding_client()
    batch_size = 10
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        response = client.embeddings.create(model=settings.embedding_model, input=batch_texts)
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    # 计算余弦相似度。
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sqrt(sum(a * a for a in vector_a))
    norm_b = sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def normalize_documents(documents: list[tuple[str, str]]) -> list[tuple[str, str]]:
    # 清理文档来源名和文本。
    normalized: list[tuple[str, str]] = []
    for source_name, text in documents:
        cleaned_source_name = source_name.strip()
        cleaned_text = text.strip()
        if not cleaned_source_name:
            raise ValueError("source_name 不能为空")
        if not cleaned_text:
            raise ValueError(f"文档 {cleaned_source_name} 的文本不能为空")
        normalized.append((cleaned_source_name, cleaned_text))
    return normalized


def parse_txt_or_md_content(file_name: str, file_bytes: bytes) -> str:
    # txt / md 按 utf-8 读取。
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"文件 {file_name} 不是有效的 UTF-8 文本") from exc


def parse_pdf_content(file_name: str, file_bytes: bytes) -> str:
    # 读取 PDF 文本。
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"文件 {file_name} 不是有效的 PDF") from exc

    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    content = "\n".join(page_texts).strip()
    if not content:
        raise ValueError(f"文件 {file_name} 没有可提取的文本内容")
    return content


def extract_text_from_upload(file_name: str, file_bytes: bytes) -> tuple[str, str]:
    # 根据后缀提取文本。
    extension = Path(file_name).suffix.lower()
    if extension in {".txt", ".md"}:
        return parse_txt_or_md_content(file_name, file_bytes), extension
    if extension == ".pdf":
        return parse_pdf_content(file_name, file_bytes), extension
    raise ValueError("当前只支持上传 .txt、.md、.pdf 文件")


def build_local_vector_index(
    index_name: str,
    source_text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    chunk_strategy: str = "fixed",
    source_name: str = "inline_text",
) -> LocalVectorIndex:
    # 构建本地向量索引。
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)
    cleaned_source_text = source_text.strip()
    cleaned_source_name = source_name.strip() or "inline_text"

    if not cleaned_source_text:
        raise ValueError("源文本不能为空")

    chunks = split_text_into_chunks(cleaned_source_text, chunk_size, chunk_overlap, normalized_strategy)
    if not chunks:
        raise ValueError("没有可用于构建索引的 chunk")

    chunk_embeddings = get_embeddings(chunks)
    items = [
        VectorChunkItem(
            text=chunk_text,
            embedding=chunk_embedding,
            source_name=cleaned_source_name,
            chunk_index=chunk_index,
        )
        for chunk_index, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings), start=1)
    ]

    index = LocalVectorIndex(
        index_name=index_name,
        source_text=cleaned_source_text,
        source_count=1,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=normalized_strategy,
        items=items,
    )
    vector_index_store[index_name] = index
    save_local_vector_index(index)
    return index


def serialize_local_vector_index(index: LocalVectorIndex) -> dict:
    # 转成 JSON 可保存结构。
    return {
        "index_name": index.index_name,
        "source_text": index.source_text,
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
    # 从 JSON 还原成内存对象。
    items = [
        VectorChunkItem(
            text=item["text"],
            embedding=item["embedding"],
            source_name=item.get("source_name", data.get("index_name", "unknown")),
            chunk_index=item.get("chunk_index", 0),
        )
        for item in data["items"]
    ]
    return LocalVectorIndex(
        index_name=data["index_name"],
        source_text=data["source_text"],
        source_count=data.get("source_count", 1),
        chunk_size=data["chunk_size"],
        chunk_overlap=data.get("chunk_overlap", 0),
        chunk_strategy=data.get("chunk_strategy", "fixed"),
        items=items,
    )


def get_vector_index_file_path(index_name: str) -> Path:
    # 获取单个索引文件的完整路径。
    safe_index_name = re.sub(r'[<>:"/\\\\|?*]', "_", index_name.strip())
    return get_vector_index_storage_dir() / f"{safe_index_name}.json"


def save_local_vector_index(index: LocalVectorIndex) -> Path:
    # 保存到本地 JSON。
    storage_dir = ensure_vector_index_storage_dir()
    file_path = storage_dir / f"{re.sub(r'[<>:\"/\\\\|?*]', '_', index.index_name.strip())}.json"
    file_path.write_text(json.dumps(serialize_local_vector_index(index), ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def load_local_vector_index(index_name: str) -> tuple[LocalVectorIndex, Path]:
    # 从磁盘加载索引。
    file_path = get_vector_index_file_path(index_name)
    if not file_path.exists():
        raise ValueError(f"索引文件不存在，请先创建：{file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    index = deserialize_local_vector_index(data)
    vector_index_store[index.index_name] = index
    return index, file_path


def get_vector_index(index_name: str) -> tuple[LocalVectorIndex, Path]:
    # 先读内存，读不到再从磁盘加载。
    cleaned_index_name = index_name.strip()
    if not cleaned_index_name:
        raise ValueError("index_name 不能为空")
    index = vector_index_store.get(cleaned_index_name)
    if index is not None:
        return index, get_vector_index_file_path(cleaned_index_name)
    return load_local_vector_index(cleaned_index_name)


def list_saved_vector_indexes() -> list[KnowledgeIndexInfo]:
    # 列出所有已保存知识库。
    storage_dir = ensure_vector_index_storage_dir()
    indexes: list[KnowledgeIndexInfo] = []
    for file_path in storage_dir.glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        indexes.append(
            KnowledgeIndexInfo(
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


def normalize_source_name_filter(source_name_filter: str) -> str:
    return source_name_filter.strip()


def search_local_vector_index(
    question: str,
    index: LocalVectorIndex,
    top_k: int,
    score_threshold: float = 0.0,
    source_name_filter: str = "",
) -> list[tuple[float, VectorChunkItem]]:
    # 在本地索引里检索相关 chunk。
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question 不能为空")

    normalized_filter = normalize_source_name_filter(source_name_filter)
    question_embedding = get_embeddings([cleaned_question])[0]

    scored_chunks: list[tuple[float, VectorChunkItem]] = []
    for item in index.items:
        if normalized_filter and item.source_name != normalized_filter:
            continue
        score = cosine_similarity(question_embedding, item.embedding)
        if score >= score_threshold:
            scored_chunks.append((score, item))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return scored_chunks[:top_k]


def build_upload_index_reply(
    index_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
    file: UploadFile,
) -> KnowledgeUploadResponse:
    # 上传文件后直接构建知识库。
    if not file.filename:
        raise ValueError("上传文件必须带有文件名")

    file_bytes = file.file.read()
    if not file_bytes:
        raise ValueError("上传文件不能为空")

    source_text, extension = extract_text_from_upload(file.filename, file_bytes)
    index = build_local_vector_index(
        index_name=index_name.strip(),
        source_text=source_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        source_name=file.filename,
    )

    return KnowledgeUploadResponse(
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


def build_knowledge_search_reply(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    # 检索知识库片段。
    index, _ = get_vector_index(request.index_name)
    scored_chunks = search_local_vector_index(
        request.question,
        index,
        request.top_k,
        score_threshold=request.score_threshold,
        source_name_filter=request.source_name_filter,
    )
    selected_chunks = [item.text for _, item in scored_chunks]
    scores = [round(score, 4) for score, _ in scored_chunks]
    source_names = [item.source_name for _, item in scored_chunks]
    return KnowledgeSearchResponse(
        selected_chunks=selected_chunks,
        scores=scores,
        source_names=source_names,
        index_name=index.index_name,
        chunk_count=len(index.items),
        source_count=index.source_count,
        score_threshold=request.score_threshold,
        source_name_filter=request.source_name_filter.strip(),
        chunk_strategy=index.chunk_strategy,
        embedding_model=settings.embedding_model,
    )


def build_knowledge_index_list_reply() -> list[KnowledgeIndexInfo]:
    # 返回当前所有已保存知识库。
    return list_saved_vector_indexes()
