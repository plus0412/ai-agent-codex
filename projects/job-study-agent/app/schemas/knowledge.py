from pydantic import BaseModel, Field


class KnowledgeIndexInfo(BaseModel):
    # 已保存知识库的名称。
    index_name: str
    # 对应本地文件路径。
    file_path: str
    # 切分后的 chunk 数量。
    chunk_count: int
    # 原始文档数量。
    source_count: int
    # 构建时的 chunk_size。
    chunk_size: int
    # 构建时的 chunk_overlap。
    chunk_overlap: int
    # 构建时的切分策略。
    chunk_strategy: str


class KnowledgeIndexListResponse(BaseModel):
    # 所有已保存知识库。
    indexes: list[KnowledgeIndexInfo]
    # 已保存知识库数量。
    count: int


class KnowledgeUploadResponse(BaseModel):
    # 新建知识库名称。
    index_name: str
    # 上传文件名称。
    file_name: str
    # 文件后缀。
    file_extension: str
    # 原始文本长度。
    source_length: int
    # 原始文档数量。
    source_count: int
    # chunk 数量。
    chunk_count: int
    # chunk_size。
    chunk_size: int
    # chunk_overlap。
    chunk_overlap: int
    # 切分策略。
    chunk_strategy: str
    # 当前 embedding 模型名称。
    embedding_model: str


class KnowledgeSearchRequest(BaseModel):
    # 知识库名称。
    index_name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    # 用户问题。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 最多返回多少个 chunk。
    top_k: int = Field(3, ge=1, le=10, description="最多返回多少个 chunk")
    # 最低相似度阈值。
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="最低相似度阈值")
    # 可选：只筛选某个来源文档。
    source_name_filter: str = Field("", max_length=200, description="可选来源文档过滤")


class KnowledgeSearchResponse(BaseModel):
    # 检索返回的片段文本。
    selected_chunks: list[str]
    # 每个片段对应的分数。
    scores: list[float]
    # 每个片段对应的来源名称。
    source_names: list[str]
    # 知识库名称。
    index_name: str
    # chunk 总数。
    chunk_count: int
    # 原始文档数量。
    source_count: int
    # 实际使用的阈值。
    score_threshold: float
    # 实际使用的来源过滤条件。
    source_name_filter: str
    # 切分策略。
    chunk_strategy: str
    # embedding 模型名称。
    embedding_model: str
