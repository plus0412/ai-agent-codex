from pydantic import BaseModel, Field


class ChatDemoRequest(BaseModel):
    # 用户发送过来的消息内容。
    message: str = Field(..., min_length=1, max_length=200, description="用户输入的消息")


class ChatDemoResponse(BaseModel):
    # 后端返回给用户的回复内容。
    reply: str


class ChatRealRequest(BaseModel):
    # 用户发送给真实大模型的消息内容。
    message: str = Field(..., min_length=1, max_length=2000, description="用户输入的消息")


class ChatRealResponse(BaseModel):
    # 真实大模型返回的回答内容。
    reply: str
    # 当前调用的模型名称。
    model: str


class ChatSummaryRequest(BaseModel):
    # 用户希望被总结或解释的问题。
    message: str = Field(..., min_length=1, max_length=2000, description="需要总结的用户输入")


class ChatSummaryResponse(BaseModel):
    # 模型提炼出的标题。
    title: str
    # 模型输出的简要总结。
    summary: str
    # 模型提取出的关键词列表。
    keywords: list[str]
    # 当前调用的模型名称。
    model: str


class ChatTurn(BaseModel):
    # 消息角色，例如 user / assistant。
    role: str
    # 消息文本内容。
    content: str


class ChatSessionRequest(BaseModel):
    # 会话 id，用于区分不同用户对话。
    session_id: str = Field(..., min_length=1, max_length=100, description="会话唯一标识")
    # 用户当前输入的消息。
    message: str = Field(..., min_length=1, max_length=2000, description="当前用户输入")


class ChatSessionResponse(BaseModel):
    # 当前轮模型回复。
    reply: str
    # 当前会话累计的消息条数。
    history_count: int
    # 当前调用的模型名称。
    model: str


class ChatRagRequest(BaseModel):
    # 用户针对知识内容提出的问题。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class ChatRagResponse(BaseModel):
    # 模型基于知识内容给出的回答。
    answer: str
    # 本次注入给模型的知识内容。
    knowledge: str
    # 当前调用的模型名称。
    model: str


class ChunkDemoRequest(BaseModel):
    # 需要被切分的原始文本。
    text: str = Field(..., min_length=1, max_length=5000, description="待切分文本")
    # 每个片段的最大长度。
    chunk_size: int = Field(100, ge=20, le=1000, description="每个片段的最大字符数")
    # 相邻片段之间重叠的字符数。
    chunk_overlap: int = Field(0, ge=0, le=300, description="相邻片段的重叠字符数")
    # 切分策略。
    chunk_strategy: str = Field(
        "fixed",
        pattern="^(fixed|paragraph)$",
        description="切分策略：fixed 或 paragraph",
    )


class ChunkDemoResponse(BaseModel):
    # 原始文本总长度。
    original_length: int
    # 切分后的片段总数。
    chunk_count: int
    # 本次使用的切分策略。
    chunk_strategy: str
    # 切分后的所有片段。
    chunks: list[str]


class ChatRagSearchRequest(BaseModel):
    # 用户提问内容。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 每个 chunk 的最大长度。
    chunk_size: int = Field(100, ge=20, le=1000, description="每个片段的最大字符数")
    # 相邻片段之间重叠的字符数。
    chunk_overlap: int = Field(0, ge=0, le=300, description="相邻片段的重叠字符数")
    # 切分策略。
    chunk_strategy: str = Field(
        "fixed",
        pattern="^(fixed|paragraph)$",
        description="切分策略：fixed 或 paragraph",
    )
    # 最多返回多少个相关 chunk。
    top_k: int = Field(2, ge=1, le=5, description="返回的相关片段数量")


class ChatRagSearchResponse(BaseModel):
    # 模型基于检索结果生成的回答。
    answer: str
    # 被选中的相关 chunk。
    selected_chunks: list[str]
    # 本次检索使用的切分策略。
    chunk_strategy: str
    # 当前调用的模型名称。
    model: str


class ChatRagEmbeddingRequest(BaseModel):
    # 用户提问内容。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 每个 chunk 的最大长度。
    chunk_size: int = Field(100, ge=20, le=1000, description="每个片段的最大字符数")
    # 相邻片段之间重叠的字符数。
    chunk_overlap: int = Field(0, ge=0, le=300, description="相邻片段的重叠字符数")
    # 切分策略。
    chunk_strategy: str = Field(
        "fixed",
        pattern="^(fixed|paragraph)$",
        description="切分策略：fixed 或 paragraph",
    )
    # 最多返回多少个相关 chunk。
    top_k: int = Field(2, ge=1, le=5, description="返回的相关片段数量")


class ChatRagEmbeddingResponse(BaseModel):
    # 模型基于语义检索结果生成的回答。
    answer: str
    # 被选中的相关 chunk。
    selected_chunks: list[str]
    # 每个选中 chunk 的相似度分数。
    scores: list[float]
    # 本次检索使用的切分策略。
    chunk_strategy: str
    # 当前调用的文本生成模型名称。
    model: str
    # 当前调用的 embedding 模型名称。
    embedding_model: str


class KnowledgeDocumentInput(BaseModel):
    # 当前文档的来源名称，比如文件名、文章标题、模块名。
    source_name: str = Field(..., min_length=1, max_length=200, description="文档来源名称")
    # 当前文档的原始文本内容。
    text: str = Field(..., min_length=1, max_length=20000, description="文档文本内容")


class ChatVectorIndexBuildRequest(BaseModel):
    # 本地向量索引名称，用来区分不同索引。
    index_name: str = Field(..., min_length=1, max_length=100, description="本地索引名称")
    # 要切分的原始文本；不传时默认读取演示知识库文件。
    source_text: str = Field("", max_length=20000, description="可选：自定义知识文本")
    # 可选：一次传入多篇文档，构造成一个多文档知识库索引。
    source_documents: list[KnowledgeDocumentInput] = Field(default_factory=list, description="可选：多篇知识文档")
    # 每个 chunk 的最大字符数。
    chunk_size: int = Field(100, ge=20, le=1000, description="每个片段的最大字符数")
    # 相邻片段之间重叠的字符数。
    chunk_overlap: int = Field(0, ge=0, le=300, description="相邻片段的重叠字符数")
    # 切分策略。
    chunk_strategy: str = Field(
        "fixed",
        pattern="^(fixed|paragraph)$",
        description="切分策略：fixed 或 paragraph",
    )


class ChatVectorIndexBuildResponse(BaseModel):
    # 本次构建完成的索引名称。
    index_name: str
    # 原始文本总长度。
    source_length: int
    # 本次索引中一共包含多少篇源文档。
    source_count: int
    # 切分后的 chunk 数量。
    chunk_count: int
    # 当前索引使用的 chunk_size。
    chunk_size: int
    # 当前索引使用的 chunk_overlap。
    chunk_overlap: int
    # 当前索引使用的切分策略。
    chunk_strategy: str
    # 当前 embedding 模型名称。
    embedding_model: str


class ChatVectorSearchRequest(BaseModel):
    # 要查询的本地索引名称。
    index_name: str = Field(..., min_length=1, max_length=100, description="本地索引名称")
    # 用户问题。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 最多返回多少个相关 chunk。
    top_k: int = Field(2, ge=1, le=5, description="返回的相关片段数量")
    # 最低相似度阈值，低于这个分数的 chunk 会被过滤掉。
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="最低相似度阈值，范围 0 到 1")
    # 可选：只在指定来源文档里检索。
    source_name_filter: str = Field("", max_length=200, description="可选：指定来源文档名称")


class ChatVectorSearchResponse(BaseModel):
    # 模型基于检索片段生成的答案。
    answer: str
    # 实际命中的相关 chunk。
    selected_chunks: list[str]
    # 每个命中 chunk 的相似度分数。
    scores: list[float]
    # 本次查询命中的索引名称。
    index_name: str
    # 当前索引中的 chunk 总数。
    chunk_count: int
    # 当前索引中包含的源文档数量。
    source_count: int
    # 本次检索实际使用的相似度阈值。
    score_threshold: float
    # 本次检索实际使用的来源文档过滤条件。
    source_name_filter: str
    # 当前索引的切分策略。
    chunk_strategy: str
    # 文本生成模型名称。
    model: str
    # embedding 模型名称。
    embedding_model: str


class ChatFaissIndexBuildRequest(BaseModel):
    # FAISS 本地索引名称。
    index_name: str = Field(..., min_length=1, max_length=100, description="FAISS 本地索引名称")
    # 要切分的原始文本；不传时默认读取演示知识库文件。
    source_text: str = Field("", max_length=20000, description="可选：自定义知识文本")
    # 可选：一次传入多篇文档，构造成一个多文档知识库索引。
    source_documents: list[KnowledgeDocumentInput] = Field(default_factory=list, description="可选：多篇知识文档")
    # 每个 chunk 的最大字符数。
    chunk_size: int = Field(100, ge=20, le=1000, description="每个片段的最大字符数")
    # 相邻片段之间重叠的字符数。
    chunk_overlap: int = Field(0, ge=0, le=300, description="相邻片段的重叠字符数")
    # 切分策略。
    chunk_strategy: str = Field(
        "fixed",
        pattern="^(fixed|paragraph)$",
        description="切分策略：fixed 或 paragraph",
    )


class ChatFaissIndexBuildResponse(BaseModel):
    # 本次构建完成的索引名称。
    index_name: str
    # 原始文本总长度。
    source_length: int
    # 本次索引中一共包含多少篇源文档。
    source_count: int
    # 切分后的 chunk 数量。
    chunk_count: int
    # 当前索引使用的 chunk_size。
    chunk_size: int
    # 当前索引使用的 chunk_overlap。
    chunk_overlap: int
    # 当前索引使用的切分策略。
    chunk_strategy: str
    # 当前向量维度。
    vector_dimension: int
    # 当前 embedding 模型名称。
    embedding_model: str


class ChatFaissSearchRequest(BaseModel):
    # 要查询的 FAISS 本地索引名称。
    index_name: str = Field(..., min_length=1, max_length=100, description="FAISS 本地索引名称")
    # 用户问题。
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 最多返回多少个相关 chunk。
    top_k: int = Field(2, ge=1, le=5, description="返回的相关片段数量")
    # 最低相似度阈值，低于这个分数的 chunk 会被过滤掉。
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="最低相似度阈值，范围 0 到 1")
    # 可选：只在指定来源文档里检索。
    source_name_filter: str = Field("", max_length=200, description="可选：指定来源文档名称")


class ChatFaissSearchResponse(BaseModel):
    # 模型基于检索片段生成的答案。
    answer: str
    # 实际命中的相关 chunk。
    selected_chunks: list[str]
    # 每个命中 chunk 的相似度分数。
    scores: list[float]
    # 每个命中 chunk 对应的来源文档名。
    source_names: list[str]
    # 本次查询命中的索引名称。
    index_name: str
    # 当前索引中的 chunk 总数。
    chunk_count: int
    # 当前索引中包含的源文档数量。
    source_count: int
    # 本次检索实际使用的相似度阈值。
    score_threshold: float
    # 本次检索实际使用的来源文档过滤条件。
    source_name_filter: str
    # 当前索引的切分策略。
    chunk_strategy: str
    # 文本生成模型名称。
    model: str
    # embedding 模型名称。
    embedding_model: str


class ChatVectorIndexSaveRequest(BaseModel):
    # 要保存到本地文件的索引名称。
    index_name: str = Field(..., min_length=1, max_length=100, description="要保存的索引名称")


class ChatVectorIndexSaveResponse(BaseModel):
    # 已保存的索引名称。
    index_name: str
    # 本地保存文件的绝对路径。
    file_path: str
    # 当前索引中的 chunk 总数。
    chunk_count: int


class ChatVectorIndexLoadRequest(BaseModel):
    # 要从本地文件重新加载到内存的索引名称。
    index_name: str = Field(..., min_length=1, max_length=100, description="要加载的索引名称")


class ChatVectorIndexLoadResponse(BaseModel):
    # 已加载到内存的索引名称。
    index_name: str
    # 本地文件的绝对路径。
    file_path: str
    # 当前索引中的 chunk 总数。
    chunk_count: int
    # 当前索引中包含的源文档数量。
    source_count: int
    # 当前索引的 chunk_size。
    chunk_size: int
    # 当前索引的 chunk_overlap。
    chunk_overlap: int
    # 当前索引的切分策略。
    chunk_strategy: str


class SavedVectorIndexInfo(BaseModel):
    # 已保存索引的名称。
    index_name: str
    # 对应本地文件的绝对路径。
    file_path: str
    # 索引中的 chunk 总数。
    chunk_count: int
    # 索引中包含的源文档数量。
    source_count: int
    # 构建该索引时使用的 chunk_size。
    chunk_size: int
    # 构建该索引时使用的 chunk_overlap。
    chunk_overlap: int
    # 构建该索引时使用的切分策略。
    chunk_strategy: str


class ChatVectorIndexListResponse(BaseModel):
    # 当前扫描到的所有已保存索引。
    indexes: list[SavedVectorIndexInfo]
    # 已保存索引数量。
    count: int


class ChatUploadIndexResponse(BaseModel):
    # 新建索引的名称。
    index_name: str
    # 上传文件的原始名称。
    file_name: str
    # 文件类型后缀。
    file_extension: str
    # 解析出的文本总长度。
    source_length: int
    # 当前索引中包含的源文档数量。
    source_count: int
    # 切分后的 chunk 数量。
    chunk_count: int
    # 使用的 chunk_size。
    chunk_size: int
    # 使用的 chunk_overlap。
    chunk_overlap: int
    # 使用的切分策略。
    chunk_strategy: str
    # 当前 embedding 模型名称。
    embedding_model: str


class ChatCitationItem(BaseModel):
    # 当前引用片段所属索引名。
    index_name: str
    # 当前引用片段来源文件名。
    source_name: str
    # 当前引用片段在原文中的序号。
    chunk_index: int
    # 当前引用片段内容。
    chunk_text: str
    # 当前引用片段与问题的相似度分数。
    score: float


class ChatVectorSearchWithCitationsResponse(BaseModel):
    # 模型基于检索片段生成的答案。
    answer: str
    # 本次检索命中的引用列表。
    citations: list[ChatCitationItem]
    # 本次查询命中的索引名称。
    index_name: str
    # 当前索引中的 chunk 总数。
    chunk_count: int
    # 当前索引中包含的源文档数量。
    source_count: int
    # 本次检索实际使用的相似度阈值。
    score_threshold: float
    # 本次检索实际使用的来源文档过滤条件。
    source_name_filter: str
    # 当前索引的切分策略。
    chunk_strategy: str
    # 文本生成模型名称。
    model: str
    # embedding 模型名称。
    embedding_model: str
