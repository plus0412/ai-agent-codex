# JobStudyAgent

一个面向求职与学习场景的 AI Agent 项目，支持多轮对话、资料上传建库、知识库检索、岗位分析和学习计划生成。

## 项目定位

这个项目不是一个单纯的聊天接口，而是一个基于 `FastAPI + LangChain + LangGraph + RAG` 的任务型 Agent。

它主要解决两类真实问题：

1. 用户上传岗位 JD、学习笔记、项目总结后，希望系统基于资料回答问题
2. 用户希望系统进一步完成任务，例如岗位分析、学习建议和阶段性学习计划

## 当前已完成能力

- FastAPI 项目基础结构
- `.env` 配置读取
- OpenAI 兼容接口大模型调用
- MySQL 持久化会话记忆
- LangGraph 主流程编排
- 知识库上传建库
- 知识库列表查询
- 知识库片段检索
- 基于知识片段回答
- 岗位分析节点
- 学习计划节点
- 基础 debug 返回结构

## 当前技术栈

- Python
- FastAPI
- Pydantic
- OpenAI SDK
- LangChain
- LangGraph
- 本地 JSON 持久化
- 向量检索
- Embedding

## 项目结构

```text
job-study-agent
├─ app
│  ├─ config.py
│  ├─ exceptions.py
│  ├─ main.py
│  ├─ graph
│  │  └─ state.py
│  ├─ routers
│  │  ├─ agent.py
│  │  ├─ health.py
│  │  └─ knowledge.py
│  ├─ schemas
│  │  ├─ agent.py
│  │  ├─ health.py
│  │  └─ knowledge.py
│  └─ services
│     ├─ agent_service.py
│     └─ knowledge_service.py
├─ docs
├─ storage
└─ README.md
```

## 核心业务能力

### 1. 普通问答

用户发送普通问题，系统直接调用模型回答。

### 2. 知识库问答

用户指定知识库后提问，系统先检索相关片段，再基于片段回答。

### 3. 岗位分析

用户指定岗位资料知识库后提问，系统先检索资料，再进入岗位分析节点，输出岗位重点、优先补强方向和学习建议。

### 4. 学习计划

用户输入目标和背景后，系统进入学习计划节点，生成结构化学习安排。

## LangGraph 主流程

当前主流程包含这些核心节点：

- `route_intent`
- `check_context`
- `retrieve_knowledge`
- `answer_with_knowledge`
- `job_analysis`
- `study_plan`
- `direct_answer`
- `fallback`
- `save_session`

## 当前接口

### 健康检查

- `GET /health`

### Agent 主入口

- `POST /agent/chat`

### 知识库相关

- `POST /agent/upload-index`
- `GET /agent/indexes`
- `POST /agent/search`

## 启动方式

先安装依赖：

```bash
pip install -r requirements.txt
```

再检查 `.env` 中是否已经填写：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`

启动服务：

```bash
uvicorn app.main:app --reload
```

Swagger 地址：

```text
http://127.0.0.1:8000/docs
```

## 典型使用流程

### 1. 上传资料建库

调用：

- `POST /agent/upload-index`

把岗位 JD、学习笔记、项目总结等资料上传成知识库。

### 2. 查看知识库

调用：

- `GET /agent/indexes`

确认当前已保存知识库名称。

### 3. 单独测试检索

调用：

- `POST /agent/search`

验证某个问题能否在知识库中检索到相关片段。

### 4. 走 Agent 主入口

调用：

- `POST /agent/chat`

系统会根据意图自动进入：

- 普通聊天
- 知识库问答
- 岗位分析
- 学习计划

## 调试能力

当 `/agent/chat` 请求里传入：

```json
{
  "debug": true
}
```

当前会返回：

- 当前意图
- 图执行步骤
- 当前阶段
- 会话消息数量
- 检索到的知识片段数量
- 检索命中的来源名称

这部分既方便学习，也方便面试时展示系统内部执行过程。

## 当前版本边界

当前版本为了突出 Agent 主线，当前已接入：

- MySQL 会话持久化

当前版本暂时还没有接入：

- Redis
- 用户登录
- 多用户权限
- 前端页面
- 多 Agent 协作

这些都可以作为后续扩展方向。
