# AI Chat API

This is a small learning project for AI application development with FastAPI.

## Features

- FastAPI application bootstrap
- Basic router organization
- Pydantic request and response schemas
- Config loading from `.env`
- Basic validation and exception handling
- `GET /health` endpoint for service checks
- `POST /hello` endpoint for request body practice
- `POST /chat/demo` endpoint for chat request/response practice
- `POST /chat/real` endpoint for real LLM chat practice
- `POST /chat/summary` endpoint for structured output practice
- `POST /chat/stream` endpoint for streaming output practice
- `POST /chat/session` endpoint for multi-turn chat practice
- `POST /chat/rag-demo` endpoint for minimal RAG practice
- `POST /chat/chunk-demo` endpoint for chunking practice
- `POST /chat/rag-search-demo` endpoint for retrieval practice
- `POST /chat/rag-embedding-demo` endpoint for embedding retrieval practice

## Project Structure

```text
app/
  main.py
  config.py
  routers/
    health.py
  schemas/
    health.py
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test Endpoints

Health check:

```text
http://127.0.0.1:8000/health
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Hello request example:

```json
{
  "name": "张三"
}
```

Chat demo request example:

```json
{
  "message": "Hello"
}
```

Real chat request example:

```json
{
  "message": "请用通俗的话解释一下什么是 FastAPI"
}
```

Summary request example:

```json
{
  "message": "请总结一下 FastAPI 是什么，它适合什么场景，并给出 3 个关键词"
}
```
