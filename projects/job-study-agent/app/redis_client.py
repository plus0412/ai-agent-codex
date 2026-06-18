import json

import redis

from app.config import settings
from app.schemas.agent import AgentMessageItem

# 创建全局 Redis 客户端，后面统一复用。
redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
)


def build_session_cache_key(session_id: str) -> str:
    # 为每个会话生成固定的 Redis key。
    return f"job_study_agent:session:{session_id}"


def load_session_messages_from_cache(session_id: str) -> list[AgentMessageItem] | None:
    # 从 Redis 读取当前会话历史；没有命中时返回 None。
    cache_key = build_session_cache_key(session_id)
    cached_value = redis_client.get(cache_key)
    if not cached_value:
        return None

    payload = json.loads(cached_value)
    return [AgentMessageItem(**item) for item in payload]


def save_session_messages_to_cache(session_id: str, messages: list[AgentMessageItem]) -> None:
    # 把当前会话历史整体写入 Redis，并设置过期时间。
    cache_key = build_session_cache_key(session_id)
    payload = [message.model_dump() for message in messages]
    redis_client.set(
        cache_key,
        json.dumps(payload, ensure_ascii=False),
        ex=settings.redis_session_ttl_seconds,
    )


def delete_session_messages_from_cache(session_id: str) -> None:
    # 删除某个会话在 Redis 中的缓存。
    cache_key = build_session_cache_key(session_id)
    redis_client.delete(cache_key)
