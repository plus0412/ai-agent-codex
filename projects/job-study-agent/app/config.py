import os

from dotenv import load_dotenv
from pydantic import BaseModel

# 先加载 .env 文件中的环境变量。
load_dotenv()


class Settings(BaseModel):
    # 项目名称。
    app_name: str = "JobStudyAgent"
    # 项目版本。
    app_version: str = "0.2.0"
    # 当前运行环境，例如 dev / test / prod。
    app_env: str = os.getenv("APP_ENV", "dev")

    # 大模型 API 配置。
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    # 会话历史最多保留多少条消息，避免上下文无限增长。
    session_max_messages: int = int(os.getenv("SESSION_MAX_MESSAGES", "20"))

    # MySQL 连接配置。
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "job_study_agent")
    mysql_echo: bool = os.getenv("MYSQL_ECHO", "false").lower() == "true"

    # Redis 连接配置。
    redis_host: str = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_session_ttl_seconds: int = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "3600"))

    @property
    def mysql_url(self) -> str:
        # SQLAlchemy 需要的数据库连接地址。
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            "?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        # redis-py 需要的连接地址。
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# 创建一个全局配置对象，其他文件可以直接导入使用。
settings = Settings()
