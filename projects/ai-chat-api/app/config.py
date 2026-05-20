from dotenv import load_dotenv
from pydantic import BaseModel
import os

# 先加载 .env 文件中的环境变量。
load_dotenv()


class Settings(BaseModel):
    # 项目名称。
    app_name: str = "AI Chat API"
    # 项目版本。
    app_version: str = "0.1.0"
    # 当前运行环境，例如 dev / test / prod。
    app_env: str = os.getenv("APP_ENV", "dev")
    # 聊天演示接口的回复前缀。
    chat_reply_prefix: str = os.getenv("CHAT_REPLY_PREFIX", "你刚才说的是：")
    # 百炼 API Key。
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    # 百炼 OpenAI 兼容接口地址。
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 当前使用的大模型名称。
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    # 当前使用的 embedding 模型名称。
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")


# 创建一个全局配置对象，其他文件可以直接导入使用。
settings = Settings()
