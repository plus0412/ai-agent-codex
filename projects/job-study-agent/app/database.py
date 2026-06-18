from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import settings

# 创建数据库引擎，后面所有数据库操作都会复用它。
engine = create_engine(
    settings.mysql_url,
    echo=settings.mysql_echo,
    pool_pre_ping=True,
)

# SessionLocal 类似于“数据库会话工厂”，每次用它创建一个新的会话对象。
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)

# Base 是所有 ORM 模型类的父类。
Base = declarative_base()


def init_db() -> None:
    # 先导入模型，让 SQLAlchemy 知道有哪些表需要创建。
    from app.models import ChatMessage  # noqa: F401

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Iterator[Session]:
    # 用上下文管理器统一处理提交、回滚和关闭。
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
