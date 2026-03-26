"""
[INPUT]: 依赖 pathlib, shutil, uuid, time, tempfile
[OUTPUT]: 对外提供 TempSession, temp_session(), cleanup_old_sessions()
[POS]: 临时文件生命周期管理，确保视频处理过程中的资源正确释放
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import re
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Iterator


# =============================================================================
#  常量
# =============================================================================

BASE_DIR = Path(gettempdir()) / "vmarker"
DEFAULT_MAX_AGE_HOURS = 24
SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def is_valid_session_id(session_id: str) -> bool:
    """校验会话 ID，拒绝路径字符和非预期格式"""
    return bool(SESSION_ID_PATTERN.fullmatch(session_id))


# =============================================================================
#  会话管理类
# =============================================================================


class TempSession:
    """
    临时文件会话管理

    使用场景：
    1. 视频上传 → 保存到临时目录
    2. ASR 处理 → 保存 SRT 文件
    3. Bar 生成 → 保存中间视频
    4. 合成输出 → 保存最终视频
    5. 处理完成 / 异常 → 清理
    """

    def __init__(self, session_id: str | None = None, create: bool = True):
        """
        初始化会话

        Args:
            session_id: 会话 ID，不提供则自动生成
            create: 是否创建会话目录，默认创建
        """
        if session_id is None:
            self.session_id = uuid.uuid4().hex[:12]
        else:
            if not is_valid_session_id(session_id):
                raise ValueError(f"invalid session_id: {session_id!r}")
            self.session_id = session_id
        self.session_dir = BASE_DIR / self.session_id
        if create:
            self.session_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing(cls, session_id: str) -> "TempSession | None":
        """打开现有会话，不隐式创建目录"""
        if not is_valid_session_id(session_id):
            return None
        session_dir = BASE_DIR / session_id
        if not session_dir.is_dir():
            return None
        return cls(session_id=session_id, create=False)

    def save_upload(self, filename: str, content: bytes) -> Path:
        """
        保存上传文件

        Args:
            filename: 文件名
            content: 文件内容

        Returns:
            保存后的文件路径
        """
        path = self.session_dir / filename
        path.write_bytes(content)
        return path

    def save_text(self, filename: str, content: str, encoding: str = "utf-8") -> Path:
        """
        保存文本文件

        Args:
            filename: 文件名
            content: 文本内容
            encoding: 编码，默认 UTF-8

        Returns:
            保存后的文件路径
        """
        path = self.session_dir / filename
        path.write_text(content, encoding=encoding)
        return path

    def get_path(self, filename: str) -> Path:
        """
        获取会话内文件路径

        Args:
            filename: 文件名

        Returns:
            文件路径（不检查是否存在）
        """
        return self.session_dir / filename

    def exists(self, filename: str) -> bool:
        """检查文件是否存在"""
        return (self.session_dir / filename).exists()

    def list_files(self, pattern: str = "*") -> list[Path]:
        """列出匹配的文件"""
        return list(self.session_dir.glob(pattern))

    def read_bytes(self, filename: str) -> bytes:
        """读取文件内容"""
        return (self.session_dir / filename).read_bytes()

    def read_text(self, filename: str, encoding: str = "utf-8") -> str:
        """读取文本文件"""
        return (self.session_dir / filename).read_text(encoding=encoding)

    def cleanup(self) -> None:
        """清理会话目录"""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)

    @property
    def is_valid(self) -> bool:
        """检查会话是否有效（目录存在）"""
        return self.session_dir.exists()

    def __enter__(self) -> "TempSession":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


# =============================================================================
#  上下文管理器
# =============================================================================


@contextmanager
def temp_session() -> Iterator[TempSession]:
    """
    临时会话上下文管理器

    使用示例：
        with temp_session() as session:
            video_path = session.save_upload("video.mp4", content)
            # ... 处理 ...
        # 自动清理
    """
    session = TempSession()
    try:
        yield session
    finally:
        session.cleanup()


# =============================================================================
#  清理函数
# =============================================================================


def cleanup_old_sessions(max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> int:
    """
    清理超时的临时会话

    Args:
        max_age_hours: 最大保留时间（小时），默认 24 小时

    Returns:
        清理的会话数量
    """
    if not BASE_DIR.exists():
        return 0

    cleaned = 0
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for session_dir in BASE_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        try:
            age = now - session_dir.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(session_dir, ignore_errors=True)
                cleaned += 1
        except OSError:
            continue

    return cleaned


def get_session(session_id: str) -> TempSession | None:
    """
    获取现有会话

    Args:
        session_id: 会话 ID

    Returns:
        TempSession 实例，如果不存在返回 None
    """
    return TempSession.from_existing(session_id)


def session_exists(session_id: str) -> bool:
    """检查会话是否存在"""
    if not is_valid_session_id(session_id):
        return False
    return (BASE_DIR / session_id).is_dir()
