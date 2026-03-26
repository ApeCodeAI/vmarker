"""
[INPUT]: 依赖 FastAPI, video_probe, asr, video_composer, video_composer_parallel, temp_manager, chapter_bar, progress_bar
[OUTPUT]: 对外提供 router (APIRouter 实例)
[POS]: 视频上传和处理 API 路由，支持 ASR 转录和视频合成（含并行）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

from vmarker import asr, chapter_bar as cb, progress_bar as pb, video_composer, video_composer_parallel, video_probe
from vmarker.models import Chapter, ChapterBarConfig, ColorScheme, VideoConfig
from vmarker.progress_bar import ProgressBarConfig
from vmarker.parser import parse_srt
from vmarker.temp_manager import TempSession, cleanup_old_sessions, get_session
from vmarker.themes import THEMES, get_theme


router = APIRouter()


# =============================================================================
#  常量
# =============================================================================

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_DURATION = 300  # 5 分钟
PARALLEL_THRESHOLD_SECONDS = 180  # 超过 3 分钟自动使用并行合成
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


# =============================================================================
#  响应模型
# =============================================================================


class VideoUploadResponse(BaseModel):
    """视频上传响应"""

    session_id: str
    duration: float
    width: int
    height: int
    fps: float
    file_size_mb: float


class ASRResponse(BaseModel):
    """ASR 结果响应"""

    session_id: str
    subtitle_count: int
    duration: float
    srt_content: str


class ComposeRequest(BaseModel):
    """合成请求"""

    feature: str  # "chapter-bar" 或 "progress-bar"
    position: str = "bottom"  # "top" 或 "bottom"
    # Chapter Bar 专用
    chapters: list[Chapter] | None = None
    theme: str = "tech-blue"
    bar_width: int = 1920
    bar_height: int = 60
    # Progress Bar 专用
    played_color: str = "#3B82F6"
    unplayed_color: str = "#E5E7EB"
    progress_height: int = 8
    # 性能优化
    key_frame_interval: float | None = None


class ComposeParallelRequest(BaseModel):
    """并行合成请求"""

    feature: str  # "chapter-bar" 或 "progress-bar"
    position: str = "bottom"  # "top" 或 "bottom"
    # Chapter Bar 专用
    chapters: list[Chapter] | None = None
    theme: str = "tech-blue"
    bar_width: int = 1920
    bar_height: int = 60
    # Progress Bar 专用
    played_color: str = "#3B82F6"
    unplayed_color: str = "#E5E7EB"
    progress_height: int = 8
    # 并行配置
    chunk_seconds: int | None = None  # 分片时长（秒），默认使用环境变量
    max_workers: int | None = None  # 并发上限，默认使用环境变量

    @field_validator("chunk_seconds")
    @classmethod
    def validate_chunk_seconds(cls, v: int | None) -> int | None:
        """验证分片时长"""
        if v is not None and v <= 0:
            raise ValueError("chunk_seconds must be positive")
        return v

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int | None) -> int | None:
        """验证并发上限"""
        if v is not None and v <= 0:
            raise ValueError("max_workers must be positive")
        return v


# =============================================================================
#  路由 - 上传
# =============================================================================


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: Annotated[UploadFile, File(description="视频文件 (mp4/mov/webm/mkv/avi, ≤500MB, ≤5min)")],
):
    """
    上传视频文件

    返回 session_id 用于后续请求（ASR、合成等）。
    会话有效期 24 小时。
    """
    # 验证文件扩展名
    filename = file.filename or "video.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}")

    session = TempSession()
    video_path = session.get_path(f"source{ext}")

    # 流式写入上传内容，避免一次性读入整个文件
    total_size = 0
    try:
        with video_path.open("wb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(400, f"文件大小超出限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)")
                output.write(chunk)
    except HTTPException:
        session.cleanup()
        raise
    except Exception as e:
        session.cleanup()
        raise HTTPException(500, f"文件保存失败: {e}") from e
    finally:
        await file.close()

    # 探测视频信息
    try:
        info = video_probe.validate_video(video_path, MAX_DURATION, MAX_FILE_SIZE / 1024 / 1024)
    except ValueError as e:
        session.cleanup()
        raise HTTPException(400, str(e))
    except Exception as e:
        session.cleanup()
        raise HTTPException(400, f"视频解析失败: {e}")

    return VideoUploadResponse(
        session_id=session.session_id,
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        file_size_mb=info.file_size / 1024 / 1024,
    )


# =============================================================================
#  路由 - ASR
# =============================================================================


@router.post("/asr/{session_id}", response_model=ASRResponse)
async def transcribe_video(session_id: str):
    """
    对已上传的视频进行 ASR 转录

    返回字幕数量和 SRT 内容，SRT 内容会保存到会话目录供后续使用。
    """
    # 获取配置
    api_key = os.getenv("API_KEY", "")
    api_base = os.getenv("ASR_API_BASE", os.getenv("API_BASE", "https://api.openai.com/v1"))
    model = os.getenv("ASR_MODEL", "whisper-1")

    if not api_key:
        raise HTTPException(400, "未配置 API Key，请在 backend/.env 中设置 API_KEY")

    # 获取会话
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在或已过期，请重新上传视频")

    # 查找视频文件
    video_files = session.list_files("source.*")
    if not video_files:
        raise HTTPException(404, "未找到上传的视频")

    video_path = video_files[0]

    # ASR 转录
    try:
        config = asr.ASRConfig(api_key=api_key, api_base=api_base, model=model)
        srt_content = await asr.transcribe_to_srt(video_path, config)
    except Exception as e:
        raise HTTPException(500, f"ASR 转录失败: {e}")

    # 保存 SRT 到会话
    session.save_text("subtitles.srt", srt_content)

    # 解析字幕获取信息
    result = parse_srt(srt_content)

    return ASRResponse(
        session_id=session_id,
        subtitle_count=len(result.subtitles),
        duration=result.duration,
        srt_content=srt_content,
    )


# =============================================================================
#  路由 - 获取会话 SRT
# =============================================================================


@router.get("/srt/{session_id}")
async def get_session_srt(session_id: str):
    """获取会话中的 SRT 内容"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在或已过期")

    if not session.exists("subtitles.srt"):
        raise HTTPException(404, "未找到字幕文件，请先进行 ASR 转录")

    srt_content = session.read_text("subtitles.srt")
    return {"srt_content": srt_content}


# =============================================================================
#  路由 - 视频合成
# =============================================================================


@router.post("/compose/{session_id}")
async def compose_video(session_id: str, request: ComposeRequest):
    """
    将 Bar 合成到原视频

    支持 Chapter Bar 和 Progress Bar 两种合成。
    视频超过 3 分钟自动使用并行合成提升速度。
    """
    # 获取会话
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在或已过期，请重新上传视频")

    # 查找视频文件
    video_files = session.list_files("source.*")
    if not video_files:
        raise HTTPException(404, "未找到上传的视频")

    source_video = video_files[0]
    source_info = video_probe.probe(source_video)

    # 验证位置参数
    if request.position not in ("top", "bottom"):
        raise HTTPException(400, "position 必须是 'top' 或 'bottom'")

    position = video_composer.OverlayPosition.TOP if request.position == "top" else video_composer.OverlayPosition.BOTTOM

    # 根据功能生成 Bar
    if request.feature == "chapter-bar":
        bar_path = _generate_chapter_bar(session, source_info, request)
    elif request.feature == "progress-bar":
        bar_path = _generate_progress_bar(session, source_info, request)
    else:
        raise HTTPException(400, f"不支持的功能: {request.feature}")

    # 合成视频 - 自动选择串行或并行
    output_path = session.get_path("output.mp4")

    # 视频超过阈值时自动使用并行合成
    use_parallel = source_info.duration > PARALLEL_THRESHOLD_SECONDS

    if use_parallel:
        # 并行合成：根据时长动态计算分片大小（目标 3-4 个分片）
        chunk_seconds = max(60, int(source_info.duration / 3))
        parallel_config = video_composer_parallel.ParallelConfig(
            position=position,
            chunk_seconds=chunk_seconds,
        )
        try:
            await video_composer_parallel.compose_vstack_parallel(
                source_video, bar_path, output_path, parallel_config
            )
        except RuntimeError as e:
            raise HTTPException(500, f"视频合成失败: {e}")
    else:
        # 串行合成
        compose_config = video_composer.CompositionConfig(position=position)
        try:
            video_composer.compose_vstack(source_video, bar_path, output_path, compose_config)
        except RuntimeError as e:
            raise HTTPException(500, f"视频合成失败: {e}")

    # 返回合成后的视频
    content = output_path.read_bytes()

    return Response(
        content=content,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="composed.mp4"'},
    )


@router.post("/compose-parallel/{session_id}")
async def compose_video_parallel(session_id: str, request: ComposeParallelRequest):
    """
    并行将 Bar 合成到原视频

    支持长视频分片并行处理，提升处理速度。
    支持 Chapter Bar 和 Progress Bar 两种合成。
    """
    # 获取会话
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在或已过期，请重新上传视频")

    # 查找视频文件
    video_files = session.list_files("source.*")
    if not video_files:
        raise HTTPException(404, "未找到上传的视频")

    source_video = video_files[0]
    source_info = video_probe.probe(source_video)

    # 验证位置参数
    if request.position not in ("top", "bottom"):
        raise HTTPException(400, "position 必须是 'top' 或 'bottom'")

    position = video_composer.OverlayPosition.TOP if request.position == "top" else video_composer.OverlayPosition.BOTTOM

    # 根据功能生成 Bar
    if request.feature == "chapter-bar":
        bar_path = _generate_chapter_bar_parallel(session, source_info, request)
    elif request.feature == "progress-bar":
        bar_path = _generate_progress_bar_parallel(session, source_info, request)
    else:
        raise HTTPException(400, f"不支持的功能: {request.feature}")

    # 并行合成视频
    output_path = session.get_path("output.mp4")
    parallel_config = video_composer_parallel.ParallelConfig(
        position=position,
        chunk_seconds=request.chunk_seconds or video_composer_parallel.DEFAULT_CHUNK_SECONDS,
        max_workers=request.max_workers or video_composer_parallel.DEFAULT_MAX_WORKERS,
    )

    try:
        await video_composer_parallel.compose_vstack_parallel(
            source_video, bar_path, output_path, parallel_config
        )
    except RuntimeError as e:
        raise HTTPException(500, f"并行视频合成失败: {e}")

    # 返回合成后的视频
    content = output_path.read_bytes()

    return Response(
        content=content,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="composed.mp4"'},
    )


def _generate_chapter_bar(session: TempSession, source_info: video_probe.VideoInfo, request: ComposeRequest) -> Path:
    """生成 Chapter Bar 视频"""
    if not request.chapters:
        raise HTTPException(400, "Chapter Bar 需要提供 chapters 参数")

    # 验证章节
    validation = cb.validate(request.chapters, source_info.duration)
    if not validation.valid:
        blocking = [i for i in validation.issues if i.blocking]
        if blocking:
            raise HTTPException(400, f"章节配置错误: {blocking[0].message}")

    # 获取配色方案
    if request.theme in THEMES:
        scheme = get_theme(request.theme)
    else:
        scheme = get_theme("tech-blue")

    # 生成配置
    config = ChapterBarConfig(
        chapters=validation.chapters,
        duration=source_info.duration,
        video=VideoConfig(width=source_info.width, height=request.bar_height),
        theme=request.theme,
    )

    # 生成 Bar 视频
    bar_path = session.get_path("chapter_bar.mp4")
    cb.generate(
        config,
        bar_path,
        format="mp4",
        scheme=scheme,
        key_frame_interval=request.key_frame_interval,
    )

    return bar_path


def _generate_progress_bar(session: TempSession, source_info: video_probe.VideoInfo, request: ComposeRequest) -> Path:
    """生成 Progress Bar 视频"""
    config = ProgressBarConfig(
        duration=source_info.duration,
        width=source_info.width,
        height=request.progress_height,
        played_color=request.played_color,
        unplayed_color=request.unplayed_color,
    )

    bar_path = session.get_path("progress_bar.mp4")
    pb.generate(
        config,
        bar_path,
        format="mp4",
        key_frame_interval=request.key_frame_interval,
    )

    return bar_path


def _generate_chapter_bar_parallel(session: TempSession, source_info: video_probe.VideoInfo, request: ComposeParallelRequest) -> Path:
    """生成 Chapter Bar 视频（并行模式）"""
    if not request.chapters:
        raise HTTPException(400, "Chapter Bar 需要提供 chapters 参数")

    # 验证章节
    validation = cb.validate(request.chapters, source_info.duration)
    if not validation.valid:
        blocking = [i for i in validation.issues if i.blocking]
        if blocking:
            raise HTTPException(400, f"章节配置错误: {blocking[0].message}")

    # 获取配色方案
    if request.theme in THEMES:
        scheme = get_theme(request.theme)
    else:
        scheme = get_theme("tech-blue")

    # 生成配置
    config = ChapterBarConfig(
        chapters=validation.chapters,
        duration=source_info.duration,
        video=VideoConfig(width=source_info.width, height=request.bar_height),
        theme=request.theme,
    )

    # 生成 Bar 视频
    bar_path = session.get_path("chapter_bar.mp4")
    cb.generate(
        config,
        bar_path,
        format="mp4",
        scheme=scheme,
        key_frame_interval=None,  # 并行模式由内部控制 GOP
    )

    return bar_path


def _generate_progress_bar_parallel(session: TempSession, source_info: video_probe.VideoInfo, request: ComposeParallelRequest) -> Path:
    """生成 Progress Bar 视频（并行模式）"""
    config = ProgressBarConfig(
        duration=source_info.duration,
        width=source_info.width,
        height=request.progress_height,
        played_color=request.played_color,
        unplayed_color=request.unplayed_color,
    )

    bar_path = session.get_path("progress_bar.mp4")
    pb.generate(
        config,
        bar_path,
        format="mp4",
        key_frame_interval=None,  # 并行模式由内部控制 GOP
    )

    return bar_path


# =============================================================================
#  路由 - 会话管理
# =============================================================================


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """清理会话"""
    session = get_session(session_id)
    if session:
        session.cleanup()
    return {"status": "cleaned"}


@router.post("/cleanup")
async def cleanup_sessions(max_age_hours: int = 24):
    """清理过期会话（管理接口）"""
    cleaned = cleanup_old_sessions(max_age_hours)
    return {"cleaned": cleaned}
