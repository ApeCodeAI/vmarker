"""
[INPUT]: 依赖 pytest, FastAPI TestClient, vmarker.api.main
[OUTPUT]: video API 路由测试
[POS]: tests/ 的视频上传接口测试
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import dataclass

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vmarker.api.main import app
from vmarker.api.routes import video as video_route
from vmarker import temp_manager


@dataclass
class FakeVideoInfo:
    duration: float = 12.5
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    codec: str = "h264"
    file_size: int = 0


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_upload_video_streams_to_disk(client, tmp_path, monkeypatch):
    """上传接口应分块写入并保留完整文件"""
    monkeypatch.setattr(temp_manager, "BASE_DIR", tmp_path)

    content = b"a" * (video_route.UPLOAD_CHUNK_SIZE + 17)

    def fake_validate_video(video_path, max_duration, max_size_mb):
        assert video_path.read_bytes() == content
        return FakeVideoInfo(file_size=len(content))

    monkeypatch.setattr(video_route.video_probe, "validate_video", fake_validate_video)

    response = client.post(
        "/api/v1/video/upload",
        files={"file": ("sample.mp4", content, "video/mp4")},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["width"] == 1920
    assert data["height"] == 1080
    assert data["fps"] == 30.0
    assert data["file_size_mb"] == pytest.approx(len(content) / 1024 / 1024)

    session_dir = tmp_path / data["session_id"]
    assert (session_dir / "source.mp4").read_bytes() == content


def test_upload_video_rejects_oversized_file_and_cleans_session(client, tmp_path, monkeypatch):
    """超限上传应立即失败并清理临时目录"""
    monkeypatch.setattr(temp_manager, "BASE_DIR", tmp_path)
    monkeypatch.setattr(video_route, "MAX_FILE_SIZE", video_route.UPLOAD_CHUNK_SIZE)

    validate_called = False

    def fake_validate_video(*args, **kwargs):
        nonlocal validate_called
        validate_called = True
        raise AssertionError("validate_video should not be called for oversized uploads")

    monkeypatch.setattr(video_route.video_probe, "validate_video", fake_validate_video)

    content = b"b" * (video_route.UPLOAD_CHUNK_SIZE + 1)

    response = client.post(
        "/api/v1/video/upload",
        files={"file": ("oversized.mp4", content, "video/mp4")},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "文件大小超出限制" in response.text
    assert validate_called is False
    assert list(tmp_path.iterdir()) == []
