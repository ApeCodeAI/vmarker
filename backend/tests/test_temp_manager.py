"""
[INPUT]: 依赖 pytest, vmarker.temp_manager
[OUTPUT]: temp_manager 模块测试
[POS]: tests/ 的临时会话管理测试
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from vmarker import temp_manager


def test_get_session_missing_returns_none_without_creating_directory(tmp_path, monkeypatch):
    """不存在的 session 不应被隐式创建"""
    monkeypatch.setattr(temp_manager, "BASE_DIR", tmp_path)

    session_id = "missing-session"

    session = temp_manager.get_session(session_id)

    assert session is None
    assert not (tmp_path / session_id).exists()


def test_get_session_existing_returns_session(tmp_path, monkeypatch):
    """已存在的 session 应正常返回"""
    monkeypatch.setattr(temp_manager, "BASE_DIR", tmp_path)

    created = temp_manager.TempSession("abcdef123456")

    session = temp_manager.get_session(created.session_id)

    assert session is not None
    assert session.session_id == created.session_id
    assert session.session_dir == created.session_dir


def test_get_session_rejects_path_traversal_session_id(tmp_path, monkeypatch):
    """非法 session_id 不应访问 BASE_DIR 之外的路径"""
    monkeypatch.setattr(temp_manager, "BASE_DIR", tmp_path)

    session = temp_manager.get_session("../../../etc")

    assert session is None
    assert list(tmp_path.iterdir()) == []


def test_temp_session_rejects_invalid_session_id():
    """显式传入非法 session_id 时应拒绝"""
    try:
        temp_manager.TempSession("../../../etc", create=False)
    except ValueError as exc:
        assert "invalid session_id" in str(exc)
    else:
        raise AssertionError("TempSession should reject invalid session_id")
