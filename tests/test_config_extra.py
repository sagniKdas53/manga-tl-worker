from worker.config import _is_sensitive, _load_docker_secrets, logger


def test_is_sensitive():
    assert _is_sensitive("/path/to/.env") is True
    assert _is_sensitive("/path/to/.ssh/key") is True
    assert _is_sensitive("/path/to/normal.txt") is False


def test_load_docker_secrets():
    _load_docker_secrets()
    assert logger is not None
