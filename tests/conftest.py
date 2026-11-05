import pytest


@pytest.fixture(autouse=True)
def _private_records(tmp_path, monkeypatch):
    """Every test gets its own records database and backup folder - never the real ones."""
    monkeypatch.setenv("CHICKEN_TRACKER_RECORDS", str(tmp_path / "_records"))
    monkeypatch.setenv("CHICKEN_TRACKER_BACKUPS", str(tmp_path / "_backups"))
