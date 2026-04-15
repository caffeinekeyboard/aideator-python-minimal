"""FastAPI smoke tests (no GEMINI_API_KEY required for /health)."""

from fastapi.testclient import TestClient

from application.api.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_session() -> None:
    r = client.post("/api/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert len(sid) > 10
