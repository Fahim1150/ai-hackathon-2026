from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_shape():
    r = client.post("/predict", json={"features": {}}).json()
    assert {"score", "rules_triggered", "reasons", "needs_human_review"} <= r.keys()
