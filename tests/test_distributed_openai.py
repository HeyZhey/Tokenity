from __future__ import annotations

from fastapi.testclient import TestClient

from tokenity.serving.distributed_openai import create_app


def test_distributed_skeleton_readiness_and_models():
    with TestClient(create_app(model="/models/qwen")) as client:
        readiness = client.get("/v1/readiness").json()
        models = client.get("/v1/models").json()

    assert readiness["phase"] == "ready"
    assert models["data"][0]["id"] == "/models/qwen"

