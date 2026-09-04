from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.responses_api.serve import expand_checkpoint_path

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

fake_tokens = encoding.encode(
    "<|channel|>final<|message|>Hey there<|return|>", allowed_special="all"
)

token_queue = fake_tokens.copy()


def stub_infer_next_token(
    tokens: list[int], temperature: float = 0.0, new_request: bool = False
) -> int:
    global token_queue
    next_tok = token_queue.pop(0)
    if len(token_queue) == 0:
        token_queue = fake_tokens.copy()
    time.sleep(0.1)
    return next_tok


@pytest.fixture
def test_client():
    return TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )


def test_health_check(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Hello, world!",
        },
    )
    print(response.json())
    assert response.status_code == 200


def test_expand_checkpoint_path_uses_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert Path(expand_checkpoint_path("~/model")) == tmp_path / "model"
