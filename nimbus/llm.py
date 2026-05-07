import json
import math
import time
import urllib.error
import urllib.request
from array import array
from typing import Sequence


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = 900,
        model: str | None = None,
    ) -> str:
        body = self.post(
            "/chat/completions",
            {
                "model": model or self.chat_model,
                "messages": list(messages),
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=600,
            label="chat completion",
        )

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {body}") from exc

    def embed_text(self, text: str) -> array:
        body = self.post(
            "/embeddings",
            {"model": self.embedding_model, "input": text},
            timeout=120,
            label="embedding",
        )

        try:
            values = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embedding response: {body}") from exc

        vector = array("f", (float(value) for value in values))
        self.normalize_vector(vector)
        return vector

    def post(self, path: str, payload: dict, timeout: int, label: str) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self.request_json(request, timeout=timeout, label=label)

    def request_json(
        self,
        request: urllib.request.Request,
        timeout: int,
        label: str,
        attempts: int = 3,
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(0.75 * (attempt + 1))
        raise RuntimeError(f"Could not complete {label} request at {self.base_url}: {last_error}")

    @staticmethod
    def normalize_vector(vector: array) -> None:
        length = math.sqrt(sum(value * value for value in vector))
        if not length:
            return
        for index, value in enumerate(vector):
            vector[index] = value / length
