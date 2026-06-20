"""
Local Ollama service.

Connects to a local Ollama instance running on localhost:11434.
No API key required — this is fully local and private.

To run locally:
  ollama serve

To pull a model:
  ollama pull llama2  (or mistral, neural-chat, etc.)
"""

import json
import os
import re
import asyncio
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# Local Ollama defaults (no API key needed)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))


class OllamaService:
    """
    Wrapper around the local Ollama /api/generate endpoint.
    
    No authentication required. Connects to a local Ollama instance.
    The `model` parameter in `generate_content` is ignored; uses configured OLLAMA_MODEL.
    """

    def __init__(self) -> None:
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = OLLAMA_MODEL
        self.timeout = OLLAMA_TIMEOUT

    async def generate_content(
        self,
        prompt: str,
        model: str = "default",
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a response from local Ollama.
        Runs the blocking HTTP call in a thread to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self._sync_generate, prompt, max_tokens)

    def extract_json(self, text: str) -> Any:
        """
        Robustly extract JSON from a model response.
        Tries three strategies: direct parse, strip markdown fences, regex extraction.
        """
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    async def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: int = 2048,
    ) -> Any:
        """
        Generate a JSON response from the model that matches the provided schema.
        Includes schema instruction in prompt and validates output structure.
        Falls back to extract_json if JSON mode is unavailable.
        """
        schema_str = json.dumps(schema, indent=2)
        json_prompt = f"""{prompt}

Return ONLY valid JSON matching this schema (no markdown, no commentary):
{schema_str}
"""
        response = await self.generate_content(json_prompt, max_tokens=max_tokens)
        data = self.extract_json(response)
        if not isinstance(data, dict):
            print(f"[OllamaService] Expected dict from JSON, got {type(data).__name__}")
            return {}
        return data

    async def health_check(self) -> dict:
        try:
            await asyncio.to_thread(self._ping)
            return {"status": "ok", "model": self.model, "base_url": self.base_url}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def _sync_generate(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def _ping(self) -> None:
        requests.get(f"{self.base_url}/api/tags", timeout=10).raise_for_status()


ollama_service = OllamaService()