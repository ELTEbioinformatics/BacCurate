"""LLM client construction."""

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
import openai
from dotenv import load_dotenv

from baccurate.llm.diagnostics import ObservedLLMTransport

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMSettings:
    api_key: str | None
    server: str | None
    model: str | None


def load_llm_settings() -> LLMSettings:
    load_dotenv()
    return LLMSettings(
        api_key=os.getenv("API_KEY"),
        server=os.getenv("SERVER"),
        model=os.getenv("LLM_MODEL"),
    )


def load_llm_client(
    settings: LLMSettings | None = None,
) -> tuple[openai.OpenAI | None, str | None]:
    """Build the configured OpenAI client, or disable model calls when unconfigured."""
    settings = settings or load_llm_settings()
    api_key = settings.api_key
    server = settings.server
    model = settings.model
    if api_key and server and model:
        server_parts = urlsplit(server)
        secrets = {api_key, server_parts.username or "", server_parts.password or ""}
        transport = ObservedLLMTransport(httpx.HTTPTransport(), configured_secrets=secrets)
        http_client = openai.DefaultHttpxClient(transport=transport)
        return openai.OpenAI(base_url=server, api_key=api_key, http_client=http_client), model
    logger.warning(
        "LLM credentials missing - LLM disabled. "
        "Set API_KEY, SERVER, and LLM_MODEL in .env to enable."
    )
    return None, None
