"""Shared loader of the LLM client and credentials from the environment."""

import logging
import os

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_llm_client() -> tuple[openai.OpenAI | None, str | None]:
    """Load LLM credentials from ``.env`` and build an OpenAI client.

    Reads ``API_KEY``, ``SERVER``, and ``LLM_MODEL`` from the environment and
    returns ``(client, model)``. If any of the three is missing, returns
    ``(None, None)``.
    """
    load_dotenv()
    api_key = os.getenv("API_KEY")
    server = os.getenv("SERVER")
    model = os.getenv("LLM_MODEL")
    if api_key and server and model:
        return openai.OpenAI(base_url=server, api_key=api_key), model
    logger.warning(
        "LLM credentials missing - LLM disabled. "
        "Set API_KEY, SERVER, and LLM_MODEL in .env to enable."
    )
    return None, None
