"""
bedrock_rag_llm.py — Amazon Bedrock LLM Generation (Claude / Nova)
──────────────────────────────────────────────────────────────────
Generates citation-backed answers using foundation models hosted on
Amazon Bedrock via LangChain AWS (`ChatBedrock`).

Supported Models:
  • anthropic.claude-3-haiku-20240307-v1:0  (Fast, cost-effective)
  • anthropic.claude-3-sonnet-20240229-v1:0 (High capability)
  • amazon.nova-lite-v1:0                   (Amazon Nova)

Environment Variables:
  AWS_REGION=us-east-1
  AWS_ACCESS_KEY_ID=your_key
  AWS_SECRET_ACCESS_KEY=your_secret
  BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


def _build_context_prompt(question: str, context_chunks: List[dict[str, Any]]) -> str:
    """Formats retrieved vector search context chunks into a structured prompt."""
    context_lines = []
    for idx, chunk in enumerate(context_chunks, start=1):
        meta = chunk.get("metadata", {})
        doc = meta.get("document_name", "Unknown Document")
        page = meta.get("page_number", "?")
        snippet = chunk.get("text_snippet", "") or meta.get("text_snippet", "")
        context_lines.append(f"[Source {idx}] {doc} (Page {page}):\n{snippet.strip()}")

    formatted_context = "\n\n".join(context_lines)
    return (
        "You are an expert academic and technical research assistant.\n"
        "Use the retrieved document context below to answer the user's question accurately.\n"
        "Cite specific document names and page numbers for all factual claims.\n"
        "If the provided context does not contain sufficient information, state that clearly.\n\n"
        f"RETRIEVED CONTEXT:\n{formatted_context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def get_bedrock_llm(model_id: Optional[str] = None):
    """
    Initializes and returns a LangChain ChatBedrock instance.
    Raises EnvironmentError if AWS credentials or region are missing.
    """
    try:
        from langchain_aws import ChatBedrock
    except ImportError:
        raise ImportError(
            "langchain_aws is not installed. Run: pip install langchain-aws boto3"
        )

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    resolved_model = model_id or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)

    return ChatBedrock(
        model_id=resolved_model,
        region_name=region,
        model_kwargs={"temperature": 0.2, "max_tokens": 2048},
    )


def generate_answer(
    question: str,
    context_chunks: List[dict[str, Any]],
    model_id: Optional[str] = None,
) -> str:
    """
    Generates an answer using Amazon Bedrock foundation models.

    Args:
        question: User query string
        context_chunks: List of retrieved context chunks with metadata & snippets
        model_id: Optional Bedrock model ID override

    Returns:
        Generated answer text string
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        logger.warning("AWS_ACCESS_KEY_ID not set in environment.")

    llm = get_bedrock_llm(model_id=model_id)
    prompt = _build_context_prompt(question, context_chunks)

    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


__all__ = ["generate_answer", "get_bedrock_llm"]
