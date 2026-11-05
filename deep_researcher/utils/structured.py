"""Robust structured output invocation utility with multi-stage sanitization and recovery."""

import json
import logging
import re
from typing import Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_and_validate_json(text: str, schema: type[T]) -> T:
    """Extract a valid JSON object from raw text and validate it against the Pydantic schema.

    Handles:
    - Leaked `<think>...</think>` tokens
    - Markdown ```json ... ``` fences
    - Extraneous commentary before or after JSON
    """
    if not text or not text.strip():
        raise ValueError("Empty output text received from model.")

    # 1. Strip think blocks (case-insensitive, multiline)
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text).strip()

    # 2. If a markdown code fence exists, prioritize its interior
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else cleaned

    # 3. Locate outermost curly braces
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        target = candidate[start : end + 1].strip()
    else:
        raise ValueError(f"No JSON object found in cleaned text: {candidate[:200]}")

    data = json.loads(target)
    return schema.model_validate(data)


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage] | list[BaseMessage],
) -> T:
    """Invoke LLM with structured output, backed by sanitization and fallback recovery.

    1. Attempts native structured output invocation (`llm.with_structured_output(schema)`).
    2. If an OutputParserException or JSONDecodeError is encountered, attempts to extract
       and validate the schema from the raw output preserved in the exception.
    3. If the exception has no raw output (e.g. empty response on token limit), makes a raw
       invocation with explicit JSON schema instructions and runs extraction/validation.
    4. If all recovery attempts fail, re-raises the exception for upstream domain fallbacks.
    """
    structured_llm = llm.with_structured_output(schema)

    try:
        return structured_llm.invoke(messages)
    except Exception as exc:
        logger.warning(
            f"[invoke_structured] Native structured output failed ({exc}). "
            "Attempting sanitization recovery..."
        )

        # Stage 1: Try recovering directly from raw text inside the exception if available
        raw_text = getattr(exc, "llm_output", None)
        if isinstance(raw_text, str) and raw_text.strip():
            try:
                recovered = extract_and_validate_json(raw_text, schema)
                logger.info(
                    "[invoke_structured] Successfully recovered structured output from parser exception raw text."
                )
                return recovered
            except Exception as recover_exc:
                logger.debug(
                    f"[invoke_structured] Recovery from exception raw text failed: {recover_exc}"
                )

        # Stage 2: Attempt json_mode fallback (bypasses brittle json_schema GBNF compiler in local runners)
        try:
            json_mode_llm = llm.with_structured_output(schema, method="json_mode")
            recovered = json_mode_llm.invoke(messages)
            logger.info(
                "[invoke_structured] Successfully recovered structured output via json_mode fallback."
            )
            return recovered
        except Exception as json_mode_exc:
            logger.debug(f"[invoke_structured] json_mode fallback failed: {json_mode_exc}")
            raw_json_mode = getattr(json_mode_exc, "llm_output", None)
            if isinstance(raw_json_mode, str) and raw_json_mode.strip():
                try:
                    return extract_and_validate_json(raw_json_mode, schema)
                except Exception:
                    pass

        # Stage 3: Fall back to raw base model invocation with schema formatting prompt
        try:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            recovery_prompt = (
                f"\n\nIMPORTANT: Respond ONLY with a valid JSON object matching this schema. "
                f"Do NOT include any commentary, thinking tags, or text outside the JSON.\n"
                f"JSON Schema:\n{schema_json}"
            )
            recovery_messages = list(messages)
            if recovery_messages and isinstance(recovery_messages[-1], HumanMessage):
                last_msg = recovery_messages[-1]
                recovery_messages[-1] = HumanMessage(
                    content=f"{last_msg.content}\n{recovery_prompt}",
                    additional_kwargs=last_msg.additional_kwargs,
                )
            else:
                recovery_messages.append(HumanMessage(content=recovery_prompt))

            raw_response = llm.invoke(recovery_messages)
            content = (
                raw_response.content
                if isinstance(raw_response.content, str)
                else str(raw_response.content)
            )
            recovered = extract_and_validate_json(content, schema)
            logger.info(
                "[invoke_structured] Successfully recovered structured output via raw model re-invocation."
            )
            return recovered
        except Exception as retry_exc:
            logger.error(
                f"[invoke_structured] Sanitization and retry recovery failed ({retry_exc}). Raising original exception."
            )
            raise exc
