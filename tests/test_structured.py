"""Tests for robust structured output parsing, think-tag sanitization, and reasoning auto-detection."""

from unittest.mock import MagicMock
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from deep_researcher.config.settings import resolve_reasoning_default
from deep_researcher.models.schemas import SupervisorDecision
from deep_researcher.utils.structured import extract_and_validate_json, invoke_structured


class SampleSchema(BaseModel):
    action: str
    reasoning: str
    score: int = 10


def test_resolve_reasoning_default():
    """Verify auto-detection and explicit env overrides for reasoning mode."""
    # Auto-detection when env_val is None or empty
    assert resolve_reasoning_default("deepseek-r1:14b", None) is True
    assert resolve_reasoning_default("deepseek-r1:70b", "") is True
    assert resolve_reasoning_default("o1-mini", None) is True
    assert resolve_reasoning_default("o3-mini", None) is True
    assert resolve_reasoning_default("qwen3.5:9b", None) is False
    assert resolve_reasoning_default("llama3.3:70b", None) is False
    assert resolve_reasoning_default("gpt-4o", None) is False
    assert resolve_reasoning_default("claude-3-5-sonnet", None) is False

    # Explicit env overrides take precedence
    assert resolve_reasoning_default("qwen3.5:9b", "true") is True
    assert resolve_reasoning_default("qwen3.5:9b", "1") is True
    assert resolve_reasoning_default("qwen3.5:9b", "yes") is True
    assert resolve_reasoning_default("deepseek-r1:14b", "false") is False
    assert resolve_reasoning_default("deepseek-r1:14b", "0") is False
    assert resolve_reasoning_default("deepseek-r1:14b", "no") is False

    # Whitespace-only falls back to model auto-detection
    assert resolve_reasoning_default("qwen3.5:9b", "   ") is False
    assert resolve_reasoning_default("deepseek-r1:14b", "   ") is True


def test_extract_and_validate_clean_json():
    """Verify clean JSON string parsing."""
    raw = '{"action": "web_research", "reasoning": "Need more info", "score": 8}'
    result = extract_and_validate_json(raw, SampleSchema)
    assert result.action == "web_research"
    assert result.reasoning == "Need more info"
    assert result.score == 8


def test_extract_and_validate_with_think_tags():
    """Verify stripping of <think>...</think> blocks."""
    raw = """<think>
    The user wants an analysis of quantum algorithms.
    I should check if we have enough empirical papers.
    </think>
    {"action": "paper_research", "reasoning": "Focus on theoretical foundation", "score": 9}"""
    result = extract_and_validate_json(raw, SampleSchema)
    assert result.action == "paper_research"
    assert result.reasoning == "Focus on theoretical foundation"
    assert result.score == 9


def test_extract_and_validate_with_markdown_fences():
    """Verify markdown code block extraction."""
    raw = """Here is the structured decision:
```json
{
  "action": "DONE",
  "reasoning": "All questions sufficiently covered",
  "score": 10
}
```
Let me know if you need more."""
    result = extract_and_validate_json(raw, SampleSchema)
    assert result.action == "DONE"
    assert result.reasoning == "All questions sufficiently covered"
    assert result.score == 10


def test_extract_and_validate_with_think_and_fences():
    """Verify handling when BOTH think blocks and code fences are present."""
    raw = """<think>
    Multi-line thinking monologue...
    </think>
    ```
    {
      "action": "web_research",
      "reasoning": "Missing market stats",
      "score": 5
    }
    ```"""
    result = extract_and_validate_json(raw, SampleSchema)
    assert result.action == "web_research"
    assert result.reasoning == "Missing market stats"
    assert result.score == 5


def test_extract_and_validate_invalid_raises():
    """Verify invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match="Empty output text"):
        extract_and_validate_json("", SampleSchema)

    with pytest.raises(ValueError, match="Empty output text"):
        extract_and_validate_json("   ", SampleSchema)

    with pytest.raises(ValueError, match="No JSON object found"):
        extract_and_validate_json("Just some plain conversational text without any braces.", SampleSchema)


def test_invoke_structured_native_success():
    """Verify direct success when llm.with_structured_output works."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    expected = SampleSchema(action="DONE", reasoning="All clear", score=10)
    mock_structured.invoke.return_value = expected
    mock_llm.with_structured_output.return_value = mock_structured

    result = invoke_structured(
        mock_llm,
        SampleSchema,
        [SystemMessage(content="Sys"), HumanMessage(content="User")],
    )
    assert result == expected
    mock_structured.invoke.assert_called_once()
    mock_llm.invoke.assert_not_called()


def test_invoke_structured_recovers_from_exception_raw_text():
    """Verify recovery when OutputParserException contains raw text with think tags."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    # Simulate OutputParserException with preserved raw output
    exc = OutputParserException(
        "Invalid json output: <think>thinking...</think> {\"action\": \"recovered\", \"reasoning\": \"via raw text\", \"score\": 7}",
        llm_output="<think>thinking...</think> {\"action\": \"recovered\", \"reasoning\": \"via raw text\", \"score\": 7}",
    )
    mock_structured.invoke.side_effect = exc
    mock_llm.with_structured_output.return_value = mock_structured

    result = invoke_structured(
        mock_llm,
        SampleSchema,
        [HumanMessage(content="Evaluate")],
    )
    assert result.action == "recovered"
    assert result.reasoning == "via raw text"
    assert result.score == 7
    # Notice base LLM was NOT re-invoked because exception raw text was successfully recovered
    mock_llm.invoke.assert_not_called()


def test_invoke_structured_recovers_via_json_mode():
    """Verify recovery via method='json_mode' when native json_schema fails."""
    mock_llm = MagicMock()
    mock_native = MagicMock()
    mock_json_mode = MagicMock()

    # Native structured output fails with empty string
    mock_native.invoke.side_effect = OutputParserException("Invalid json output: ", llm_output="")
    # json_mode succeeds
    expected = SampleSchema(action="web_research", reasoning="json_mode success", score=9)
    mock_json_mode.invoke.return_value = expected

    def fake_with_structured(schema, **kwargs):
        if kwargs.get("method") == "json_mode":
            return mock_json_mode
        return mock_native

    mock_llm.with_structured_output.side_effect = fake_with_structured

    result = invoke_structured(
        mock_llm,
        SampleSchema,
        [HumanMessage(content="Evaluate")],
    )
    assert result == expected
    mock_json_mode.invoke.assert_called_once()
    # Raw LLM was not needed because json_mode recovered
    mock_llm.invoke.assert_not_called()


def test_invoke_structured_recovers_via_raw_reinvocation():
    """Verify recovery via raw LLM re-invocation when exception has no raw text (token exhaustion)."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    # Simulate empty output parser exception (what Ollama gives on token exhaustion)
    exc = OutputParserException("Invalid json output: ", llm_output="")
    mock_structured.invoke.side_effect = exc
    mock_llm.with_structured_output.return_value = mock_structured

    # Mock raw LLM return
    mock_llm.invoke.return_value = AIMessage(
        content='```json\n{"action": "web_research", "reasoning": "retry worked", "score": 9}\n```'
    )

    result = invoke_structured(
        mock_llm,
        SampleSchema,
        [HumanMessage(content="Evaluate")],
    )
    assert result.action == "web_research"
    assert result.reasoning == "retry worked"
    assert result.score == 9
    mock_llm.invoke.assert_called_once()


def test_invoke_structured_raises_on_complete_failure():
    """Verify exception is re-raised if both native and retry recovery fail."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    original_exc = OutputParserException("Invalid json output: ", llm_output="")
    mock_structured.invoke.side_effect = original_exc
    mock_llm.with_structured_output.return_value = mock_structured

    # Raw invoke returns unparseable junk
    mock_llm.invoke.return_value = AIMessage(content="Total gibberish without any JSON")

    with pytest.raises(OutputParserException):
        invoke_structured(
            mock_llm,
            SampleSchema,
            [HumanMessage(content="Evaluate")],
        )
