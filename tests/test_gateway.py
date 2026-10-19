
import pytest

from src.gateway import UnifiedLLMGateway
from types import SimpleNamespace
from unittest.mock import AsyncMock
from typing import cast
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


@pytest.mark.asyncio
async def test_empty_prompt_is_rejected():
    async with UnifiedLLMGateway() as gateway:
        with pytest.raises(ValueError):
            async for _ in gateway.stream(
                prompt="",
                model="gpt-4o-mini",
            ):
                pass

@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [-0.1, 1.1])
async def test_invalid_temperature_is_rejected(temperature):
    async with UnifiedLLMGateway() as gateway:
        with pytest.raises(ValueError):
            async for _ in gateway.stream(
                prompt="Hello",
                model="gpt-4o-mini",
                temperature=temperature,
            ):
                pass




class FakeOpenAIStream:
    def __init__(self):
        self.close = AsyncMock()

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for text in ("Hello", " world"):
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text)
                    )
                ],
            )

        yield SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
            ),
            choices=[],
        )


@pytest.mark.asyncio
async def test_openai_streaming_and_telemetry():
    gateway = UnifiedLLMGateway()

    fake_stream = FakeOpenAIStream()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=fake_stream)
            )
        ),
        close=AsyncMock(),
    )

    gateway.openai_client = cast(AsyncOpenAI, fake_client)

    async with gateway:
        chunks = [
            chunk
            async for chunk in gateway.stream(
                prompt="Say hello",
                model="gpt-4o-mini",
            )
        ]

    assert [chunk.delta_text for chunk in chunks] == [
        "Hello",
        " world",
        "",
    ]

    assert chunks[-1].is_final is True
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.input_tokens == 10
    assert chunks[-1].telemetry.output_tokens == 2
    assert chunks[-1].telemetry.total_latency_seconds >= 0
    assert chunks[-1].telemetry.ttft_ms >= 0
    assert chunks[-1].telemetry.tokens_per_second > 0
    assert chunks[-1].telemetry.total_cost_usd == 0.000003

    fake_stream.close.assert_awaited_once()

class FakeAnthropicStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    @property
    def text_stream(self):
        return self._text_chunks()

    async def _text_chunks(self):
        yield "Hello"
        yield " from Anthropic"

    async def get_final_message(self):
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=4,
            )
        )


@pytest.mark.asyncio
async def test_anthropic_streaming_and_telemetry():
    gateway = UnifiedLLMGateway()

    fake_stream = FakeAnthropicStream()
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            stream=lambda **kwargs: fake_stream
        ),
        close=AsyncMock(),
    )

    gateway.anthropic_client = cast(AsyncAnthropic, fake_client)

    async with gateway:
        chunks = [
            chunk
            async for chunk in gateway.stream(
                prompt="Say hello",
                model="claude-sonnet-5-5",
            )
        ]

    assert [chunk.delta_text for chunk in chunks] == [
        "Hello",
        " from Anthropic",
        "",
    ]

    assert chunks[-1].is_final is True
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.input_tokens == 12
    assert chunks[-1].telemetry.output_tokens == 4
    assert chunks[-1].telemetry.total_cost_usd == 0.000064

    fake_client.close.assert_awaited_once()

import tiktoken


@pytest.mark.asyncio
async def test_openai_token_fallback():
    class FakeStreamWithoutUsage:
        def __init__(self):
            self.close = AsyncMock()

        def __aiter__(self):
            return self.events()

        async def events(self):
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hello world")
                    )
                ],
            )

    gateway = UnifiedLLMGateway()
    fake_stream = FakeStreamWithoutUsage()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=fake_stream)
            )
        ),
        close=AsyncMock(),
    )

    gateway.openai_client = cast(AsyncOpenAI, fake_client)

    async with gateway:
        chunks = [
            chunk
            async for chunk in gateway.stream(
                prompt="Say hello",
                model="gpt-4o-mini",
            )
        ]

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    expected_input = len(encoding.encode("Say hello"))
    expected_output = len(encoding.encode("Hello world"))

    telemetry = chunks[-1].telemetry

    assert telemetry is not None
    assert telemetry.input_tokens == expected_input
    assert telemetry.output_tokens == expected_output
    assert chunks[-1].is_final is True



