
import pytest

from src.gateway import (
    UnifiedLLMGateway,
    GatewayAuthenticationError,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, environment_variable",
    [
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ],
)
async def test_missing_api_key(monkeypatch, model, environment_variable):
    monkeypatch.delenv(environment_variable, raising=False)

    async with UnifiedLLMGateway() as gateway:
        with pytest.raises(GatewayAuthenticationError) as error:
            async for _ in gateway.stream(
                prompt="Hello",
                model=model,
            ):
                pass

    assert "sk-" not in str(error.value)
