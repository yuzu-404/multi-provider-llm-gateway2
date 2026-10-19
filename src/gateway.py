
import os
import time
from collections.abc import AsyncGenerator
from typing import Self

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from src.pricing import MODEL_PRICING, calculate_cost
from src.schemas import ChunkTelemetry, StreamChunk


class GatewayProviderError(Exception):
    """Raised when an AI provider request fails."""


class GatewayAuthenticationError(GatewayProviderError):
    """Raised when API credentials are missing or invalid."""


class RateLimitExceededError(GatewayProviderError):
    """Raised when a provider rate limit is exceeded."""


class UnifiedLLMGateway:
    def __init__(self) -> None:
        self.openai_client = (
            AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            if os.getenv("OPENAI_API_KEY")
            else None
        )

        self.anthropic_client = (
            AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            if os.getenv("ANTHROPIC_API_KEY")
            else None
        )

    async def close(self) -> None:
        """Close persistent provider connections."""
        if self.openai_client is not None:
            await self.openai_client.close()

        if self.anthropic_client is not None:
            await self.anthropic_client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.close()

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response and yield final request telemetry."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1.")

        if model not in MODEL_PRICING:
            raise ValueError(f"Unsupported model: {model}")

        if model.startswith("gpt-"):
            provider = "openai"
        elif model.startswith("claude-"):
            provider = "anthropic"
        else:
            raise ValueError(f"Unknown provider for model: {model}")

        start = time.perf_counter()
        first_token_time: float | None = None
        input_tokens = 0
        output_tokens = 0

        try:
            if provider == "openai":
                if self.openai_client is None:
                    raise GatewayAuthenticationError(
                        "OpenAI API key is not configured."
                    )

                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                generated_text = ""
                usage_received = False

                try:
                    async for event in response:
                        if event.usage is not None:
                            input_tokens = event.usage.prompt_tokens
                            output_tokens = event.usage.completion_tokens
                            usage_received = True

                        if not event.choices:
                            continue

                        delta = event.choices[0].delta.content

                        if not delta:
                            continue

                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        generated_text += delta

                        yield StreamChunk(
                            delta_text=delta,
                            provider=provider,
                            model=model,
                        )
                finally:
                    await response.close()

                if not usage_received:
                    encoding = tiktoken.encoding_for_model(model)
                    input_tokens = len(encoding.encode(prompt))
                    output_tokens = len(
                        encoding.encode(generated_text)
                    )

            else:
                if self.anthropic_client is None:
                    raise GatewayAuthenticationError(
                        "Anthropic API key is not configured."
                    )

                async with self.anthropic_client.messages.stream(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                ) as response:
                    async for delta in response.text_stream:
                        if not delta:
                            continue

                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        yield StreamChunk(
                            delta_text=delta,
                            provider=provider,
                            model=model,
                        )

                    final_message = await response.get_final_message()

                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens

            end = time.perf_counter()
            latency = end - start

            ttft_ms = (
                (first_token_time - start) * 1000
                if first_token_time is not None
                else 0.0
            )

            generation_seconds = (
                end - first_token_time
                if first_token_time is not None
                else 0.0
            )

            tokens_per_second = (
                output_tokens / generation_seconds
                if generation_seconds > 0
                else 0.0
            )

            cost = calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            telemetry = ChunkTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_seconds=latency,
                ttft_ms=ttft_ms,
                tokens_per_second=tokens_per_second,
                total_cost_usd=float(cost),
            )

            yield StreamChunk(
                delta_text="",
                provider=provider,
                model=model,
                is_final=True,
                telemetry=telemetry,
            )

        except (
            openai.AuthenticationError,
            anthropic.AuthenticationError,
        ) as error:
            raise GatewayAuthenticationError(
                "Provider authentication failed. Check your API key."
            ) from error

        except (
            openai.RateLimitError,
            anthropic.RateLimitError,
        ) as error:
            raise RateLimitExceededError(
                "Provider rate limit exceeded. Try again later."
            ) from error

        except (
            openai.APIConnectionError,
            anthropic.APIConnectionError,
        ) as error:
            raise GatewayProviderError(
                "Could not connect to the AI provider."
            ) from error
