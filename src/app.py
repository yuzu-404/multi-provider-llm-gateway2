import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from src.gateway import (
    GatewayAuthenticationError,
    GatewayProviderError,
    RateLimitExceededError,
    UnifiedLLMGateway,
)

load_dotenv()

st.set_page_config(
    page_title="Multi-Provider LLM Gateway",
    page_icon="🤖",
    layout="wide",
)

st.title("Multi-Provider LLM Gateway")
st.caption("Stream responses from OpenAI and Anthropic with live telemetry.")

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
]

with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Model",
        options=MODEL_OPTIONS,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        disabled=model.startswith("claude-"),
    )

    provider = "OpenAI" if model.startswith("gpt-") else "Anthropic"

    st.write(f"**Provider:** {provider}")

    key_name = (
        "OPENAI_API_KEY"
        if provider == "OpenAI"
        else "ANTHROPIC_API_KEY"
    )

    if os.getenv(key_name):
        st.success("API key configured")
    else:
        st.warning("API key not configured")

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Explain asynchronous programming in simple terms.",
    height=140,
)

run_button = st.button(
    "Generate Response",
    type="primary",
    disabled=not prompt.strip(),
)

st.subheader("Streaming Response")
response_placeholder = st.empty()

st.subheader("Request Telemetry")

col1, col2, col3 = st.columns(3)

input_metric = col1.empty()
output_metric = col2.empty()
cost_metric = col3.empty()

col4, col5, col6 = st.columns(3)

ttft_metric = col4.empty()
speed_metric = col5.empty()
latency_metric = col6.empty()


def show_metrics(telemetry):
    input_metric.metric("Input Tokens", telemetry.input_tokens)
    output_metric.metric("Output Tokens", telemetry.output_tokens)
    cost_metric.metric(
        "Estimated Cost",
        f"${telemetry.total_cost_usd:.6f}",
    )
    ttft_metric.metric(
        "Time to First Token",
        f"{telemetry.ttft_ms:.2f} ms",
    )
    speed_metric.metric(
        "Generation Speed",
        f"{telemetry.tokens_per_second:.2f} tokens/s",
    )
    latency_metric.metric(
        "Total Latency",
        f"{telemetry.total_latency_seconds:.2f} s",
    )


async def generate_response():
    accumulated_text = ""

    async with UnifiedLLMGateway() as gateway:
        async for chunk in gateway.stream(
            prompt=prompt,
            model=model,
            temperature=temperature,
        ):
            if chunk.delta_text:
                accumulated_text += chunk.delta_text
                response_placeholder.markdown(accumulated_text + "▌")

            if chunk.is_final:
                response_placeholder.markdown(accumulated_text)
                if chunk.telemetry is not None:
                    show_metrics(chunk.telemetry)


if run_button:
    if not os.getenv(key_name):
        st.error(
            f"{provider} API key is missing. "
            "Configure your .env file before making requests."
        )
    else:
        try:
            asyncio.run(generate_response())
        except GatewayAuthenticationError as error:
            st.error(str(error))
        except RateLimitExceededError as error:
            st.error(str(error))
        except GatewayProviderError as error:
            st.error(str(error))
        except Exception:
            st.error(
                "An unexpected error occurred. "
                "Check the application logs."
            )
