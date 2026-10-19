
# AI Prompt Log and Code Audit

## 1. Initial AI Prompt

AI assistant used: GitHub Copilot / ChatGPT

The initial implementation was generated with AI assistance following the assignment's suggested prompt.

The requested implementation included:

- Python 3.11+ and Pydantic v2
- A `UnifiedLLMGateway` class with an asynchronous `stream()` method
- OpenAI and Anthropic support using official asynchronous SDKs
- A normalized `StreamChunk` response format
- Input/output token tracking
- Time-to-first-token, latency, tokens-per-second, and USD cost calculations
- Typed error handling for authentication, rate limits, and connection failures
- Code organized into `gateway.py`, `schemas.py`, and `pricing.py`

## 2. AI-Generated Code Audit

To be completed after reviewing the generated implementation.

## 3. Testing and Verification

The current project passes 8 automated pytest tests.

Live provider requests have not yet been verified.


### Issue: Misleading Temperature Control for Anthropic

**Problem:**
The Streamlit interface allowed users to adjust temperature for both providers, but the gateway only forwarded the temperature value to OpenAI. The selected Anthropic SDK methods did not expose a temperature parameter.

**Before:**
```python
temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
)
```

**After:**
```python
temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    disabled=model.startswith("claude-"),
)
```

**Verification:**
Inspected the installed Anthropic SDK method signatures and confirmed neither `messages.stream()` nor `messages.create()` exposes a temperature argument. Manually tested the Streamlit interface and confirmed the slider becomes disabled when an Anthropic model is selected.

**Result:**
The interface no longer suggests that Anthropic temperature can be adjusted. This is a UI correction, not an implementation of Anthropic temperature support.
