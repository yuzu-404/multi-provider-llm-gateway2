from pydantic import BaseModel, Field


class ChunkTelemetry(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_latency_seconds: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    tokens_per_second: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)


class StreamChunk(BaseModel):
    delta_text: str
    provider: str
    model: str
    is_final: bool = False
    telemetry: ChunkTelemetry | None = None
