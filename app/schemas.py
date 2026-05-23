from pydantic import BaseModel


class PredictResponse(BaseModel):
    status: str
    is_real: bool
    liveness_score: float
    message: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
