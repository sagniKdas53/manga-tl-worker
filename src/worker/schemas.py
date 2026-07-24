from typing import Any

from pydantic import BaseModel


class JobCompletionRequest(BaseModel):
    pageId: str
    imageId: str
    status: str
    message: str | None = None
    data: dict[str, Any] | None = None


class JobFailureRequest(BaseModel):
    pageId: str
    imageId: str
    status: str = "FAILED"
    errorReason: str
    errorMessage: str
    details: dict[str, Any] | None = None


class JobData(BaseModel):
    jobId: str
    imageId: str
    pageId: str | None = None
    attempt: int = 1
    maxAttempts: int = 3
    # Allow extra fields for specific job types
    model_config = {"extra": "allow"}


class JobSubmitRequest(BaseModel):
    queue_name: str
    job_data: JobData
