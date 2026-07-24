from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Any, Dict


class JobCompletionRequest(BaseModel):
    pageId: str
    imageId: str
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class JobFailureRequest(BaseModel):
    pageId: str
    imageId: str
    status: str = "FAILED"
    errorReason: str
    errorMessage: str
    details: Optional[Dict[str, Any]] = None


class JobData(BaseModel):
    jobId: str
    imageId: str
    pageId: Optional[str] = None
    attempt: int = 1
    maxAttempts: int = 3
    # Allow extra fields for specific job types
    model_config = {"extra": "allow"}


class JobSubmitRequest(BaseModel):
    queue_name: str
    job_data: JobData
