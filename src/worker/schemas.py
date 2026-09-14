from typing import Any, Literal

from pydantic import BaseModel, model_validator


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

class PageSceneRenderRequest(BaseModel):
    """New-format worker input. A legacy project payload fails before renderer dispatch."""

    contract_version: Literal["page-scene/v1"]
    page_scene: dict[str, Any]

    @model_validator(mode="after")
    def validate_scene_artifact(self) -> "PageSceneRenderRequest":
        from worker.page_scene import validate_page_scene

        validate_page_scene(self.page_scene)
        return self
