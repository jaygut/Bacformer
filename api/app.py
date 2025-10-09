"""FastAPI skeleton for FoodGuard AI MVP.

Note: This module is a scaffold. It references FastAPI but does not require
it to be installed for the rest of the repository to function. Import this
module only when running the API service.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore
    UploadFile = object  # type: ignore
    JSONResponse = dict  # type: ignore

from foodguard.config import PipelineConfig
from foodguard import FoodGuardPipeline


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

    app = FastAPI()
    pipeline = FoodGuardPipeline(PipelineConfig(use_stub=True))  # default to stub in dev

    @app.post("/analyze")
    async def analyze(file: UploadFile) -> Any:  # type: ignore[valid-type]
        contents = await file.read()
        tmp = "tmp_upload.gbff"
        with open(tmp, "wb") as f:
            f.write(contents)
        out: Dict[str, Any] = pipeline.process_genome(tmp)
        return JSONResponse(out)

    return app


if __name__ == "__main__":  # pragma: no cover
    # Lazy local runner: uvicorn api.app:create_app --factory
    import uvicorn

    uvicorn.run("api.app:create_app", host="0.0.0.0", port=8000, factory=True)

