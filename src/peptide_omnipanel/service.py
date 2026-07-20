"""Optional FastAPI adapter for the Peptide-OmniPanel inference contract."""

import os

from .contracts import InputValidationError, PeptideInput
from .panel import ENDPOINT_FAMILIES, PanelPredictor


def create_app(panel_predictor: PanelPredictor | None = None):
    """Create the HTTP service without making FastAPI a core dependency."""

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI service requires optional fastapi/pydantic dependencies"
        ) from exc

    if panel_predictor is None and os.environ.get("PEPTIDE_OMNIPANEL_MANIFEST"):
        from .inference import load_model_bundle

        panel_predictor = PanelPredictor(
            load_model_bundle(os.environ["PEPTIDE_OMNIPANEL_MANIFEST"])
        )
    predictor = panel_predictor or PanelPredictor()

    class PredictRequest(BaseModel):
        sequence: str
        format: str = "auto"

    application = FastAPI(
        title="Peptide-OmniPanel",
        version=predictor.model_bundle.version,
    )

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "bundle_version": predictor.model_bundle.version,
            "loaded_model_count": predictor.model_bundle.loaded_model_count,
            "endpoint_families": list(ENDPOINT_FAMILIES),
        }

    @application.post("/predict")
    def predict(request: PredictRequest) -> dict:
        try:
            result = predictor.predict(
                PeptideInput(sequence=request.sequence, format=request.format)
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.to_dict()

    return application


try:
    app = create_app()
except RuntimeError:
    # Core sequence inference remains importable without service dependencies.
    app = None


__all__ = ["app", "create_app"]
