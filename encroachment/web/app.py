"""FastAPI service: upload a before/after GeoTIFF pair, get the risk, area and annotated images.

uvicorn encroachment.web.app:app --port 8000
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import alarm
from ..analysis import Scorer
from ..features import read_pair
from ..imagery import tile_png

STATIC = Path(__file__).parent / "static"
FEDDAN_M2 = 4200.0

app = FastAPI(title="KEMET1 encroachment detector", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
_scorer: Scorer | None = None


def scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "thresholds": {"medium": alarm.MEDIUM_THRESHOLD, "high": alarm.HIGH_THRESHOLD}}


@app.post("/predict")
async def predict(before: UploadFile = File(...), after: UploadFile = File(...)) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, upload in (("before.tif", before), ("after.tif", after)):
            path = Path(tmp) / name
            path.write_bytes(await upload.read())
            paths.append(path)
        try:
            pair = read_pair(*paths)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, f"could not read GeoTIFF: {exc}") from exc
    result = scorer().score(pair, "upload")
    area_m2 = round(result.changed_ha * 10_000, 2)
    return {
        **result.summary(),
        "area_m2": area_m2,
        "area_feddan": round(area_m2 / FEDDAN_M2, 2),
        # names kept for the original web page and clients
        "rf_prob": round(result.probability, 4),
        "fusion_score": round(result.probability, 4),
        "before_img": tile_png(pair.before, "BEFORE", reference=pair.after),
        "after_img": tile_png(pair.after, f"AFTER - {result.risk}", clusters=result.clusters, reference=pair.before),
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
