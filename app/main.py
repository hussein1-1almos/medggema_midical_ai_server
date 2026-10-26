from __future__ import annotations

import io
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydicom
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
MODEL_DIR_RAW = os.getenv("MEDGEMMA_MODEL_DIR", "").strip()
MODEL_DIR = (BASE_DIR / MODEL_DIR_RAW).resolve() if MODEL_DIR_RAW and not Path(MODEL_DIR_RAW).is_absolute() else (Path(MODEL_DIR_RAW).resolve() if MODEL_DIR_RAW else None)
DEVICE = "cpu"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1536"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TORCH_THREADS = max(1, int(os.getenv("TORCH_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))))
ALLOW_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MODEL_LOCK = threading.Lock()
model = None
processor = None
model_load_error: str | None = None


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    kind: Literal["anatomy", "possible_finding", "uncertain"] = "uncertain"
    box: Box
    confidence: float | None = Field(default=None, ge=0, le=1, description="Only populated if model explicitly supplies a numeric score; not calibrated.")
    notes: str | None = None


class LocalizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    modality: str | None
    image_width: int
    image_height: int
    coordinate_system: Literal["original_image_pixels"] = "original_image_pixels"
    findings: list[Finding]
    model_raw_text: str
    limitations: list[str]
    review_required: Literal[True] = True
    experimental: Literal[True] = True


def _load_model() -> None:
    global model, processor, model_load_error
    torch.set_num_threads(TORCH_THREADS)
    source = str(MODEL_DIR) if MODEL_DIR else MODEL_ID
    try:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=False)
        # CPU float32 is the most broadly compatible Windows CPU option, but RAM use is substantial.
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.to(DEVICE)
        model.eval()
        model_load_error = None
    except Exception as exc:
        model = None
        processor = None
        model_load_error = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="MedGemma Localization API (Experimental)",
    version="0.1.0",
    description=(
        "Experimental bounding-box localization using MedGemma 1.5 4B. "
        "Research/testing only; outputs require independent clinical review."
    ),
    lifespan=lifespan,
)


def _dicom_to_image(raw: bytes) -> tuple[Image.Image, str | None]:
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    except Exception as exc:
        raise HTTPException(400, f"File is neither a readable DICOM nor a supported image: {exc}") from exc
    if not hasattr(ds, "PixelData"):
        raise HTTPException(400, "DICOM contains no PixelData.")
    modality = str(getattr(ds, "Modality", "") or "UNKNOWN").upper()
    try:
        arr = ds.pixel_array
        arr = apply_modality_lut(arr, ds)
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                arr = arr[..., :3]
            else:
                raise HTTPException(400, "Multi-frame/volume DICOM is not supported by this single-image endpoint.")
        if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] == 3):
            raise HTTPException(400, f"Unsupported DICOM pixel array shape: {arr.shape}")
        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            finite = np.isfinite(arr)
            if not finite.any():
                raise HTTPException(400, "DICOM pixel data contains no finite values.")
            lo, hi = np.percentile(arr[finite], [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(np.min(arr)), float(np.max(arr))
            if hi <= lo:
                arr = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                arr = 255 - arr
            image = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr).convert("RGB")
        return image, modality
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Unable to decode DICOM pixels: {type(exc).__name__}: {exc}") from exc


def _decode_upload(raw: bytes, filename: str, content_type: str | None) -> tuple[Image.Image, str | None]:
    is_dicom = filename.lower().endswith((".dcm", ".dicom")) or (content_type or "").lower() in {
        "application/dicom", "application/dicom+json"
    }
    if is_dicom:
        return _dicom_to_image(raw)
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError):
        # Some DICOM files have no extension/content type; try DICOM as a fallback.
        try:
            return _dicom_to_image(raw)
        except HTTPException as exc:
            raise HTTPException(400, "Unsupported file. Upload DICOM (.dcm) or PNG/JPEG/WebP.") from exc


def _resize_for_model(image: Image.Image) -> Image.Image:
    image = image.copy()
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return image


def _extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Attempt to locate the first JSON object or array in surrounding prose.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("Model output did not contain parseable JSON.")


def _to_pixel_box(value: Any, width: int, height: int) -> Box | None:
    if isinstance(value, dict):
        keys = {k.lower(): v for k, v in value.items()}
        # Support common key variants.
        aliases = [
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        ]
        coords = None
        for names in aliases:
            if all(n in keys for n in names):
                coords = [keys[n] for n in names]
                break
        if coords is None:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        coords = list(value)
    else:
        return None
    try:
        coords = [float(v) for v in coords]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(coords)):
        return None
    # The prompt requests coordinates normalized to [0, 1000].
    # Also accept [0, 1] normalized coordinates for compatibility with other model outputs.
    if all(0 <= v <= 1 for v in coords):
        coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
    elif all(0 <= v <= 1000 for v in coords):
        coords = [coords[0] * width / 1000, coords[1] * height / 1000,
                  coords[2] * width / 1000, coords[3] * height / 1000]
    x1, y1, x2, y2 = [int(round(v)) for v in coords]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x_min=x1, y_min=y1, x_max=x2, y_max=y2)


def _parse_findings(output: Any, width: int, height: int) -> list[Finding]:
    if isinstance(output, dict):
        candidates = output.get("findings", output.get("detections", output.get("objects", [])))
        if not candidates and any(k in output for k in ("box", "bbox", "bounding_box")):
            candidates = [output]
    elif isinstance(output, list):
        candidates = output
    else:
        candidates = []
    findings: list[Finding] = []
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        box_value = item.get("box", item.get("bbox", item.get("bounding_box", item.get("coordinates"))))
        if box_value is None and all(k in item for k in ("x_min", "y_min", "x_max", "y_max")):
            box_value = item
        box = _to_pixel_box(box_value, width, height)
        if box is None:
            continue
        label = str(item.get("label", item.get("name", item.get("finding", "unspecified")))).strip()[:200]
        kind = str(item.get("kind", "uncertain")).lower()
        if kind not in {"anatomy", "possible_finding", "uncertain"}:
            kind = "uncertain"
        score = item.get("confidence", item.get("score"))
        try:
            score = float(score) if score is not None else None
            if score is not None and not 0 <= score <= 1:
                score = None
        except (TypeError, ValueError):
            score = None
        notes = item.get("notes")
        findings.append(Finding(label=label or "unspecified", kind=kind, box=box, confidence=score,
                                notes=str(notes)[:500] if notes is not None else None))
    return findings


def _run_inference(image: Image.Image, prompt: str) -> str:
    if model is None or processor is None:
        raise HTTPException(
            503,
            "Model is not loaded. Check /health and server console. "
            + (model_load_error or "No model-loading error recorded.")
        )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with MODEL_LOCK, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        new_tokens = generated[0][input_len:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {type(exc).__name__}: {exc}") from exc


@app.get("/")
def root():
    return {
        "service": "MedGemma Localization API",
        "model_id": MODEL_ID,
        "docs": "/docs",
        "health": "/health",
        "warning": "Experimental only. Not for diagnosis or treatment decisions.",
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "model_unavailable",
        "model_id": MODEL_ID,
        "model_loaded": model is not None,
        "device": DEVICE,
        "torch_threads": TORCH_THREADS,
        "model_load_error": model_load_error,
    }


@app.post("/v1/localize", response_model=LocalizationResponse)
async def localize(
    file: UploadFile = File(...),
    target: str = Form(
        default="Identify and localize visible anatomical structures and any clearly visible candidate findings in this image."
    ),
):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds MAX_UPLOAD_BYTES ({MAX_UPLOAD_BYTES} bytes).")
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")
    filename = file.filename or "upload"
    image, modality = _decode_upload(raw, filename, file.content_type)
    original_width, original_height = image.size
    model_image = _resize_for_model(image)
    prompt = f"""
You are an experimental medical-image localization assistant. This is research output only, not a diagnosis.
Task requested by the user: {target}

Analyze the provided single image. Return ONLY valid JSON with this schema:
{{
  "findings": [
    {{
      "label": "short anatomical structure or possible visible finding",
      "kind": "anatomy | possible_finding | uncertain",
      "box": [x_min, y_min, x_max, y_max],
      "notes": "brief visual rationale or uncertainty"
    }}
  ],
  "limitations": ["brief limitations"]
}}

Coordinates MUST be normalized to the image you see, with x and y values between 0 and 1000 inclusive. Use [x_min, y_min, x_max, y_max] where (0,0) is top-left and (1000,1000) is bottom-right. Boxes must enclose the feature, with x_min < x_max and y_min < y_max. Do not invent precise boxes if you cannot localize the feature; return an empty findings list instead. Do not claim a finding is confirmed disease. Include only a small number of useful boxes. Do not include patient identifiers. Do not include a confidence score unless you have a meaningful calibrated score; generally omit it.
"""
    # The model image may have been resized. Map normalized coordinates through model-image size
    # to original-image coordinates by parsing at model-image dimensions, then scaling boxes.
    raw_text = _run_inference(model_image, prompt)
    try:
        parsed = _extract_json(raw_text)
    except ValueError:
        raise HTTPException(502, detail={"message": "Model returned invalid JSON", "raw_output": raw_text[:3000]})
    findings_model = _parse_findings(parsed, model_image.width, model_image.height)
    findings_original = []
    sx, sy = original_width / model_image.width, original_height / model_image.height
    for finding in findings_model:
        b = finding.box
        x1, y1 = round(b.x_min * sx), round(b.y_min * sy)
        x2, y2 = round(b.x_max * sx), round(b.y_max * sy)
        x1, x2 = max(0, min(original_width - 1, x1)), max(0, min(original_width - 1, x2))
        y1, y2 = max(0, min(original_height - 1, y1)), max(0, min(original_height - 1, y2))
        if x2 > x1 and y2 > y1:
            findings_original.append(Finding(
                label=finding.label, kind=finding.kind,
                box=Box(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                confidence=finding.confidence, notes=finding.notes
            ))
    limits = [
        "Experimental model output; not a diagnosis and not for autonomous clinical decisions.",
        "Single 2D rendered image only; this endpoint does not interpret a complete DICOM study or CT/MRI volume.",
        "Bounding boxes and labels may be inaccurate; independent expert review and validation against labeled data are required.",
        "Coordinates are approximate pixel positions on the original decoded image.",
    ]
    if modality:
        limits.append(f"DICOM modality tag: {modality}; modality alone does not establish that this task is supported.")
    if isinstance(parsed, dict) and isinstance(parsed.get("limitations"), list):
        limits.extend(str(x)[:300] for x in parsed["limitations"][:5])
    return LocalizationResponse(
        model=MODEL_ID,
        modality=modality,
        image_width=original_width,
        image_height=original_height,
        findings=findings_original,
        model_raw_text=raw_text[:8000],
        limitations=limits,
    )
