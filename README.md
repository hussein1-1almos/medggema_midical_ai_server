# MedGemma 1.5 4B Localization API (Windows CPU)

Experimental FastAPI service that accepts one DICOM instance or a PNG/JPEG/WebP image and requests bounding-box localization from Google's `google/medgemma-1.5-4b-it`.

**Research prototype only. Not a medical device. Do not use the output for diagnosis, triage, treatment, or other clinical decisions. Every output requires independent qualified review.**

## Important hardware note

MedGemma 1.5 4B is a multimodal model. On CPU, load time and inference can be slow, and RAM requirements are substantial. Plan for at least 16 GB system RAM as a starting point; 24–32 GB is more comfortable, especially on Windows where other applications consume memory. This is not a guarantee that it will run well on every machine. The code uses CPU float32 for broad compatibility; it does not use your earlier GGUF files or `llama-cpp-python`.

The official model card documents bounding-box anatomical localization primarily for chest X-rays. Do not assume equivalent localization performance for arbitrary findings or all modalities.

## 1. Create environment (Python 3.12)

Open PowerShell in the `server` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch/Transformers dependencies are large. If pip reports a PyTorch wheel issue on Windows, use the official PyTorch selector for the correct Windows/Pip/CPU command: https://pytorch.org/get-started/locally/

## 2. Request model access and download

1. Sign in to Hugging Face.
2. Open https://huggingface.co/google/medgemma-1.5-4b-it
3. Read and accept the applicable Health AI Developer Foundations terms / request access if prompted.
4. In PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
.\download_model.ps1
```

The download is several GB and needs free disk space. Do not download the earlier `medgemma-4b-it-Q4_K_M.gguf` for this project: this app loads the official Transformers checkpoint, not GGUF.

If `hf auth login` fails, install/update the CLI (`python -m pip install -U "huggingface_hub[cli]"`) and retry. Do not put your Hugging Face token in source code or commit it.

## 3. Configure

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

The app loads `.env` from the project root, independent of the current working directory. The simplest option is to set the local model folder directly in `.env`:

```powershell
$env:MEDGEMMA_MODEL_DIR = (Resolve-Path ".\models\medgemma-1.5-4b-it").Path
$env:TORCH_THREADS = "4"
$env:MAX_IMAGE_EDGE = "1536"
$env:MAX_NEW_TOKENS = "512"
```

You can also set these values in the current PowerShell session instead of `.env`.

## 4. Start server

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

First startup loads the model and can take a while. Browse to:

- Health: http://127.0.0.1:8000/health
- Interactive API: http://127.0.0.1:8000/docs

Do not use `--workers 2` or more: each worker would load another model copy and can exhaust RAM. Bind to `127.0.0.1` for local testing; do not expose this prototype to a network without authentication, access controls, privacy review, and security hardening.

## 5. Connect from the separate client machine

This project ZIP contains two independent folders: `server` and `client`. Install the model and run Uvicorn only on the server PC. Copy the `client` folder to the other PC and follow `client/README.md`.

On the server PC, use a trusted LAN only. Find its LAN IPv4 address with `ipconfig`, then start Uvicorn so it listens on the network:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Restrict inbound TCP port 8000 in Windows Firewall to the client PC's IP. From the client PC, test `http://SERVER_LAN_IP:8000/health`, then run `client.py` using `--server "http://SERVER_LAN_IP:8000"`. Do not expose this unauthenticated prototype to the public internet.

## API

`POST /v1/localize` as `multipart/form-data`:
- `file`: one `.dcm`, `.dicom`, `.png`, `.jpg`, `.jpeg`, or `.webp` file
- `target`: optional text instruction describing the localization task

Example response:

```json
{
  "model": "google/medgemma-1.5-4b-it",
  "modality": "DX",
  "image_width": 2048,
  "image_height": 2048,
  "coordinate_system": "original_image_pixels",
  "findings": [
    {
      "label": "left lung",
      "kind": "anatomy",
      "box": {
        "x_min": 100,
        "y_min": 200,
        "x_max": 950,
        "y_max": 1700
      },
      "confidence": null,
      "notes": "Illustrative example only"
    }
  ],
  "model_raw_text": "...",
  "limitations": ["..."],
  "review_required": true,
  "experimental": true
}
```

Coordinates above are illustrative, not actual inference output. This starter asks the model for normalized 0–1000 boxes and converts them to the original decoded image's pixel coordinates. A language model can emit malformed or inaccurate coordinates; syntactic validation does not establish localization accuracy.

## Supported input limitations

- Single 2D image / single-frame DICOM instance only. This is not a complete-study or volumetric CT/MRI viewer.
- DICOM transfer syntaxes may require additional pixel handlers/codecs depending on the file.
- The simple intensity percentile scaling is a generic fallback; it is not a substitute for a modality- and task-appropriate medical image rendering pipeline.
- DICOM metadata and patient identifiers are not intentionally sent as prompt text, but this prototype does not constitute a complete privacy/security implementation.
- The service currently has no authentication, rate limiting, audit log, encrypted transport, or PHI governance layer.

## Visualize boxes

`draw_boxes.py` currently handles PNG/JPEG/WebP source images (not DICOM rendering):

```powershell
python draw_boxes.py "C:\path\to\chest_xray.png" outputs\result.json --output outputs\localized.png
```

## Evaluation before any clinical use

Build a representative, appropriately de-identified, expert-labeled test set. Measure box intersection-over-union (IoU), localization sensitivity at predeclared IoU thresholds, false-positive rate, and results stratified by anatomy, finding, modality, scanner/site, image quality, and patient subgroup. Have qualified clinicians review misses and incorrect boxes. A successful HTTP response or valid JSON is not evidence of clinical performance.

## Official references

- MedGemma 1.5 model card: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
- Model access: https://huggingface.co/google/medgemma-1.5-4b-it
- Google Health repository: https://github.com/Google-Health/medgemma
- PyTorch install selector: https://pytorch.org/get-started/locally/


## Remote server security reminder

Binding to `0.0.0.0` makes the API listen on network interfaces. Only do this on a trusted network, restrict Windows Firewall inbound TCP 8000 to the client machine's IP, and do not expose this unauthenticated prototype to the public internet. Do not send identifiable patient data until privacy, security, and governance controls are implemented.
