# AI vs Real Detection Backend

FastAPI backend for serving the local Bombek1 AI-image detector.

## Model Files

The backend expects the model here:

```text
backend/
  models/
    bombek1/
      model.py
      pytorch_model.pt
```

Those files are already copied into `backend/models/bombek1`.

To rebuild that folder from the existing project cache:

```powershell
py -3.11 tools/export_bombek1_local.py --output-dir backend/models/bombek1
```

## Install

From the project root:

```powershell
py -3.11 -m pip install -r requirements-backend.txt
```

## Run

```powershell
py -3.11 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API

### `GET /health`

Returns API status and model metadata.

### `POST /api/predict`

Accepts a multipart image upload with field name `file`.

Response shape:

```json
{
  "filename": "image.png",
  "label": "AI Generated",
  "raw_label": "ai",
  "upstream_label": "ai",
  "confidence": 0.94,
  "ai_probability": 0.94,
  "real_probability": 0.06,
  "thresholds": {
    "ai_threshold": 0.5,
    "uncertainty_margin": 0.1
  },
  "model": {
    "provider": "local",
    "backend": "bombek1",
    "model_dir": "E:\\my-projects\\AI-Real-Detection\\backend\\models\\bombek1",
    "device": "cuda",
    "loaded": true
  }
}
```

## Next.js Example

```ts
async function detectImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}
```

## Configuration

Copy `.env.example` values into your environment if you want overrides.

Useful variables:

```text
MODEL_DEVICE=auto
BOMBEK1_MODEL_DIR=backend/models/bombek1
LOAD_MODEL_ON_STARTUP=false
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
AI_THRESHOLD=0.50
UNCERTAINTY_MARGIN=0.10
MAX_UPLOAD_MB=15
```
