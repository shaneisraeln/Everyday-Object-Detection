---
title: Everyday Object Vision & Voice Assistant
emoji: 🎯
colorFrom: black
colorTo: gray
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# Interactive Visual Object Recognition & Voice-Based Knowledge Assistant

An intelligent, local-first interactive assistant that couples real-time visual object detection (via lightweight YOLO11) with grounded knowledge retrieval and browser voice interaction (Web Speech API push-to-talk speech recognition and speech synthesis).

---

## Key Features

- **Live Vision & Detection**: Real-time camera feed inference using pretrained Ultralytics YOLO11 (`yolo11n.pt`), with bounding box overlays, class tags, and confidence scores.
- **Active Object Context Tracking**: Automatically locks onto the active target object and freezes context during question answering so camera changes do not disrupt answers mid-turn.
- **Multiple Object Disambiguation**: Interactive dropdown selector when multiple everyday objects are in view.
- **Push-to-Talk Voice Interaction**: Integrated browser Web Speech API for hands-free spoken questions, with live transcription and typed input fallback.
- **Text-to-Speech Output**: Speaks answers aloud via browser `speechSynthesis`, with replay, stop, and auto-speak controls.
- **Dual Answer Providers**:
  - **Local Knowledge Base (Default & Offline)**: Deterministic, fact-checked responses grounded in a structured knowledge base without requiring network or cloud credentials.
  - **Amazon Bedrock (Optional)**: Bounded, privacy-preserving LLM generation through AWS Bedrock Runtime Converse API, with automatic fallback to local knowledge if offline or unauthorized.
- **Zero Media Cloud Transmission**: Webcam frames and audio are strictly processed locally; only text labels and user questions are sent to the LLM if Bedrock is enabled.

---

## Architecture Overview

```text
Browser UI (Gradio 6)
  ├── Webcam Stream / Image Upload
  │     └── Frame Sampler (interval throttling & FPS tracking)
  │           └── OpenCV preprocessing
  │                 └── Ultralytics YOLO11n (local PyTorch/CUDA/CPU)
  │                       └── Detection Overlay & Class Mapping
  │
  ├── Voice Input (Browser Web Speech API Push-to-Talk / Typed Fallback)
  │     └── Editable Transcription
  │
  ├── Active Object Context Manager
  │     └── Freezes selected object for the Q&A turn
  │
  └── QuestionAnsweringService
        ├── KnowledgeRepository (src/knowledge/data/objects.json)
        ├── PromptBuilder (strict grounding & limitation rules)
        └── AnswerProvider:
              ├── LocalKnowledgeAnswerProvider (offline, default)
              └── BedrockAnswerProvider (optional AWS Bedrock Runtime Converse API)
```

---

## System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.11 or higher recommended.
- **Hardware**: Standard laptop webcam and microphone. CPU inference is fast with YOLO11n (~30-60ms per frame), and CUDA acceleration is used automatically if an NVIDIA GPU is available.

---

## Quickstart & Setup

### 1. Clone the repository & Navigate
```bash
git clone <repo-url>
cd "Everyday Object Detection"
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:

```bash
# On Windows
copy .env.example .env

# On Linux / macOS
cp .env.example .env
```

The default configuration runs **completely offline in Local Knowledge Base mode**:
```dotenv
APP_ENV=development
DETECTION_MODEL=yolo11n.pt
DETECTION_CONFIDENCE=0.45
DETECTION_INTERVAL_MS=250
MAX_DETECTIONS=10

ANSWER_PROVIDER=local
BEDROCK_REGION=
BEDROCK_MODEL_ID=
BEDROCK_TIMEOUT_SECONDS=20
BEDROCK_MAX_OUTPUT_TOKENS=250

KNOWLEDGE_BASE_PATH=src/knowledge/data/objects.json
LOG_LEVEL=INFO
```

---

## Running the Application

Launch the local web application:

```bash
python app.py
```

Open your browser to:
```
http://127.0.0.1:7860
```

### First Run Note:
On the first execution, if using base weights, Ultralytics will automatically download the weights into the project directory if not present.

---

## Fine-Tuning the Model on Everyday Objects

A turnkey training pipeline (`train.py`) is included to fine-tune YOLO11 on custom everyday household, desk, or personal items.

### 1. Dataset Setup
Prepare your dataset in YOLOv8/YOLO11 format following `dataset_template.yaml`:
```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 2. Start Fine-Tuning
Run the training script (supports CUDA GPU auto-detection):
```bash
python train.py --data dataset/data.yaml --epochs 50 --model yolo11s.pt --batch 16
```

### 3. Activate Fine-Tuned Weights
Once training finishes, the best checkpoint is automatically saved as `best.pt`. Update `.env`:
```dotenv
DETECTION_MODEL=best.pt
```
Restart `python app.py` to use your newly fine-tuned model.

---

## Optional: Enabling Amazon Bedrock

To route question answering through Amazon Bedrock:

1. Ensure your AWS credentials are configured via AWS CLI or environment variables:
   ```bash
   aws configure
   ```
2. Update `.env`:
   ```dotenv
   ANSWER_PROVIDER=bedrock
   BEDROCK_REGION=us-east-1
   BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   ```
   *(Or any available Bedrock model such as Amazon Nova, Meta Llama 3, or Claude 3.5)*
3. If AWS credentials fail, expire, or model access is missing, the application **automatically falls back to the Local Knowledge Base** and alerts you in the UI.

---

## Running Automated Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```

All 25 automated tests verify:
- Active object context management & freezing during Q&A
- Detection contracts, schemas, and overlay rendering
- Knowledge repository loading, validation, and alias resolution
- Grounded question answering intents, care notes, safety, and limitations
- Amazon Bedrock Converse API mocking and graceful offline fallback

---

## Troubleshooting

### Camera Access
- **Browser Permission**: When prompted by your browser, click **Allow** for camera access.
- **Index Conflicts**: If using an external USB webcam, ensure no other video app (Teams, Zoom) is holding exclusive access to the camera device.

### Voice Push-to-Talk
- **Browser Support**: The Web Speech API is supported natively in Google Chrome, Microsoft Edge, and modern Chromium browsers.
- **Typed Fallback**: If using a browser without Web Speech support (e.g. Firefox), type your question in the text box and click **Ask Question**.

### Model & Inference
- **Weights Download**: If network access is restricted behind a proxy, download `yolo11n.pt` manually and place it in the project root.
- **GPU Acceleration**: Torch with CUDA is supported automatically if detected; otherwise inference seamlessly runs on CPU with lightweight YOLO11n.
