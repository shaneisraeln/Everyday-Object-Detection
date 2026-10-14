# Technical Requirements Document (TRD)
## Interactive Visual Object Recognition and Voice-Based Knowledge Assistant

**Document status:** Implementation specification  
**Version:** 1.0  
**Related document:** `PRD.md`  
**Target:** Local-first laptop application with optional Amazon Bedrock text generation

---

## 1. Technical Overview

Implement a modular Python application with a browser-based UI. The webcam feed is processed locally using OpenCV and a pretrained object detector. The user explicitly activates speech input; the recognized text is combined with the currently selected detection and a structured local knowledge record. An answer provider generates a concise response. Browser text-to-speech reads the answer aloud.

### 1.1 Proposed baseline stack

- Python 3.11 (use a compatible version based on the selected package wheels).
- Gradio for the UI.
- OpenCV for camera/frame handling.
- Ultralytics YOLO11n pretrained weights for baseline object detection, subject to licence review.
- Browser Web Speech API for speech recognition where supported.
- Browser `speechSynthesis` for speech output.
- JSON initially, with SQLite optional if knowledge grows.
- Optional Amazon Bedrock Runtime through the AWS SDK for Python (`boto3`).
- `pytest` for automated tests.
- `python-dotenv` for local configuration.

Do not add a second frontend framework or a separate web server unless a concrete limitation requires it. Keep the first version easy to run locally.

## 2. High-Level Architecture

```text
Browser UI
  ├── Webcam stream
  │     └── Frame sampler
  │           └── OpenCV preprocessing
  │                 └── YOLO detector (local)
  │                       └── Detection results + overlay
  │
  ├── Push-to-talk
  │     └── Speech provider
  │           └── Transcribed question
  │
  ├── Active-object selector
  │     └── Frozen object context for this turn
  │
  └── Answer request
        └── QuestionAnsweringService
              ├── KnowledgeRepository (JSON/SQLite)
              ├── PromptBuilder / grounding rules
              └── AnswerProvider interface
                    ├── LocalKnowledgeAnswerProvider (required)
                    └── BedrockAnswerProvider (optional)
                          └── Amazon Bedrock Runtime
        └── Text answer
              ├── Display in UI
              └── Browser speech synthesis
```

## 3. Component Responsibilities

### 3.1 CameraService

Responsibilities:
- Open and release the webcam safely.
- Return frames in a consistent format.
- Handle camera-unavailable and permission/device errors.
- Avoid opening multiple camera instances.
- Support start/stop lifecycle and clean shutdown.

Implementation guidance:
- Use OpenCV `VideoCapture` in the backend if supported by the selected Gradio version and environment.
- If browser-native webcam capture is more reliable in the chosen Gradio configuration, use it but preserve the same frame interface to the detector.
- Keep camera capture and inference decoupled so a slow detector cannot freeze the UI.

### 3.2 FrameSampler

Responsibilities:
- Limit detection frequency to a configurable rate.
- Avoid unbounded frame queues.
- Prefer the newest frame rather than processing a backlog of stale frames.
- Allow the display to remain responsive when inference is slower than capture.

Suggested configuration:
- `DETECTION_INTERVAL_MS=250` as an initial value, then tune based on measured latency.
- `DETECTION_CONFIDENCE=0.45` as an initial configurable threshold, not a guaranteed optimum.
- `MAX_DETECTIONS=10` per frame as an initial safety/performance limit.

### 3.3 ObjectDetector

Suggested interface:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

class ObjectDetector(Protocol):
    def detect(self, frame) -> list[Detection]:
        ...
```

Requirements:
- Load model weights once during application startup, not once per frame.
- Return normalized detection records independent of the detector library.
- Apply configurable confidence filtering.
- Preserve class names from the model’s actual label map.
- Render bounding boxes and labels in a separate overlay function.
- Do not treat the model’s confidence score as a calibrated probability.
- Handle empty detections without errors.
- Keep detector initialization errors actionable.
- Add unit tests using mocked predictions; add an optional smoke test that loads actual weights when available.

Baseline model:
- YOLO11n pretrained on COCO.
- Use the model's pretrained COCO classes for the initial MVP; do not train a custom detector before a live inference baseline works.
- Verify the exact class list and dependency compatibility during setup.
- Review the applicable Ultralytics licence before redistribution.

### 3.4 ActiveObjectContext

Responsibilities:
- Track the user's selected object for the current question.
- If only one stable detection exists, it may be preselected, but the UI must display that selection.
- If multiple detections exist, provide an explicit selector using label and bounding-box/instance information.
- Freeze the selected detection when a question is submitted; new camera frames must not silently replace it mid-answer.
- If the selected detection disappears, retain the last selected label only for the current interaction/session and show that it is no longer currently visible.
- Reject object-specific questions when there is no valid selection, and ask the user to select or show an object.
- Never silently choose an arbitrary object when several are detected.

Suggested structure:

```python
@dataclass(frozen=True)
class ObjectContext:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int] | None
    knowledge_key: str | None
    observed_at: float
```

Avoid keeping full camera frames in the context unless a later explicitly approved visual-question-answering feature needs them.

### 3.5 KnowledgeRepository

Use a version-controlled JSON file for the initial knowledge base. Keep schema validation and retrieval separate from UI code.

Example schema:

```json
{
  "bottle": {
    "canonical_name": "Bottle",
    "description": "A container designed to hold liquids.",
    "purpose": "Used to store and carry liquids.",
    "common_uses": [
      "Carrying drinking water",
      "Storing other suitable beverages"
    ],
    "components": [
      "Container body",
      "Opening or neck",
      "Cap, when present"
    ],
    "care_notes": [
      "Clean according to the material and manufacturer guidance."
    ],
    "safety_notes": [
      "Do not assume a container is suitable for hot liquids unless it is labelled for that use."
    ],
    "limitations": [
      "The detector identifies the broad bottle category and cannot determine contents or material reliably from the class label alone."
    ]
  }
}
```

Requirements:
- Include records for the supported common classes where meaningful.
- Use cautious, general facts.
- Do not assert properties such as material, contents, brand, or food safety unless verified from available information.
- Validate records at startup and report missing fields.
- Use a canonical key mapping only where explicitly defined; do not rely on fuzzy label guesses without tests.
- Retrieval must return a compact structured record for the selected object.

### 3.6 QuestionAnsweringService

Responsibilities:
- Validate that there is an active object context and non-empty question.
- Retrieve the matching knowledge record.
- Build a bounded, grounded prompt or use a deterministic local response.
- Call the selected answer provider through an interface.
- Return a typed result with answer text, provider name, and optional error/status metadata.
- Apply a request timeout and handle provider failures.
- Never expose credentials, raw stack traces, or internal prompts to the user.

Suggested interface:

```python
@dataclass(frozen=True)
class AnswerResult:
    answer: str
    provider: str
    grounded: bool
    warning: str | None = None

class AnswerProvider(Protocol):
    def answer(
        self,
        question: str,
        object_context: ObjectContext,
        knowledge: dict,
    ) -> AnswerResult:
        ...
```

Local provider (required):
- Must work without network access or AWS credentials.
- Answer common questions using the knowledge record and deterministic templates.
- If the knowledge does not support a requested detail, explicitly say that the information is not available.

Bedrock provider (optional):
- Use `boto3` Bedrock Runtime with credentials supplied through the standard AWS credential chain or an explicitly configured supported profile.
- Use `BEDROCK_REGION` and `BEDROCK_MODEL_ID` configuration.
- Confirm the selected model is available to the account and region and supports the required text input/output API.
- Use a small, concise prompt with explicit grounding instructions.
- Send only the object label, confidence, question, and relevant knowledge text by default. Do not send webcam images or audio in the MVP.
- Set request timeouts and bounded output tokens where supported.
- Handle missing credentials, permission errors, model access errors, throttling, timeouts, and malformed responses.
- Do not automatically retry indefinitely. Use a small bounded retry policy only for retryable transient failures.
- If Bedrock fails, use local knowledge mode and visibly indicate the fallback.
- Do not hard-code a particular model ID without validating current regional availability.
- Document that Bedrock usage may incur charges.

Suggested prompt policy:
- Treat the provided object label as a detector prediction, not unquestionable truth.
- Answer only from supplied knowledge and general low-risk facts.
- Do not infer unseen properties, exact model, brand, contents, or condition.
- Acknowledge when details cannot be determined.
- Keep the answer concise and appropriate for spoken delivery.
- Do not provide hazardous instructions; for safety-sensitive questions, give cautious high-level guidance.

### 3.7 SpeechInputProvider

MVP provider:
- Use browser Web Speech API speech recognition where available.
- Trigger recognition only after the user activates the microphone control.
- Return a plain transcription string or a structured error.
- Display the transcription so it can be checked.
- Provide typed-question fallback if unsupported, denied, or unsuccessful.
- Clearly document that browser implementations vary and may process speech through a browser/vendor service.

Implementation note:
- Browser speech recognition support may be limited in some browsers. If the selected UI framework cannot expose the browser API cleanly, use a small custom Gradio component or a documented local speech provider; do not silently introduce an external speech API.
- Do not store raw audio by default.

Future optional provider:
- A local Whisper/faster-whisper provider may be added after the basic interaction works, but is not required for the MVP.

### 3.8 SpeechOutputProvider

MVP provider:
- Use browser `speechSynthesis` and `SpeechSynthesisUtterance` where available.
- Speak the same answer shown in the UI.
- Provide stop/replay and mute controls.
- Handle unavailable voices or synthesis failures.
- Keep the UI usable without audio.

Do not add Amazon Polly for the MVP unless browser speech synthesis proves insufficient and the user explicitly approves the extra cloud dependency/cost.

## 4. UI Requirements

The main screen should contain:

1. Header with the technical project title.
2. Camera start/stop controls and camera status.
3. Live feed with detection overlay.
4. Detected-object list and active-object selector.
5. Confidence indicator.
6. Push-to-talk button and microphone status.
7. Editable transcription field and submit/retry control.
8. Answer text area.
9. Speak, stop, and replay controls.
10. Answer-provider indicator: `Local knowledge` or `Amazon Bedrock`.
11. Error/status area with human-readable messages.

The interface must remain useful when:
- The camera is unavailable.
- No object is detected.
- Multiple objects are present.
- The microphone is unsupported or denied.
- Speech recognition returns empty text.
- Bedrock is not configured or is unavailable.
- Text-to-speech is unavailable.

## 5. Suggested Repository Structure

```text
project-root/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── PRD.md
├── TRD.md
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── camera/
│   │   ├── service.py
│   │   └── sampler.py
│   ├── vision/
│   │   ├── detector.py
│   │   └── overlay.py
│   ├── context/
│   │   └── active_object.py
│   ├── knowledge/
│   │   ├── repository.py
│   │   └── data/
│   │       └── objects.json
│   ├── qa/
│   │   ├── service.py
│   │   ├── prompt_builder.py
│   │   └── providers/
│   │       ├── base.py
│   │       ├── local_provider.py
│   │       └── bedrock_provider.py
│   ├── speech/
│   │   └── contracts.py
│   └── ui/
│       └── components.py
└── tests/
    ├── test_active_object.py
    ├── test_knowledge_repository.py
    ├── test_question_answering.py
    ├── test_provider_fallback.py
    └── test_detection_contract.py
```

The coding agent may adjust the structure if the selected Gradio/browser integration requires it, but must preserve separation of concerns. Avoid empty abstractions that add complexity without a use case.

## 6. Configuration

Provide `.env.example` with safe defaults and no real credentials:

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

Requirements:
- Validate configuration at startup.
- `ANSWER_PROVIDER=local` must work without AWS configuration.
- If `ANSWER_PROVIDER=bedrock` is selected without required settings, fail gracefully with a clear message or explicitly fall back to local mode.
- Do not commit `.env`.
- Add `.env`, model cache directories, temporary files, and Python caches to `.gitignore` as appropriate.
- Never ask the user to paste AWS secrets into source code or commit them.

## 7. Error Handling

Define clear error categories:
- `CAMERA_UNAVAILABLE`
- `DETECTOR_INITIALIZATION_FAILED`
- `NO_OBJECT_DETECTED`
- `LOW_CONFIDENCE_DETECTION`
- `OBJECT_SELECTION_REQUIRED`
- `SPEECH_UNSUPPORTED`
- `SPEECH_PERMISSION_DENIED`
- `EMPTY_TRANSCRIPTION`
- `KNOWLEDGE_ENTRY_MISSING`
- `ANSWER_PROVIDER_UNAVAILABLE`
- `BEDROCK_ACCESS_DENIED`
- `BEDROCK_THROTTLED`
- `ANSWER_TIMEOUT`
- `TTS_UNAVAILABLE`

Map internal errors to helpful user messages. Log technical details server-side without secrets or raw media.

## 8. Performance and Resource Requirements

- Load detector weights once and reuse the model instance.
- Limit inference frequency and avoid unbounded frame queues.
- Run blocking inference and network calls outside the UI event loop where supported.
- Keep question-answering calls independent of camera-frame processing.
- Set timeouts on external requests.
- Measure real FPS and latency on the target machine; do not promise a fixed FPS in advance.
- If GPU inference is available, allow the detector library to select/configure the device; support CPU fallback.
- Do not require a GPU to launch the application.

## 9. Testing Strategy

### Unit tests
- Empty and valid detection results map to the correct schema.
- Confidence filtering behaves correctly at threshold boundaries.
- Active-object selection is stable for a submitted question.
- No-selection and multiple-selection cases return helpful states.
- Knowledge records load and validate.
- Unknown knowledge keys return a controlled response.
- Local provider answers supported common questions.
- Local provider admits when the requested fact is missing.
- Bedrock provider handles missing config, mocked success, timeout, access denied, throttling, and malformed responses.
- Provider fallback does not expose internal exceptions.
- Speech and TTS interfaces have testable contracts.

### Integration/smoke tests
- App starts with local configuration and no AWS credentials.
- Detector can load weights and process a sample image when weights are available.
- A sample supported object produces a detection record (this is a smoke test, not a full accuracy benchmark).
- Full question flow works with mocked speech input and a mocked provider.
- Optional manual test verifies actual webcam, microphone, and browser TTS behavior.

### Evaluation
- Create a small, separate manually labelled test set with diverse lighting, backgrounds, distances, and object orientations.
- Keep test images separate from any fine-tuning/training images.
- Report the sample size and methodology alongside measured metrics.
- Do not claim a research-grade benchmark from a tiny smoke-test set.

## 10. Setup and Run Expectations

The README must include:
1. Supported Python version and operating system assumptions.
2. Virtual environment setup commands for Windows PowerShell and a generic Unix shell.
3. Dependency installation.
4. Model-weight download behavior and first-run notes.
5. Camera permission instructions.
6. Local-mode run command.
7. Optional AWS configuration using a named AWS profile or standard credential chain.
8. Instructions to select a Bedrock model ID available to the user’s region/account.
9. Test command.
10. Troubleshooting for camera, microphone, model download, GPU/CPU, and AWS access failures.

Do not hard-code user-specific paths.

## 11. Security and Privacy

- No default persistence of images, microphone recordings, or transcripts.
- Do not transmit webcam images to Bedrock in the MVP.
- Keep credentials out of source control and logs.
- Do not render raw exception messages directly to the UI.
- Explicitly label optional cloud use.
- Avoid continuous listening; microphone capture is user-triggered.
- Explain that browser speech recognition may use a browser/vendor service, so the application is not necessarily fully offline when it is enabled.

## 12. Implementation Sequence

### Milestone 1 — App skeleton and camera
- Create project structure, config, README, dependency file, and UI shell.
- Verify camera start/stop and live preview.
- Add basic error handling and tests.

### Milestone 2 — Pretrained detection
- Load YOLO weights once.
- Add frame sampling and detection overlays.
- Show label, confidence, and selected object.
- Verify supported classes from the model label map.

### Milestone 3 — Knowledge-grounded typed Q&A
- Add JSON knowledge records and retrieval.
- Add local deterministic provider.
- Allow typed questions and display answers.
- Test no-object and missing-knowledge cases.

### Milestone 4 — Voice interaction
- Add push-to-talk speech-to-text.
- Show editable transcription.
- Add browser speech synthesis with replay/stop.
- Provide typed-input fallback.

### Milestone 5 — Optional Bedrock integration
- Implement the Bedrock provider behind the same interface as the local provider.
- Verify model availability, permissions, request format, timeout, and cost assumptions.
- Test mocked errors and fallback behavior.
- Keep cloud integration disabled by default.

### Milestone 6 — Evaluation and handoff
- Run automated tests and a manual end-to-end check.
- Record measured FPS and latency.
- Fix major failure modes.
- Update README with actual tested behavior and limitations.
- Provide a final implementation report and exact launch/test commands.

## 13. Definition of Done

The MVP is done when:
- A user can start a webcam and see live detection overlays.
- At least common supported classes can be detected in a manual smoke test.
- A user can choose the relevant object and submit a spoken question.
- Transcription is visible and has a usable fallback.
- The system produces a knowledge-grounded text answer and speaks it when TTS is available.
- Local mode runs without AWS credentials.
- Optional Bedrock mode is configurable and fails safely.
- No webcam image or audio is persisted or sent to Bedrock by default.
- Automated tests pass and test results are reported truthfully.
- The README lets another developer set up and run the project.
- The coding agent documents any unimplemented requirement instead of claiming full completion.
