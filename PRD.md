# Product Requirements Document (PRD)
## Interactive Visual Object Recognition and Voice-Based Knowledge Assistant

**Document status:** MVP specification  
**Version:** 1.0  
**Project name:** Temporary technical title; product name to be decided later  
**Primary platform:** Local web application running on a laptop  
**Target users:** Student developers, demonstrators, and people who want to learn about objects in their surroundings

---

## 1. Product Summary

Build a software-only assistant that uses a laptop webcam to identify common everyday objects, accepts spoken questions about the currently visible object, and responds with a useful spoken explanation.

Example interaction:

1. The user points the webcam at a water bottle.
2. The application detects a `bottle` and displays a bounding box and confidence score.
3. The user presses a microphone button and asks, “What is this used for?”
4. Speech recognition converts the question to text.
5. The application combines the detected object label with trusted object information and generates an answer.
6. Text-to-speech reads the answer aloud, while the answer is also displayed on screen.

The MVP should prioritize a dependable end-to-end experience over training a custom model immediately.

## 2. Problem Statement

Object detectors can label things in an image, but a class label alone does not explain what an object is for. This project connects real-time visual recognition with question answering and voice interaction so that users can ask natural-language questions about objects in front of a camera.

## 3. Goals

### 3.1 MVP goals

- Display a live webcam feed in a browser-based interface.
- Detect supported everyday objects in near real time.
- Draw bounding boxes, labels, and confidence scores.
- Maintain a clear “currently selected object” for question answering.
- Accept spoken questions using a microphone control.
- Show the recognized speech as text so users can correct or retry it.
- Answer questions about an object’s identity, purpose, common uses, basic components, and ordinary care or safety considerations.
- Speak answers aloud and display the same answer as text.
- Handle no-object, low-confidence, microphone, model-loading, and answer-generation failures gracefully.
- Keep the core vision pipeline local; use Amazon Bedrock only through an optional, explicitly configured answer-generation adapter.
- Provide setup instructions, configuration examples, and automated tests.

### 3.2 Learning and research goals

The implementation should make it possible to learn and evaluate:
- Real-time computer vision inference.
- Transfer learning and object-detection datasets as a later extension.
- Speech-to-text and text-to-speech integration.
- Retrieval-grounded question answering.
- Confidence thresholds and limitations of closed-set recognition.
- Latency, detection performance, and end-to-end reliability.

## 4. Non-Goals for the MVP

Do not implement these unless the core MVP is working and tested:

- Training a large model from scratch.
- Building a custom hardware device.
- Identifying exact brands, product models, or hidden object properties from appearance alone.
- Claiming the detector knows every object.
- Continual learning or automatic addition of new visual classes.
- Medical, legal, financial, or other high-stakes advice.
- Always-on microphone listening or background recording.
- Cloud-hosting the whole application.
- Multi-user accounts, authentication, billing, or a production SaaS platform.
- Full scene understanding or guaranteed recognition of every object in a cluttered scene.

## 5. Target Users and Primary User Story

**Primary user story:**  
As a user, I want to show an everyday object to my webcam, ask a question about it by voice, and hear a concise answer so that I can learn about the object without manually searching for it.

### Supporting user stories

- As a user, I can see which object the system detected and how confident it is.
- As a user, I can ask a follow-up question about the same object.
- As a user, I can read the answer if I cannot hear the audio.
- As a user, I can retry speech recognition if the question was transcribed incorrectly.
- As a user, I receive an honest “I’m not sure” response when the object is not recognized reliably.
- As a developer, I can run the application without AWS credentials by using the local knowledge-base answer mode.

## 6. MVP Functional Requirements

### FR-01: Webcam access
- Request webcam access only when the user starts the camera.
- Display the live feed and a clear camera status.
- Provide start/stop camera controls.
- Explain how to enable browser camera permissions if access fails.

### FR-02: Object detection
- Use a pretrained lightweight object detector as the initial baseline.
- Initial proposal: Ultralytics YOLO11n pretrained on COCO, subject to dependency and licence review.
- Run inference on sampled frames rather than blocking the UI on every camera frame.
- Display bounding boxes, class labels, and confidence scores.
- Allow the user to select a detected object if multiple objects are visible.
- Use a configurable confidence threshold.
- Do not label a low-confidence prediction as certain.

### FR-03: Supported object scope
Start with common classes covered by the pretrained model, such as:
- bottle
- cup
- book
- cell phone
- chair
- backpack
- laptop
- keyboard
- mouse
- remote
- scissors
- clock
- umbrella
- handbag

The exact supported list must be verified against the chosen model’s class vocabulary. The model’s `bottle` class does not guarantee recognition of a specific bottle type, brand, material, or contents.

### FR-04: Active object context
- Maintain the latest stable detection or the user-selected bounding box as the active object.
- Include the object label and confidence in the question-answering request.
- If no suitable object is detected, ask the user to position an object in view before answering object-specific questions.
- Avoid silently switching the active object during a spoken question. Freeze the selected object context for the duration of a question-answering turn.

### FR-05: Voice input
- Provide an explicit push-to-talk / microphone button.
- Convert speech to text.
- Display the transcription before or while processing the answer.
- Provide a retry path for empty or failed transcription.
- Do not record continuously in the background.
- Preferred MVP implementation: browser speech recognition where supported, behind a provider interface. Provide a documented fallback (typed question; optionally a local transcription provider later).
- Clearly indicate that browser speech recognition support and its privacy/network behavior vary by browser.

### FR-06: Answer generation
- Answer questions about the detected object using a structured knowledge source first.
- Use a knowledge record containing, where available: canonical name, short description, purpose, common uses, basic components, care notes, and safety caveats.
- The answer generator must receive the detected object context, the user’s question, and relevant knowledge entries.
- If Amazon Bedrock is configured, use a supported text-generation model for grounded answer generation.
- Do not require Bedrock for basic MVP functionality; the local knowledge-base mode must work without cloud credentials.
- Never ask the language model to invent a precise identity that the detector has not established.
- If the requested information is absent or uncertain, say so rather than fabricate details.
- Keep answers concise by default (approximately 2–5 sentences), unless the user asks for more detail.

### FR-07: Voice output
- Display every generated answer as text.
- Read the answer aloud using browser text-to-speech for the initial MVP.
- Provide a mute/stop-speaking control.
- Handle unavailable speech synthesis gracefully.
- Do not require a cloud TTS service for the MVP.

### FR-08: Interface
The primary screen must include:
- Application title (use the technical title until a product name is chosen).
- Live webcam feed with bounding-box overlay.
- Start/stop camera controls.
- Detected object list or selected-object panel.
- Object label and confidence.
- Push-to-talk control.
- Speech transcription.
- Answer text and speak/stop controls.
- Status/error area for camera, detector, speech, and answer provider.
- Provider status that makes it clear whether answers are local knowledge-based or Bedrock-generated.

### FR-09: Configuration
- Read configuration from environment variables or a local `.env` file.
- Provide `.env.example` with placeholders only.
- Never commit AWS access keys, secret keys, tokens, or other secrets.
- Allow model weights, confidence threshold, inference interval, answer provider, and Bedrock region/model ID to be configured.

### FR-10: Logging and privacy
- Do not persist webcam frames, audio, or transcripts by default.
- Log application errors and timing metrics without storing sensitive media.
- Do not send images to Bedrock for the MVP unless explicitly designed, documented, and enabled in a later feature. The default Bedrock request should be text-only, using the object label and retrieved knowledge.
- Disclose that a configured cloud LLM receives the text question and relevant object context.
- Never claim the system is entirely offline when browser speech recognition or Bedrock is using a network service.

## 7. User Experience and Acceptance Criteria

The MVP is accepted when all of the following can be demonstrated:

1. A user can start the webcam and see a live image.
2. A supported object such as a bottle or cup can be detected in a normal, well-lit scene.
3. The UI displays a bounding box, class label, and confidence score.
4. A user can select or confirm the detected object.
5. The user can ask a question by pressing the microphone button.
6. The transcribed question is visible and recoverable when recognition fails.
7. The application generates an answer tied to the active object, not an unrelated recent detection.
8. The answer appears as text and is spoken aloud when browser TTS is available.
9. With Bedrock credentials absent, local knowledge-base mode still answers common questions.
10. If no object is detected or confidence is below threshold, the system communicates uncertainty instead of inventing an identity.
11. Camera, speech, model, and provider failures produce helpful messages rather than crashing the app.
12. Automated tests cover object-context handling, knowledge retrieval, provider fallback, and error handling.

## 8. Quality Requirements

- **Responsiveness:** avoid blocking the camera display while inference or answer generation runs.
- **Configurability:** thresholds and provider settings must not be hard-coded throughout the codebase.
- **Reliability:** provider errors should not crash the UI.
- **Testability:** vision, speech, knowledge, LLM, and TTS layers should be replaceable and mockable.
- **Accessibility:** controls have readable labels; all spoken answers also appear as text.
- **Privacy:** no media persistence by default.
- **Reproducibility:** include dependency versions or sensible constraints, setup instructions, and a repeatable test command.

## 9. Success Metrics

Measure actual results; do not invent benchmark numbers.

- Webcam startup success rate on the development machine.
- Detection FPS and per-frame inference latency.
- Precision/recall or mAP on a separately labelled evaluation sample if evaluation is performed.
- Speech transcription success rate on a small manually tested set.
- End-to-end time from submitting a question to receiving an answer.
- Fraction of answers supported by the selected knowledge record.
- Number of false or unsupported object claims found during manual review.
- Successful local fallback when Bedrock is unavailable.

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Detector does not support the object | Display unsupported/unknown state; add a custom dataset later |
| Similar-looking objects are confused | Expose confidence and allow user selection/confirmation |
| Browser speech recognition unavailable | Provide typed-question fallback and a replaceable speech provider |
| AWS credentials, permissions, region, or model access fail | Keep local knowledge mode functional; show actionable errors |
| LLM hallucinates object facts | Retrieve known facts, constrain the prompt, and acknowledge missing information |
| Camera inference slows the UI | Sample frames, use a small model, and run inference off the UI path |
| Licensing constraints | Review model/library licences before redistribution or commercial deployment |

## 11. Future Enhancements (Not MVP)

- Fine-tune the detector on a custom everyday-object dataset.
- Add open-set recognition and embedding-based reference-image matching.
- Add a local speech-to-text provider.
- Add optional image-based reasoning through a Bedrock vision-language model after explicit privacy, cost, and model-availability review.
- Add multilingual voice interaction.
- Add user-curated object knowledge.
- Add benchmark datasets, ablation studies, and a formal research report.

## 12. Delivery Expectations for the Coding Agent

The coding agent must:
1. Inspect the existing repository before changing files.
2. Implement the MVP in small, testable phases.
3. Avoid adding unnecessary infrastructure or services.
4. Provide commands for setup, running, and testing.
5. Keep AWS integration optional and use least-privilege credentials.
6. Never claim a feature works unless it has been run or tested.
7. At each milestone, summarize files changed, commands run, test results, known limitations, and exact next steps.
