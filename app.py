"""Interactive Visual Object Recognition and Voice-Based Knowledge Assistant.

Minimalist Obsidian Dark Studio UI supporting both YOLO-World open-vocabulary
detection and pretrained YOLO11 fallback with zero emojis.
"""

import logging
import os
import time
from pathlib import Path
import cv2
import numpy as np
import gradio as gr

from src.config import get_config
from src.schemas import Detection, ObjectContext, AnswerResult, ErrorCode, ERROR_MESSAGES
from src.knowledge.repository import KnowledgeRepository
from src.vision.detector import YOLODetector
from src.vision.world_detector import YOLOWorldDetector, DEFAULT_YOLO_WORLD_CLASSES
from src.vision.overlay import render_detections_overlay
from src.context.active_object import ActiveObjectManager
from src.qa.service import QuestionAnsweringService
from src.ui.components import (
    CUSTOM_CSS,
    BROWSER_SPEECH_JS,
    ICON_SCAN,
    ICON_TARGET,
    ICON_SPARK,
    ICON_VOLUME,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# Initialize services
config = get_config()
knowledge_repo = KnowledgeRepository(config.knowledge_base_path)

# Initialize detector backends
DETECTOR_REGISTRY = {}

# 1. Primary: Fine-tuned everyday desk weights
if Path("best.pt").exists():
    try:
        fine_tuned_detector = YOLODetector(
            model_name="best.pt",
            confidence_threshold=config.detection_confidence,
            max_detections=config.max_detections,
            device=config.device,
        )
        DETECTOR_REGISTRY["Fine-Tuned YOLO11m (Everyday Desk)"] = fine_tuned_detector
        logger.info("Registered fine-tuned model checkpoint: best.pt")
    except Exception as e:
        logger.warning("Could not load fine-tuned weights best.pt: %s", e)

# 2. YOLO-World Open-Vocabulary
world_detector = None
try:
    world_detector = YOLOWorldDetector(
        model_path="yolov8s-worldv2.pt",
        classes=DEFAULT_YOLO_WORLD_CLASSES,
        confidence_threshold=config.detection_confidence,
        max_detections=config.max_detections,
        device=config.device,
    )
    DETECTOR_REGISTRY["YOLO-World (Open Vocabulary)"] = world_detector
except Exception as e:
    logger.warning("YOLO-World detector skipped: %s", e)

# 3. YOLO11-Nano Baseline
try:
    yolo_nano_detector = YOLODetector(
        model_name="yolo11n.pt",
        confidence_threshold=config.detection_confidence,
        max_detections=config.max_detections,
        device=config.device,
    )
    DETECTOR_REGISTRY["YOLO11-Nano (Pretrained COCO)"] = yolo_nano_detector
except Exception as e:
    logger.warning("YOLO11-Nano skipped: %s", e)

# Fallback default if nothing loaded
if not DETECTOR_REGISTRY:
    yolo_fallback = YOLODetector(model_name="yolo11n.pt")
    DETECTOR_REGISTRY["YOLO11-Nano (Pretrained COCO)"] = yolo_fallback

if "Fine-Tuned YOLO11m (Everyday Desk)" in DETECTOR_REGISTRY:
    current_model_key = "Fine-Tuned YOLO11m (Everyday Desk)"
else:
    current_model_key = list(DETECTOR_REGISTRY.keys())[0]

active_detector = DETECTOR_REGISTRY[current_model_key]

active_mgr = ActiveObjectManager()
qa_service = QuestionAnsweringService(config, knowledge_repo)

# State cache
current_detections_cache: list[Detection] = []


def downscale_frame(frame: np.ndarray, max_dim: int = 960) -> np.ndarray:
    """Resize high-res webcam frames to max_dim (960px) to balance detail and latency."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def run_detection(
    frame: np.ndarray | None,
    conf_threshold: float | None = 0.25,
) -> tuple[np.ndarray | None, str, gr.Dropdown, str]:
    """Run detection using the active detector backend.
    
    Returns:
        (annotated_image, active_object_text, dropdown_update, status_text)
    """
    global current_detections_cache

    if conf_threshold is None or not isinstance(conf_threshold, (int, float)):
        conf_threshold = 0.25

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return (
            None,
            "No active object",
            gr.Dropdown(choices=[], value=None),
            "Status: Camera ready. Click the circular shutter button to capture a frame, or drag and drop an image.",
        )

    inference_frame = downscale_frame(frame, max_dim=960)

    start_time = time.perf_counter()
    detections = active_detector.detect(inference_frame, confidence=conf_threshold)
    latency_ms = (time.perf_counter() - start_time) * 1000
    current_detections_cache = detections

    is_world = "world" in current_model_key.lower()
    vocab_desc = (
        f"{len(world_detector.get_classes())} prompted classes"
        if is_world
        else "80 COCO classes"
    )

    if not detections:
        status_msg = (
            f"Active: {current_model_key} | Latency: {latency_ms:.1f}ms ({config.device or 'GPU'}) | "
            f"Sensitivity: {conf_threshold:.2f} | 0 objects detected. "
            "Note: Prompt inclusion guides detector attention; recognition depends on lighting, angle, and distance."
        )
        return (
            inference_frame,
            "No objects detected",
            gr.Dropdown(choices=[], value=None),
            status_msg,
        )

    # Update active context
    active_ctx, _ = active_mgr.update_from_detections(detections)

    # Render bounding boxes and reticle
    annotated = render_detections_overlay(
        image=inference_frame,
        detections=detections,
        active_label=active_ctx.label if active_ctx else None,
        suppress_person_when_object_present=True,
    )

    # Prioritize non-person items in dropdown
    sorted_dets = sorted(
        detections,
        key=lambda d: (d.label.lower() in ("person", "tv", "chair", "bed"), -d.confidence),
    )
    choices = [f"[{i + 1}] {d.label} ({d.confidence:.0%})" for i, d in enumerate(sorted_dets)]

    selected_choice = None
    if active_ctx:
        for c in choices:
            if active_ctx.label.lower() in c.lower():
                selected_choice = c
                break

    dropdown_update = gr.Dropdown(choices=choices, value=selected_choice)
    active_str = (
        f"{active_ctx.label.upper()} ({active_ctx.confidence:.0%})"
        if active_ctx
        else "Select an object"
    )

    status_summary = (
        f"Active: {current_model_key} | Latency: {latency_ms:.1f}ms ({config.device or 'GPU'}) | "
        f"Sensitivity: {conf_threshold:.2f} | Detected: {len(detections)} | Vocabulary: {vocab_desc}. "
        "Note: Prompt inclusion guides detector attention; recognition depends on lighting, angle, and distance."
    )

    return annotated, active_str, dropdown_update, status_summary


def on_dropdown_select(selection: str) -> tuple[str, str]:
    """Handle explicit user object selection from dropdown."""
    global current_detections_cache
    if not selection or not current_detections_cache:
        return "None", "No object selected"

    try:
        parts = selection.split(" ")
        if len(parts) >= 2:
            target_label = parts[1].strip().lower()
            matching = [d for d in current_detections_cache if d.label.lower() == target_label]
            if matching:
                best_match = max(matching, key=lambda d: d.confidence)
                ctx = active_mgr.select_explicit(best_match)
                return (
                    f"{ctx.label.upper()} ({ctx.confidence:.0%})",
                    f"Selected {ctx.label}",
                )
    except Exception as e:
        logger.warning(f"Error selecting from dropdown: {e}")

    return "None", "Selection failed"


def on_clear_context():
    """Clear active object context."""
    active_mgr.clear()
    return "None", gr.Dropdown(value=None), "Selection cleared"


def on_model_switch(choice: str) -> tuple[gr.Group, str, str]:
    """Switch active detection model dynamically."""
    global active_detector, current_model_key
    if choice not in DETECTOR_REGISTRY:
        choice = "YOLO-World (Open Vocabulary)"

    current_model_key = choice
    active_detector = DETECTOR_REGISTRY[choice]
    is_world = "world" in choice.lower()

    group_update = gr.Group(visible=is_world)

    if is_world and world_detector:
        count = len(world_detector.get_classes())
        p_status = f"Active vocabulary: {count} prompted everyday classes."
        v_status = (
            f"Active model: {choice} ({config.device or 'GPU'}) | {count} prompted classes active. "
            "Note: Prompt inclusion guides detector attention; it does not guarantee recognition under poor lighting, motion blur, or occlusion."
        )
    elif "fine-tuned" in choice.lower():
        p_status = "Custom 16 everyday classes active (pen, pencil, keys, wallet, charger, earbuds, calculator, eyeglasses, notebook, mug, bottle, phone, laptop, scissors, book, backpack)."
        v_status = f"Active model: {choice} ({config.device or 'GPU'}) | Fine-tuned on custom everyday objects (mAP@50: 83.7%)."
    else:
        p_status = "Open-vocabulary prompts inactive (using closed-set COCO label map)."
        v_status = f"Active model: {choice} ({config.device or 'GPU'}) | Standard 80 COCO classes."

    logger.info(f"Switched model backend to {choice}")
    return group_update, p_status, v_status


def on_update_prompts(prompts_text: str) -> tuple[str, str]:
    """Dynamically update candidate prompts for YOLO-World."""
    if not world_detector:
        return "YOLO-World not active", "YOLO-World not active"
    try:
        updated = world_detector.set_classes(prompts_text)
        count = len(updated)
        msg = f"Active vocabulary updated: {count} classes active ({', '.join(updated[:6])}{'...' if count > 6 else ''})."
        v_status = f"Vocabulary updated ({count} classes). Ready to scan frame."
        return msg, v_status
    except ValueError as e:
        msg = f"Prompt validation error: {e}"
        return msg, msg


def on_reset_prompts() -> tuple[str, str, str]:
    """Reset YOLO-World candidate prompts to defaults."""
    if not world_detector:
        return ", ".join(DEFAULT_YOLO_WORLD_CLASSES), "YOLO-World not active", "YOLO-World not active"
    res = world_detector.reset_classes()
    return (
        world_detector.get_classes_str(),
        f"Active vocabulary reset to {len(res)} default everyday classes.",
        "Vocabulary reset to defaults. Ready to scan frame.",
    )


def on_ask_question(
    question: str,
    auto_speak: bool,
) -> tuple[str, str, str]:
    """Generate grounded answer for active object context."""
    clean_q = question.strip() if question else ""
    if not clean_q:
        return (
            ERROR_MESSAGES[ErrorCode.EMPTY_TRANSCRIPTION],
            "Validator",
            "",
        )

    ctx = active_mgr.freeze_for_qa()
    try:
        res: AnswerResult = qa_service.answer_question(clean_q, ctx)
        tts_action = res.answer if auto_speak else ""
        badge = f"{res.provider}"
        return res.answer, badge, tts_action
    finally:
        active_mgr.unfreeze_qa()


def create_ui() -> gr.Blocks:
    """Build minimalist obsidian dark Gradio UI with SVG icons and zero emojis."""
    with gr.Blocks(title="Everyday Object Vision & Voice Assistant") as demo:
        # Minimalist Workstation Header
        gr.HTML(
            f"""
            <div class="app-header">
                <div class="brand-wrapper">
                    <div class="brand-icon">{ICON_SCAN}</div>
                    <div>
                        <div class="app-title">Everyday Object Vision & Voice Assistant</div>
                    </div>
                </div>
                <div class="app-meta">
                    <span class="meta-pill"><span class="status-indicator"></span><strong>{config.device or 'CUDA:0'}</strong></span>
                    <span class="meta-pill">{qa_service.provider.name}</span>
                </div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # Left: Vision Studio
            with gr.Column(scale=6):
                camera_input = gr.Image(
                    sources=["webcam", "upload"],
                    type="numpy",
                    label="Camera Feed / Image Input",
                    elem_id="camera_input",
                )

                with gr.Row():
                    scan_btn = gr.Button(
                        "Scan Frame",
                        variant="primary",
                        icon="assets/scan.svg",
                        elem_classes=["btn-primary"],
                    )

                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.15,
                        maximum=0.85,
                        value=config.detection_confidence,
                        step=0.05,
                        label="Detection Sensitivity",
                        info="Adjust threshold for faint or handheld objects.",
                        scale=6,
                    )
                    model_selector = gr.Dropdown(
                        choices=list(DETECTOR_REGISTRY.keys()),
                        value=current_model_key,
                        label="Vision Model Engine",
                        info="Switch detector backend.",
                        interactive=True,
                        scale=6,
                    )

                # Open-Vocabulary Prompt Configuration Box (visible when YOLO-World is active)
                with gr.Group(elem_classes=["panel-card"], visible=("world" in current_model_key.lower())) as world_prompt_group:
                    gr.Markdown("#### Open-Vocabulary Candidate Prompts")
                    prompt_input = gr.Textbox(
                        label="Target Objects (Comma-Separated)",
                        value=world_detector.get_classes_str() if world_detector else ", ".join(DEFAULT_YOLO_WORLD_CLASSES),
                        placeholder="pen, pencil, keys, wallet, charger, earbuds, calculator, glasses, notebook, mug, book...",
                        lines=2,
                        interactive=True,
                        info="Enter custom everyday items to detect. Changes apply without restarting the application.",
                    )
                    with gr.Row():
                        update_prompts_btn = gr.Button(
                            "Apply Prompts",
                            size="sm",
                            elem_classes=["btn-secondary"],
                            scale=1,
                        )
                        reset_prompts_btn = gr.Button(
                            "Reset Defaults",
                            size="sm",
                            elem_classes=["btn-secondary"],
                            scale=1,
                        )
                    prompt_status = gr.Markdown(
                        f"Active vocabulary: {len(world_detector.get_classes()) if world_detector else len(DEFAULT_YOLO_WORLD_CLASSES)} prompted everyday classes."
                    )

                annotated_preview = gr.Image(
                    label="Vision HUD & Detections",
                    type="numpy",
                    interactive=False,
                    elem_id="annotated_preview",
                )

                vision_status = gr.Markdown(
                    "Status: Ready. Snap a camera photo, upload an image, or click a Quick Test scene below."
                )

            # Right: Active Target Context & Voice Q&A
            with gr.Column(scale=6):
                # Target Focus Card
                with gr.Group(elem_classes=["panel-card"]):
                    gr.HTML(
                        f"""
                        <div class="panel-header">
                            <div class="panel-title">{ICON_TARGET}<span>Active Target</span></div>
                        </div>
                        """
                    )
                    with gr.Row():
                        active_object_display = gr.Textbox(
                            value="None",
                            label="Focused Object",
                            interactive=False,
                            scale=8,
                        )
                        clear_btn = gr.Button(
                            "Reset",
                            icon="assets/clear.svg",
                            size="sm",
                            elem_classes=["btn-secondary"],
                            scale=2,
                        )

                    object_dropdown = gr.Dropdown(
                        label="Disambiguation Selector",
                        choices=[],
                        value=None,
                        interactive=True,
                        allow_custom_value=True,
                        info="Switch target if multiple everyday objects are in view.",
                    )

                # Voice & Question Query Card
                with gr.Group(elem_classes=["panel-card"]):
                    gr.HTML(
                        f"""
                        <div class="panel-header">
                            <div class="panel-title">{ICON_SPARK}<span>Inquire</span></div>
                        </div>
                        """
                    )
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about usage, care instructions, safety, or purpose...",
                        elem_id="question_input",
                        lines=2,
                    )
                    with gr.Row():
                        ptt_btn = gr.Button(
                            "Voice Input",
                            icon="assets/mic.svg",
                            elem_id="ptt_btn",
                            elem_classes=["btn-secondary"],
                            scale=1,
                        )
                        ask_btn = gr.Button(
                            "Ask Question",
                            icon="assets/send.svg",
                            variant="primary",
                            elem_classes=["btn-primary"],
                            scale=1,
                        )

                    mic_status = gr.HTML(
                        """<div id="mic_status_display" class="status-bar">
                        Microphone ready. Click Voice Input to speak, or type above.
                        </div>"""
                    )

                # Grounded Knowledge Response Card
                with gr.Group(elem_classes=["panel-card"]):
                    gr.HTML(
                        f"""
                        <div class="panel-header">
                            <div class="panel-title">{ICON_VOLUME}<span>Grounded Knowledge Response</span></div>
                        </div>
                        """
                    )
                    answer_display = gr.Textbox(
                        label="Response",
                        elem_id="answer_display",
                        interactive=False,
                        lines=4,
                        elem_classes=["answer-box"],
                    )

                    with gr.Row():
                        speak_btn = gr.Button(
                            "Listen",
                            icon="assets/volume.svg",
                            size="sm",
                            elem_classes=["btn-secondary"],
                            scale=1,
                        )
                        stop_speak_btn = gr.Button(
                            "Stop",
                            icon="assets/stop.svg",
                            size="sm",
                            elem_classes=["btn-secondary"],
                            scale=1,
                        )
                        auto_speak_chk = gr.Checkbox(
                            value=True,
                            label="Auto-read answer",
                            scale=1,
                        )

                    provider_badge = gr.Textbox(visible=False, elem_id="provider_badge")
                    tts_hidden = gr.Textbox(visible=False, elem_id="tts_hidden_output")

        # 1-Click Quick Test Examples
        gr.Examples(
            examples=[
                ["user_scissors_frame.png", 0.30],
                ["sample_images/desk_bottle.png", 0.30],
                ["sample_images/coffee_cup.png", 0.30],
            ],
            inputs=[camera_input, conf_slider],
            outputs=[annotated_preview, active_object_display, object_dropdown, vision_status],
            fn=run_detection,
            label="Quick Test Everyday Object Scenes",
        )

        # Event Wire-up

        # Auto-detect immediately on snapshot capture or file upload
        camera_input.change(
            fn=run_detection,
            inputs=[camera_input, conf_slider],
            outputs=[annotated_preview, active_object_display, object_dropdown, vision_status],
        )

        # Scan Frame Click (auto-triggers snapshot if live webcam video is playing)
        scan_btn.click(
            fn=run_detection,
            inputs=[camera_input, conf_slider],
            outputs=[annotated_preview, active_object_display, object_dropdown, vision_status],
            js="""() => {
                const cam = document.querySelector('#camera_input');
                if (cam && cam.querySelector('video')) {
                    const btns = cam.querySelectorAll('button');
                    for (const b of btns) {
                        if (b.querySelector('svg') || (b.getAttribute('aria-label') && b.getAttribute('aria-label').toLowerCase().includes('capture'))) {
                            b.click();
                            break;
                        }
                    }
                }
            }""",
        )

        # Model switch
        model_selector.change(
            fn=on_model_switch,
            inputs=[model_selector],
            outputs=[world_prompt_group, prompt_status, vision_status],
        )

        # Prompt updates
        update_prompts_btn.click(
            fn=on_update_prompts,
            inputs=[prompt_input],
            outputs=[prompt_status, vision_status],
        )

        prompt_input.submit(
            fn=on_update_prompts,
            inputs=[prompt_input],
            outputs=[prompt_status, vision_status],
        )

        # Reset prompts
        reset_prompts_btn.click(
            fn=on_reset_prompts,
            inputs=[],
            outputs=[prompt_input, prompt_status, vision_status],
        )

        # Dropdown selection
        object_dropdown.input(
            fn=on_dropdown_select,
            inputs=[object_dropdown],
            outputs=[active_object_display, vision_status],
        )

        # Reset selection
        clear_btn.click(
            fn=on_clear_context,
            inputs=[],
            outputs=[active_object_display, object_dropdown, vision_status],
        )

        # Push-to-Talk JS trigger
        ptt_btn.click(
            fn=None,
            js="() => { window.toggleVoiceRecognition(); }",
        )

        # Ask question submission
        ask_btn.click(
            fn=on_ask_question,
            inputs=[question_input, auto_speak_chk],
            outputs=[answer_display, provider_badge, tts_hidden],
        ).then(
            fn=None,
            inputs=[tts_hidden],
            js="(text) => { if (text && text.trim()) { window.speakCurrentAnswer(); } }",
        )

        question_input.submit(
            fn=on_ask_question,
            inputs=[question_input, auto_speak_chk],
            outputs=[answer_display, provider_badge, tts_hidden],
        ).then(
            fn=None,
            inputs=[tts_hidden],
            js="(text) => { if (text && text.trim()) { window.speakCurrentAnswer(); } }",
        )

        # Speech output controls
        speak_btn.click(
            fn=None,
            js="() => { window.speakCurrentAnswer(); }",
        )
        stop_speak_btn.click(
            fn=None,
            js="() => { window.stopSpeaking(); }",
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    server_port = int(os.getenv("PORT", "7860"))
    logger.info(f"Launching Obsidian Dark studio on port {server_port}...")
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        inbrowser=False,
        share=False,
        css=CUSTOM_CSS,
        head=BROWSER_SPEECH_JS,
    )
