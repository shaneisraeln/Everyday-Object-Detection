"""Production-grade FastAPI inference API for Everyday Object Detection.

Provides RESTful endpoints for real-time object detection with Pydantic
schema validation, health checks, class metadata, and local Q&A integration.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import time
from typing import List, Optional
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO

from src.config import get_config
from src.knowledge.repository import KnowledgeRepository
from src.qa.service import QuestionAnsweringService
from src.schemas import ObjectContext


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left x coordinate in pixels")
    y1: float = Field(..., description="Top-left y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right x coordinate in pixels")
    y2: float = Field(..., description="Bottom-right y coordinate in pixels")


class DetectionItem(BaseModel):
    class_id: int = Field(..., description="Zero-indexed class identifier")
    class_name: str = Field(..., description="Predicted object category name")
    confidence: float = Field(..., description="Detection confidence score between 0.0 and 1.0")
    box_pixels: BoundingBox = Field(..., description="Absolute bounding box in pixel coordinates")
    box_normalized: List[float] = Field(
        ...,
        description="Normalized coordinates [x1, y1, x2, y2] relative to image dimensions [0, 1]",
    )


class DetectionResponse(BaseModel):
    status: str = "success"
    model_name: str
    inference_time_ms: float
    image_width: int
    image_height: int
    total_detections: int
    detections: List[DetectionItem]


class QuestionRequest(BaseModel):
    object_class: str = Field(..., description="Name of the detected everyday object")
    question: str = Field(..., description="User query about the object")


class QuestionResponse(BaseModel):
    status: str = "success"
    object_class: str
    question: str
    answer: str
    provider: str


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    vram_allocated_mb: Optional[float] = None
    model_loaded: str
    num_classes: int


# ---------------------------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Everyday Object Detection & Voice Assistant API",
    description="Production-grade REST API powered by fine-tuned YOLO11m and YOLO-World.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration and services
config = get_config()
knowledge_repo = KnowledgeRepository(config.knowledge_base_path)
qa_service = QuestionAnsweringService(config, knowledge_repo)

# Determine model weights to load
MODEL_PATH = "best.pt" if __import__("pathlib").Path("best.pt").exists() else "yolo11n.pt"
device_target = config.device if torch.cuda.is_available() else "cpu"

try:
    detector_model = YOLO(MODEL_PATH)
    if device_target != "cpu":
        detector_model.to(device_target)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service metadata."""
    return {
        "service": "Everyday Object Detection REST API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "model": MODEL_PATH,
        "classes_count": len(detector_model.names),
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint reporting GPU and runtime telemetry."""
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    vram_alloc = torch.cuda.memory_allocated(0) / (1024**2) if cuda_ok else 0.0

    return HealthResponse(
        status="healthy",
        device=device_target,
        cuda_available=cuda_ok,
        gpu_name=gpu_name,
        vram_allocated_mb=round(vram_alloc, 2),
        model_loaded=MODEL_PATH,
        num_classes=len(detector_model.names),
    )


@app.get("/classes", tags=["Metadata"])
async def get_supported_classes():
    """Retrieve mapping of supported object class indices to human-readable names."""
    return {
        "total_classes": len(detector_model.names),
        "classes": detector_model.names,
    }


@app.post("/predict", response_model=DetectionResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Target image file (JPEG or PNG)"),
    confidence_threshold: float = Form(
        0.25, description="Confidence threshold filter (0.05 to 0.95)"
    ),
):
    """Run real-time inference on an uploaded image, returning detected everyday objects."""
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Please upload a JPEG, PNG, or WebP image.",
        )

    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}",
        )

    height, width = img.shape[:2]
    t0 = time.perf_counter()

    # Execute inference
    try:
        results = detector_model.predict(
            source=img,
            conf=confidence_threshold,
            device=device_target,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    inference_ms = (time.perf_counter() - t0) * 1000.0

    detections: List[DetectionItem] = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = detector_model.names.get(cls_id, f"class_{cls_id}")
            score = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy

            norm_x1 = max(0.0, min(1.0, x1 / width))
            norm_y1 = max(0.0, min(1.0, y1 / height))
            norm_x2 = max(0.0, min(1.0, x2 / width))
            norm_y2 = max(0.0, min(1.0, y2 / height))

            detections.append(
                DetectionItem(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(score, 4),
                    box_pixels=BoundingBox(
                        x1=round(x1, 1),
                        y1=round(y1, 1),
                        x2=round(x2, 1),
                        y2=round(y2, 1),
                    ),
                    box_normalized=[
                        round(norm_x1, 4),
                        round(norm_y1, 4),
                        round(norm_x2, 4),
                        round(norm_y2, 4),
                    ],
                )
            )

    return DetectionResponse(
        model_name=MODEL_PATH,
        inference_time_ms=round(inference_ms, 2),
        image_width=width,
        image_height=height,
        total_detections=len(detections),
        detections=detections,
    )


@app.post("/qa", response_model=QuestionResponse, tags=["Voice & Knowledge"])
async def ask_object_question(payload: QuestionRequest):
    """Retrieve grounded knowledge answering a user question about a detected object."""
    ctx = ObjectContext(
        label=payload.object_class,
        confidence=0.95,
        bbox_xyxy=(0, 0, 100, 100),
    )
    result = qa_service.answer_question(payload.question, ctx)
    return QuestionResponse(
        object_class=payload.object_class,
        question=payload.question,
        answer=result.answer,
        provider=result.provider,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
