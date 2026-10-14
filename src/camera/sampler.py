"""Frame sampler to throttle inference frequency and track performance metrics."""

import time
from typing import Callable, Any
import numpy as np
from src.schemas import Detection

class FrameSampler:
    """Throttles detection inference based on a time interval and tracks FPS & latency."""

    def __init__(self, interval_ms: int = 250):
        self.interval_sec = interval_ms / 1000.0
        self._last_inference_time = 0.0
        self._last_detections: list[Detection] = []
        self._last_inference_latency_ms: float = 0.0
        self._fps_counter: int = 0
        self._fps_start_time = time.time()
        self._measured_fps: float = 0.0

    @property
    def last_detections(self) -> list[Detection]:
        return self._last_detections

    @property
    def latency_ms(self) -> float:
        return self._last_inference_latency_ms

    @property
    def fps(self) -> float:
        return self._measured_fps

    def should_sample(self, current_time: float | None = None) -> bool:
        """Check if enough time has elapsed to run the next detection."""
        now = current_time if current_time is not None else time.time()
        return (now - self._last_inference_time) >= self.interval_sec

    def process_frame(
        self,
        frame: np.ndarray,
        detect_fn: Callable[[np.ndarray], list[Detection]],
        force: bool = False,
    ) -> tuple[list[Detection], bool]:
        """Run detection if interval elapsed or forced, or return cached detections.
        
        Returns:
            Tuple of (detections, did_run_inference_flag)
        """
        now = time.time()
        # Update FPS
        self._fps_counter += 1
        elapsed = now - self._fps_start_time
        if elapsed >= 1.0:
            self._measured_fps = round(self._fps_counter / elapsed, 1)
            self._fps_counter = 0
            self._fps_start_time = now

        if force or self.should_sample(now):
            start = time.perf_counter()
            self._last_detections = detect_fn(frame)
            self._last_inference_latency_ms = round((time.perf_counter() - start) * 1000, 1)
            self._last_inference_time = now
            return self._last_detections, True

        return self._last_detections, False
