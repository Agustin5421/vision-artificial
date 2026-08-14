import os

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")


class HandTracker:
    """Wrapper fino sobre HandLandmarker en modo VIDEO (síncrono) para una sola mano."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. "
                "Corré: bash scripts/download_model.sh"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb, timestamp_ms):
        """Devuelve la lista de 21 landmarks de la mano detectada, o None si no hay mano."""
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def close(self):
        self._landmarker.close()
