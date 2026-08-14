import math

WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
PALM_REF = 17  # base del meñique, referencia estable para el pulgar

FINGER_JOINTS = [
    (8, 6),    # índice: tip, pip
    (12, 10),  # medio
    (16, 14),  # anular
    (20, 18),  # meñique
]

MARGIN_Y = 0.02
THUMB_MARGIN = 1.1


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_finger_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y - MARGIN_Y


def _is_thumb_extended(landmarks):
    ref = landmarks[PALM_REF]
    return _dist(landmarks[THUMB_TIP], ref) > _dist(landmarks[THUMB_MCP], ref) * THUMB_MARGIN


def count_extended_fingers(landmarks):
    """landmarks: secuencia de 21 puntos con atributos .x/.y normalizados (MediaPipe Hands)."""
    extended = [_is_thumb_extended(landmarks)]
    extended += [_is_finger_extended(landmarks, tip, pip) for tip, pip in FINGER_JOINTS]
    return sum(extended)
