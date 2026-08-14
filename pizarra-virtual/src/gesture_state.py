MODE_BY_COUNT = {0: "PAUSE", 1: "DRAW", 2: "MOVE", 5: "CLEAR"}
DEFAULT_MODE = "PAUSE"


class GestureStabilizer:
    """Convierte un conteo de dedos ruidoso, frame a frame, en un conteo estable
    (requiere N frames consecutivos iguales antes de aceptar el cambio)."""

    def __init__(self, required_frames=5):
        self.required_frames = required_frames
        self.stable_count = 0
        self._candidate = None
        self._streak = 0

    def update(self, raw_count):
        """Registra un conteo crudo de dedos para el frame actual y devuelve el
        conteo estable (solo cambia tras `required_frames` frames consecutivos iguales)."""
        if raw_count == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw_count
            self._streak = 1

        if self._streak >= self.required_frames:
            self.stable_count = self._candidate

        return self.stable_count


def mode_for_count(count, previous_mode=DEFAULT_MODE):
    """Mapea un conteo estable de dedos a un modo. Conteos sin gesto definido (3, 4)
    conservan el modo anterior en lugar de forzar una transición."""
    return MODE_BY_COUNT.get(count, previous_mode)
