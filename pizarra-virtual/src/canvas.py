import cv2
import numpy as np

DRAW_COLOR = (0, 0, 255)  # rojo (BGR)
LINE_THICKNESS = 4


class DrawingCanvas:
    """Canvas persistente superpuesto al frame de video."""

    def __init__(self, width, height):
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def draw_line(self, start_point, end_point):
        cv2.line(self._canvas, start_point, end_point, DRAW_COLOR, LINE_THICKNESS)

    def clear(self):
        self._canvas[:] = 0

    def compose(self, frame):
        return cv2.add(frame, self._canvas)
