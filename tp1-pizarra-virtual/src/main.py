import sys
import time

import cv2

from src.canvas import DrawingCanvas
from src.finger_counter import count_extended_fingers
from src.gesture_state import GestureStabilizer, mode_for_count
from src.hand_tracker import HandTracker

DEBOUNCE_FRAMES = 5
CAMERA_INDEX = 0
INDEX_TIP = 8


def to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            f"Error: no se pudo abrir la cámara (índice {CAMERA_INDEX}). "
            "Probá otro índice o revisá los permisos de cámara en macOS "
            "(Ajustes del Sistema > Privacidad y Seguridad > Cámara)."
        )
        sys.exit(1)

    try:
        tracker = HandTracker()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        cap.release()
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    canvas = DrawingCanvas(width, height)
    stabilizer = GestureStabilizer(required_frames=DEBOUNCE_FRAMES)

    mode = "PAUSE"
    prev_point = None
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Aviso: no se pudo leer un frame de la cámara, reintentando...")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            landmarks = tracker.detect(rgb_frame, timestamp_ms)

            if landmarks is not None:
                raw_count = count_extended_fingers(landmarks)
                stable_count = stabilizer.update(raw_count)
                mode = mode_for_count(stable_count, previous_mode=mode)

                for lm in landmarks:
                    cv2.circle(frame, to_pixel(lm, width, height), 3, (0, 255, 0), -1)

                current_point = to_pixel(landmarks[INDEX_TIP], width, height)

                if mode == "DRAW":
                    if prev_point is not None:
                        canvas.draw_line(prev_point, current_point)
                    prev_point = current_point
                else:
                    prev_point = None

                if mode == "CLEAR":
                    canvas.clear()
            else:
                prev_point = None

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else fps
            prev_time = now

            display = canvas.compose(frame)
            cv2.putText(display, f"Modo: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(
                display,
                "1 dedo=dibujar 2=mover 5=borrar puno=pausa | q=salir c=limpiar",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Pizarra virtual", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' o ESC
                break
            if key == ord("c"):
                canvas.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    run()
