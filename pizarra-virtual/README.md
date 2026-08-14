# Pizarra virtual (air drawing) con MediaPipe Hand Landmarker

TP de visión por computadora: pizarra controlada por gestos de una mano. El dedo índice
funciona como lápiz sobre un canvas superpuesto al video de la webcam en tiempo real.

## Gestos

| Gesto | Dedos extendidos | Modo | Efecto |
|---|---|---|---|
| Índice solo | 1 | `DRAW` | Dibuja siguiendo la punta del índice |
| Índice + medio | 2 | `MOVE` | Mueve el puntero sin dibujar |
| Puño cerrado | 0 | `PAUSE` | No dibuja ni mueve |
| Mano abierta | 5 | `CLEAR` | Borra todo el canvas (una vez al entrar al gesto) |
| Otras combinaciones (3-4 dedos) | — | — | No definido, mantiene el modo anterior |

Controles de teclado de respaldo: `q` o `ESC` para salir, `c` para limpiar el canvas.

## Setup

Requiere Python 3.12 (MediaPipe no soporta versiones más nuevas). Si el `python3` global
del sistema es otra versión, instalar 3.12 vía Homebrew: `brew install python@3.12`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_model.sh   # descarga models/hand_landmarker.task (~7.5 MB, no versionado)
```

> Nota: se pinea `mediapipe==0.10.21` a propósito. La versión más nueva (1.0.x) tiene un bug
> conocido en macOS que crashea al cargar `HandLandmarker` (`DrishtiMetalHelper ... Service
> is unavailable`). 0.10.21 es la última verificada sin ese problema en Apple Silicon.

## Correr la app

```bash
source .venv/bin/activate
python -m src.main
```

macOS va a pedir permiso de cámara la primera vez — hay que autorizarlo desde el diálogo
del sistema (o en Ajustes del Sistema > Privacidad y Seguridad > Cámara).

## Verificación sin cámara

Estos chequeos no requieren webcam ni una mano real:

```bash
source .venv/bin/activate
python -c "import cv2, mediapipe; from mediapipe.tasks.python.vision import HandLandmarker; print('OK')"
python -m py_compile src/*.py
pytest tests/ -v
```

Los tests cubren la lógica pura de conteo de dedos (`finger_counter.py`) y de
estabilización/mapeo de modo (`gesture_state.py`) con landmarks sintéticos. El resto
(captura de cámara, detección real, trazo en pantalla) solo se puede validar corriendo
`python -m src.main` y probando cada gesto con la mano frente a la cámara.

## Estructura

```
src/
  main.py            # loop principal: captura, detección, UI, orquesta todo
  hand_tracker.py     # wrapper de HandLandmarker (modo VIDEO, una mano)
  finger_counter.py   # función pura: landmarks -> cantidad de dedos extendidos
  gesture_state.py    # debounce de gestos + mapeo conteo -> modo
  canvas.py           # canvas persistente y su composición sobre el frame
tests/                 # tests de finger_counter.py y gesture_state.py
scripts/download_model.sh
```

## Limitaciones conocidas

- El criterio de dedo extendido (comparar altura de la punta contra la articulación PIP)
  asume la mano razonablemente erguida frente a cámara; con la mano muy rotada o de perfil
  el conteo puede fallar.
- Solo se trackea una mano (`num_hands=1`).
- El overlay del canvas usa `cv2.add`, que puede saturar a blanco donde se cruzan dos trazos
  superpuestos — es un detalle estético, no funcional.
- Iluminación pobre o fondo muy similar al color de piel puede degradar la detección de
  MediaPipe.
