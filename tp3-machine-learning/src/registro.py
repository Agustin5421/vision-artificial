"""
Registro del plano: lo que pasa al apretar r.

Toma el marcador que haya en escena como regla y como origen del mundo,
y calcula una sola vez lo que después usa el loop en cada frame:

    H_img_mm    imagen (px) -> mundo (mm)     para medir
    H_img_vis   imagen (px) -> ventana W2D    para dibujar
    fondo       foto cenital del plano, de fondo para W2D
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Registro:
    """Todo lo que se calcula al apretar r"""

    H_img_mm: np.ndarray   # imagen -> mm
    H_mm_vis: np.ndarray   # mm -> ventana W2D
    H_img_vis: np.ndarray  # imagen -> ventana W2D
    fondo: np.ndarray      # foto cenital tomada al registrar
    id_referencia: int     # marcador que se usó como referencia
    lado_mm: float         # lado real de ese marcador


def esquinas_en_mm(lado_mm):
    """
    Las 4 esquinas del marcador de referencia en mm, con el origen en su
    centro, x hacia la derecha e y hacia arriba. Van en el mismo orden que
    las de Aruco, así cada una se empareja con su esquina en la imagen.
    """
    mitad = lado_mm / 2.0
    return np.array([
        [-mitad,  mitad],
        [ mitad,  mitad],
        [ mitad, -mitad],
        [-mitad, -mitad],
    ], dtype=np.float32)


def matriz_mm_a_vista(ancho, alto, escala):
    """
    Pasa de mm a píxeles de la ventana W2D: el origen queda en el centro y
    el eje y se invierte, porque en una imagen crece hacia abajo.
    escala está en píxeles por mm.
    """
    return np.array([
        [escala, 0.0, ancho / 2.0],
        [0.0, -escala, alto / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def elegir_marcador(ids):
    """Con varios marcadores, usa el de id más chico: así siempre es el mismo."""
    return int(np.argmin(ids))


def registrar_plano(frame, esquinas, ids, lado_mm, ancho_vista, alto_vista,
                    escala):
    """Calcula el registro con el frame actual, o devuelve None si no hay marcadores."""
    # 1. Sin marcador no hay referencia: el registro se cancela
    if not ids:
        return None

    # 2. Elige el marcador de referencia.
    indice = elegir_marcador(ids)

    # 3. Arma 4 parejas de puntos: cada esquina en la imagen (px) y la
    #    misma esquina en el mundo (mm).
    esquinas_img = esquinas[indice].astype(np.float32)
    destino_mm = esquinas_en_mm(lado_mm)

    # 4. Con esas 4 parejas, OpenCV calcula la homografía imagen -> mm.
    H_img_mm = cv2.getPerspectiveTransform(esquinas_img, destino_mm)

    # 5. Homografía imagen -> ventana W2D: primero a mm, después a píxeles de la ventana
    H_mm_vis = matriz_mm_a_vista(ancho_vista, alto_vista, escala)
    H_img_vis = H_mm_vis @ H_img_mm

    # 6. Foto cenital
    fondo = cv2.warpPerspective(frame, H_img_vis, (ancho_vista, alto_vista))

    # 7. Devuelve todo y main.py lo guarda en la variable registro
    return Registro(
        H_img_mm=H_img_mm,
        H_mm_vis=H_mm_vis,
        H_img_vis=H_img_vis,
        fondo=fondo,
        id_referencia=int(ids[indice]),
        lado_mm=float(lado_mm),
    )
