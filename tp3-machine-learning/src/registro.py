"""
Registro del plano métrico.

El registro es instantáneo: se dispara con la tecla r y congela el
sistema de referencia del mundo sobre el marcador Aruco que haya en ese
momento en escena. De ahí salen las dos homografías que usa todo el
resto del programa:

    H_img_mm   imagen (px) -> mundo (mm)      da el resultado buscado
    H_img_vis  imagen (px) -> vista cenital   sirve para anotar

y la imagen de fondo de la ventana W2D, que es la vista cenital del
plano rectificada en ese instante. Esa rectificación se hace una sola
vez, acá, y no en el bucle de cámara.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Registro:
    """Resultado del registro: el marco de referencia del mundo."""

    H_img_mm: np.ndarray   # imagen -> mm
    H_mm_vis: np.ndarray   # mm -> vista cenital
    H_img_vis: np.ndarray  # imagen -> vista cenital (H_mm_vis @ H_img_mm)
    fondo: np.ndarray      # vista cenital congelada en el momento del registro
    id_referencia: int     # marcador que definió el sistema de referencia
    lado_mm: float         # lado del marcador, la unidad métrica del registro


def esquinas_en_mm(lado_mm):
    """
    Las cuatro esquinas del marcador de referencia expresadas en el
    sistema del mundo: origen en el centro del marcador, x hacia la
    derecha, y hacia arriba, en milímetros.

    El orden es el que devuelve Aruco (superior izquierda, superior
    derecha, inferior derecha, inferior izquierda), así que la
    correspondencia con las esquinas de la imagen es directa.
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
    Homografía (acá apenas una semejanza) de milímetros a píxeles de la
    ventana W2D: el origen del mundo queda en el centro de la imagen, x
    horizontal hacia la derecha e y vertical hacia arriba. El signo
    negativo invierte el eje vertical, que en imagen crece hacia abajo.

    escala está en píxeles por milímetro.
    """
    return np.array([
        [escala, 0.0, ancho / 2.0],
        [0.0, -escala, alto / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def elegir_marcador(ids):
    """
    Con varios marcadores en escena hay que elegir uno como referencia.
    El criterio es el id más chico: cualquiera sirve, pero éste es
    reproducible entre corridas.
    """
    return int(np.argmin(ids))


def registrar_plano(frame, esquinas, ids, lado_mm, ancho_vista, alto_vista,
                    escala):
    """
    Calcula el registro a partir del frame actual.

    Devuelve None si no hay ningún marcador detectado: sin marcador no
    hay sistema de referencia posible y el registro no se lleva a cabo.
    """
    if not ids:
        return None

    indice = elegir_marcador(ids)
    esquinas_img = esquinas[indice].astype(np.float32)
    destino_mm = esquinas_en_mm(lado_mm)

    # Cuatro puntos exactos: getPerspectiveTransform resuelve el sistema
    # sin ajuste por mínimos cuadrados, que acá no aportaría nada.
    H_img_mm = cv2.getPerspectiveTransform(esquinas_img, destino_mm)

    H_mm_vis = matriz_mm_a_vista(ancho_vista, alto_vista, escala)
    H_img_vis = H_mm_vis @ H_img_mm

    fondo = cv2.warpPerspective(frame, H_img_vis, (ancho_vista, alto_vista))

    return Registro(
        H_img_mm=H_img_mm,
        H_mm_vis=H_mm_vis,
        H_img_vis=H_img_vis,
        fondo=fondo,
        id_referencia=int(ids[indice]),
        lado_mm=float(lado_mm),
    )
