"""
Localización: pose 2D de un marcador en el plano métrico registrado.

Todo el trabajo lo hace la homografía imagen -> mm calculada en el
registro. Acá sólo se transportan las cuatro esquinas del marcador al
mundo y se leen de ahí la posición y la orientación.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Pose:
    """Pose 2D de un marcador en el sistema de referencia del mundo."""

    identificador: int
    x_mm: float
    y_mm: float
    angulo_grados: float       # respecto del eje x, positivo antihorario
    esquinas_mm: np.ndarray    # (4, 2), mismo orden que las de la imagen

    @property
    def centro_mm(self):
        return np.array([self.x_mm, self.y_mm], dtype=np.float64)


def transformar_puntos(puntos, homografia):
    """
    Aplica una homografía a un array (N, 2) y devuelve otro (N, 2).
    Envuelve a perspectiveTransform, que exige la forma (N, 1, 2).
    """
    puntos = np.asarray(puntos, dtype=np.float32).reshape(-1, 1, 2)
    transformados = cv2.perspectiveTransform(puntos, homografia)
    return transformados.reshape(-1, 2).astype(np.float64)


def pose_en_mm(esquinas_img, identificador, H_img_mm):
    """
    Pose del marcador cuyas esquinas en la imagen son esquinas_img.

    La posición es el centro del cuadrado, o sea el promedio de las
    cuatro esquinas ya llevadas a milímetros (el promedio se hace en el
    mundo y no en la imagen: la perspectiva no conserva el punto medio).

    La orientación es la dirección del eje x propio del marcador, que va
    del lado izquierdo al derecho, es decir del punto medio entre las
    esquinas 0 y 3 al punto medio entre las esquinas 1 y 2. Así el
    marcador que se usó para registrar el plano queda, en el instante
    del registro, en la pose (0, 0, 0).
    """
    esquinas_mm = transformar_puntos(esquinas_img, H_img_mm)

    centro = esquinas_mm.mean(axis=0)

    medio_izquierdo = (esquinas_mm[0] + esquinas_mm[3]) / 2.0
    medio_derecho = (esquinas_mm[1] + esquinas_mm[2]) / 2.0
    eje_x = medio_derecho - medio_izquierdo

    angulo = np.degrees(np.arctan2(eje_x[1], eje_x[0]))

    return Pose(
        identificador=int(identificador),
        x_mm=float(centro[0]),
        y_mm=float(centro[1]),
        angulo_grados=float(angulo),
        esquinas_mm=esquinas_mm,
    )


def localizar(esquinas, ids, H_img_mm):
    """Pose de todos los marcadores detectados en el frame."""
    return [pose_en_mm(puntos, identificador, H_img_mm)
            for puntos, identificador in zip(esquinas, ids)]
