"""
Localización: dónde está cada marcador y hacia dónde apunta, en mm.

Usa la homografía imagen -> mm del registro. Pasa las 4 esquinas del
marcador de píxeles a mm y de ahí saca su posición y su ángulo.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Pose:
    """Posición y orientación de un marcador en el mundo"""

    identificador: int
    x_mm: float
    y_mm: float
    angulo_grados: float       # desde el eje x, positivo antihorario
    esquinas_mm: np.ndarray    # las 4 esquinas ya en mm

    @property
    def centro_mm(self):
        return np.array([self.x_mm, self.y_mm], dtype=np.float64)


def transformar_puntos(puntos, homografia):
    """
    Aplica una homografía a una lista de puntos (x, y). Adapta la forma
    del array a la que pide perspectiveTransform y la devuelve igual.
    """
    puntos = np.asarray(puntos, dtype=np.float32).reshape(-1, 1, 2)
    transformados = cv2.perspectiveTransform(puntos, homografia)
    return transformados.reshape(-1, 2).astype(np.float64)


def pose_en_mm(esquinas_img, identificador, H_img_mm):
    """Posición y ángulo de un marcador a partir de sus esquinas en la imagen"""
    # 1. Pasa las 4 esquinas de píxeles a mm
    esquinas_mm = transformar_puntos(esquinas_img, H_img_mm)

    # 2. Posición: el promedio de las 4 esquinas. Se hace en mm y no en
    #    píxeles porque la perspectiva no respeta el punto medio.
    centro = esquinas_mm.mean(axis=0)

    # 3. Ángulo: la dirección del lado izquierdo al derecho del marcador
    #    (esquinas 0 y 3 hacia esquinas 1 y 2), medida desde el eje x.
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
    """La pose de cada marcador detectado en el frame."""
    return [pose_en_mm(puntos, identificador, H_img_mm)
            for puntos, identificador in zip(esquinas, ids)]
