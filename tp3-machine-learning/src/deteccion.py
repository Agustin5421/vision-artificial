"""
Detección de marcadores Aruco.

Envuelve la API de cv2.aruco para aislar al resto del programa de los
cambios de interfaz entre versiones de OpenCV (a partir de 4.7 el
detector es un objeto, antes era una función suelta).
"""

import cv2
import numpy as np

# Diccionarios habilitados desde la línea de comandos. 4X4_50 es el más
# tolerante a cámaras web pobres: pocas celdas, celdas grandes.
DICCIONARIOS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "7x7_250": cv2.aruco.DICT_7X7_250,
}

COLOR_CONTORNO = (0, 255, 0)
COLOR_ETIQUETA = (0, 255, 255)
COLOR_PRIMERA_ESQUINA = (255, 0, 255)


class Detector:
    """Detector de marcadores Aruco de un diccionario fijo."""

    def __init__(self, nombre_diccionario="4x4_50"):
        if nombre_diccionario not in DICCIONARIOS:
            raise ValueError(
                f"Diccionario desconocido: {nombre_diccionario}. "
                f"Opciones: {', '.join(DICCIONARIOS)}")

        self.nombre = nombre_diccionario
        id_diccionario = DICCIONARIOS[nombre_diccionario]

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.diccionario = cv2.aruco.getPredefinedDictionary(id_diccionario)
        else:  # OpenCV viejo
            self.diccionario = cv2.aruco.Dictionary_get(id_diccionario)

        if hasattr(cv2.aruco, "ArucoDetector"):
            parametros = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self.diccionario, parametros)
        else:  # OpenCV viejo: no hay objeto detector
            self._detector = None
            self._parametros = (cv2.aruco.DetectorParameters_create()
                                if hasattr(cv2.aruco, "DetectorParameters_create")
                                else cv2.aruco.DetectorParameters())

    def detectar(self, frame):
        """
        Devuelve (esquinas, ids):

        - esquinas: lista de arrays (4, 2) float32, en píxeles, una por
          marcador. El orden de las esquinas lo garantiza Aruco y es
          siempre el mismo respecto del marcador: 0 superior izquierda,
          1 superior derecha, 2 inferior derecha, 3 inferior izquierda.
          Ese orden es lo que después permite medir la orientación.
        - ids: lista de enteros, alineada con esquinas.
        """
        if self._detector is not None:
            esquinas, ids, _ = self._detector.detectMarkers(frame)
        else:
            esquinas, ids, _ = cv2.aruco.detectMarkers(
                frame, self.diccionario, parameters=self._parametros)

        if ids is None or len(ids) == 0:
            return [], []

        esquinas = [c.reshape(4, 2).astype(np.float32) for c in esquinas]
        ids = [int(i) for i in ids.flatten()]
        return esquinas, ids


def dibujar_detecciones(frame, esquinas, ids):
    """
    Anota sobre una copia del frame el contorno y la etiqueta de cada
    marcador detectado. Marca además la esquina 0 con un círculo, para
    que se vea a simple vista de dónde sale la orientación.
    """
    anotado = frame.copy()

    for puntos, identificador in zip(esquinas, ids):
        contorno = puntos.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(anotado, [contorno], True, COLOR_CONTORNO, 2)
        cv2.circle(anotado, tuple(puntos[0].astype(int)), 5,
                   COLOR_PRIMERA_ESQUINA, -1)

        centro = puntos.mean(axis=0).astype(int)
        cv2.putText(anotado, f"id={identificador}",
                    (centro[0] - 25, centro[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(anotado, f"id={identificador}",
                    (centro[0] - 25, centro[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ETIQUETA, 1, cv2.LINE_AA)

    return anotado
