"""
Detección de marcadores Aruco.

1. Encuentra los marcadores en un frame: dónde están sus 4 esquinas y
   qué número tienen. La detección en sí la hace OpenCV.
2. Averigua a qué diccionario pertenecen.
3. Dibuja los marcadores en la ventana Cam.
"""

import cv2
import numpy as np

AUTOMATICO = "auto"

COLOR_CONTORNO = (0, 255, 0)
COLOR_ETIQUETA = (0, 255, 255)
COLOR_PRIMERA_ESQUINA = (255, 0, 255)


def _diccionarios_predefinidos():
    """Los diccionarios Aruco de OpenCV, con nombre corto: DICT_4X4_50 -> "4x4_50"."""
    return {nombre.replace("DICT_", "").lower(): getattr(cv2.aruco, nombre)
            for nombre in dir(cv2.aruco)
            if nombre.startswith("DICT_") and "APRILTAG" not in nombre}


DICCIONARIOS = _diccionarios_predefinidos()

ORDEN_BUSQUEDA = sorted(DICCIONARIOS)

# Frames seguidos sin detectar nada antes de volver a probar todos los
# diccionarios, por si se había fijado uno equivocado.
FRAMES_ANTES_DE_REBUSCAR = 15


class _DetectorDeUnDiccionario:
    """Detector de un solo diccionario."""

    def __init__(self, nombre):
        identificador = DICCIONARIOS[nombre]

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.diccionario = cv2.aruco.getPredefinedDictionary(identificador)
        else:  # OpenCV viejo
            self.diccionario = cv2.aruco.Dictionary_get(identificador)

        if hasattr(cv2.aruco, "ArucoDetector"):
            parametros = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self.diccionario,
                                                     parametros)
            self._parametros = None
        else:  # OpenCV anterior a 4.7: no hay objeto detector
            self._detector = None
            self._parametros = (cv2.aruco.DetectorParameters_create()
                                if hasattr(cv2.aruco, "DetectorParameters_create")
                                else cv2.aruco.DetectorParameters())

    def detectar(self, frame):
        # 1. OpenCV encuentra los marcadores
        if self._detector is not None:
            esquinas, ids, _ = self._detector.detectMarkers(frame)
        else:
            esquinas, ids, _ = cv2.aruco.detectMarkers(
                frame, self.diccionario, parameters=self._parametros)

        if ids is None or len(ids) == 0:
            return [], []

        esquinas = [c.reshape(4, 2).astype(np.float32) for c in esquinas]
        return esquinas, [int(i) for i in ids.flatten()]


class Detector:
    """
    Detector de marcadores Aruco.

    Con un diccionario concreto usa sólo ese, con "auto" prueba todos
    hasta que alguno reconozca un marcador, y desde ahí se queda con ese.
    """

    def __init__(self, nombre_diccionario=AUTOMATICO):
        self._automatico = nombre_diccionario == AUTOMATICO
        self._frames_sin_detectar = 0

        if self._automatico:
            self.nombre = None
            self._candidatos = {nombre: _DetectorDeUnDiccionario(nombre)
                                for nombre in ORDEN_BUSQUEDA}
        elif nombre_diccionario in DICCIONARIOS:
            self.nombre = nombre_diccionario
            self._candidatos = {
                nombre_diccionario: _DetectorDeUnDiccionario(nombre_diccionario)
            }
        else:
            raise ValueError(
                f"Diccionario desconocido: {nombre_diccionario}. "
                f"Opciones: {AUTOMATICO}, {', '.join(sorted(DICCIONARIOS))}")

    @property
    def buscando(self):
        """¿Todavía no encontró el diccionario?"""
        return self.nombre is None

    def detectar(self, frame):
        """
        Devuelve (esquinas, ids), una entrada por marcador.

        Cada esquina es un array (4, 2) en píxeles, siempre en el orden
        superior izquierda, superior derecha, inferior derecha, inferior
        izquierda. De ese orden sale después la orientación.
        """
        # 2. Averigua el diccionario: si todavía no lo sabe, prueba todos.
        if self.nombre is None:
            return self._buscar_diccionario(frame)

        esquinas, ids = self._candidatos[self.nombre].detectar(frame)
        if ids:
            self._frames_sin_detectar = 0
            return esquinas, ids

        # Si no ve nada por un rato, vuelve a probar todos los diccionarios.
        self._frames_sin_detectar += 1
        if (self._automatico
                and self._frames_sin_detectar >= FRAMES_ANTES_DE_REBUSCAR):
            self._frames_sin_detectar = 0
            return self._buscar_diccionario(frame)

        return [], []

    def _buscar_diccionario(self, frame):
        """
        Prueba todos los diccionarios y se queda con el que reconoció más
        marcadores. Si varios reconocen el mismo, cualquiera sirve: las
        esquinas son las mismas.
        """
        mejor_nombre, mejor_resultado = None, ([], [])

        for nombre in ORDEN_BUSQUEDA:
            esquinas, ids = self._candidatos[nombre].detectar(frame)
            if len(ids) > len(mejor_resultado[1]):
                mejor_nombre, mejor_resultado = nombre, (esquinas, ids)

        if mejor_nombre is None:
            return [], []

        self.nombre = mejor_nombre
        self._frames_sin_detectar = 0
        return mejor_resultado


def dibujar_detecciones(frame, esquinas, ids):
    """
    Dibuja sobre una copia del frame el contorno y el id de cada marcador,
    y un punto en la esquina 0, que es de donde sale la orientación.
    """
    # 3. Dibuja cada marcador en la ventana Cam: contorno, esquina 0 e id.
    anotado = frame.copy()

    for puntos, identificador in zip(esquinas, ids):
        contorno = puntos.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(anotado, [contorno], True, COLOR_CONTORNO, 2)
        cv2.circle(anotado, tuple(puntos[0].astype(int)), 5,
                   COLOR_PRIMERA_ESQUINA, -1)

        # El texto va dos veces, negro grueso y color fino encima, para
        # que se lea sobre cualquier fondo.
        centro = puntos.mean(axis=0).astype(int)
        cv2.putText(anotado, f"id={identificador}",
                    (centro[0] - 25, centro[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(anotado, f"id={identificador}",
                    (centro[0] - 25, centro[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ETIQUETA, 1, cv2.LINE_AA)

    return anotado
