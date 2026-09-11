"""
Detección de marcadores Aruco.

Envuelve la API de cv2.aruco para aislar al resto del programa de los
cambios de interfaz entre versiones de OpenCV (a partir de 4.7 el
detector es un objeto, antes era una función suelta) y para resolver
solo cuál es el diccionario del marcador que uno tiene a mano, que rara
vez es el que uno supone.
"""

import cv2
import numpy as np

AUTOMATICO = "auto"

COLOR_CONTORNO = (0, 255, 0)
COLOR_ETIQUETA = (0, 255, 255)
COLOR_PRIMERA_ESQUINA = (255, 0, 255)


def _diccionarios_predefinidos():
    """
    Todos los diccionarios que trae esta versión de OpenCV, con nombre
    corto en minúscula: DICT_4X4_50 -> "4x4_50".

    Se arma por introspección y no a mano para no dejar afuera al
    diccionario del marcador que uno se haya bajado de internet, que
    bien puede ser un AprilTag en vez de un Aruco clásico.
    """
    return {nombre.replace("DICT_", "").lower(): getattr(cv2.aruco, nombre)
            for nombre in dir(cv2.aruco) if nombre.startswith("DICT_")}


DICCIONARIOS = _diccionarios_predefinidos()

# Orden en que la búsqueda automática prueba los diccionarios. apriltag_16h5
# va último a propósito: tiene pocos bits de redundancia y es conocido por
# reconocer basura, así que si otro diccionario también da positivo conviene
# quedarse con ese otro.
ORDEN_BUSQUEDA = sorted(DICCIONARIOS, key=lambda n: (n == "apriltag_16h5", n))

# Cuántos cuadros seguidos sin detectar nada aguanta el diccionario ya
# fijado antes de volver a barrer todos. Sin esto, un falso positivo en un
# cuadro de ruido deja al detector fijado en un diccionario equivocado y no
# vuelve a encontrar el marcador nunca más. Con esto se recupera solo, y de
# paso permite cambiar de marcador en marcha.
CUADROS_ANTES_DE_REBUSCAR = 15


class _DetectorDeUnDiccionario:
    """Detector de un único diccionario, con la API nueva o la vieja."""

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
        else:  # OpenCV viejo: no hay objeto detector
            self._detector = None
            self._parametros = (cv2.aruco.DetectorParameters_create()
                                if hasattr(cv2.aruco, "DetectorParameters_create")
                                else cv2.aruco.DetectorParameters())

    def detectar(self, frame):
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

    Con un nombre de diccionario concreto usa ese y nada más. Con "auto"
    prueba todos hasta que alguno reconozca un marcador, y a partir de
    ahí se queda con ese: la búsqueda cuesta más de veinte detecciones
    por cuadro, así que sólo se paga hasta encontrarlo.
    """

    def __init__(self, nombre_diccionario=AUTOMATICO):
        self._automatico = nombre_diccionario == AUTOMATICO
        self._cuadros_sin_detectar = 0

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
        """¿Todavía no sabe con qué diccionario está trabajando?"""
        return self.nombre is None

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
        if self.nombre is None:
            return self._buscar_diccionario(frame)

        esquinas, ids = self._candidatos[self.nombre].detectar(frame)
        if ids:
            self._cuadros_sin_detectar = 0
            return esquinas, ids

        # Nada con el diccionario fijado. Puede ser simplemente que el
        # marcador no esté en escena, pero también que el diccionario esté
        # mal fijado, así que cada tanto se vuelve a barrer.
        self._cuadros_sin_detectar += 1
        if (self._automatico
                and self._cuadros_sin_detectar >= CUADROS_ANTES_DE_REBUSCAR):
            self._cuadros_sin_detectar = 0
            return self._buscar_diccionario(frame)

        return [], []

    def _buscar_diccionario(self, frame):
        """
        Prueba todos los diccionarios y se queda con el que haya
        reconocido más marcadores. Un mismo marcador suele dar positivo
        en varios diccionarios emparentados (4x4_50 está contenido en
        4x4_100, y así), y en ese caso cualquiera de ellos sirve: lo que
        interesa es que la geometría de las esquinas es la misma.
        """
        mejor_nombre, mejor_resultado = None, ([], [])

        for nombre in ORDEN_BUSQUEDA:
            esquinas, ids = self._candidatos[nombre].detectar(frame)
            if len(ids) > len(mejor_resultado[1]):
                mejor_nombre, mejor_resultado = nombre, (esquinas, ids)

        if mejor_nombre is None:
            return [], []

        self.nombre = mejor_nombre
        self._cuadros_sin_detectar = 0
        return mejor_resultado


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
