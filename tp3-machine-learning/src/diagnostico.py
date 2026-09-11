"""
Diagnóstico: por qué la cámara no reconoce el marcador.

Prueba todos los diccionarios Aruco que trae OpenCV sobre cada cuadro y
reporta cuáles reconocen lo que le estés mostrando. Dibuja además, en
rojo, los candidatos rechazados: los cuadriláteros que el detector
encontró pero no pudo decodificar.

Cómo leer lo que se ve:

  contorno verde      el marcador se reconoce; el nombre del diccionario
                      que aparece arriba es el que hay que pasarle a
                      main.py con --diccionario
  contorno rojo       se ve el cuadrado pero no se puede leer el código:
                      diccionario equivocado, imagen movida o borrosa,
                      marcador demasiado chico o demasiado oblicuo
  nada                no se distingue ni el cuadrado: falta contraste,
                      hay reflejo sobre el papel, o el marcador quedó sin
                      margen blanco alrededor

Teclas:
    g     guarda el cuadro actual en captura_diagnostico.png, para poder
          mirar con calma qué es lo que está viendo la cámara
    ESC   salir

Ejemplo:
    python src/diagnostico.py --camara 0
"""

import argparse
import sys
from pathlib import Path

import cv2

# Cada cuántos cuadros se barren todos los diccionarios. El barrido es
# caro (son más de veinte detecciones), así que se hace de a ratos y el
# resultado se conserva entre medio.
CADA_CUANTOS_CUADROS = 5

VENTANA = "Diagnostico"
TECLA_ESC = 27
ARCHIVO_CAPTURA = Path("captura_diagnostico.png")


def nombres_de_diccionarios():
    """Todos los diccionarios predefinidos de esta versión de OpenCV."""
    return sorted(nombre for nombre in dir(cv2.aruco)
                  if nombre.startswith("DICT_"))


def crear_detectores():
    """Un detector por diccionario predefinido."""
    detectores = {}
    for nombre in nombres_de_diccionarios():
        diccionario = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, nombre))
        parametros = cv2.aruco.DetectorParameters()
        detectores[nombre] = cv2.aruco.ArucoDetector(diccionario, parametros)
    return detectores


def barrer(frame, detectores):
    """
    Prueba todos los diccionarios. Devuelve:
      - hallazgos: {nombre del diccionario: [ids]} de los que reconocieron algo
      - esquinas, rechazados: del primer diccionario que haya reconocido
        algo, o del primero a secas si ninguno reconoció nada
    """
    hallazgos = {}
    esquinas_buenas = []
    rechazados_muestra = None

    for nombre, detector in detectores.items():
        esquinas, ids, rechazados = detector.detectMarkers(frame)
        if rechazados_muestra is None:
            rechazados_muestra = rechazados
        if ids is not None and len(ids):
            hallazgos[nombre] = [int(i) for i in ids.flatten()]
            if not esquinas_buenas:
                esquinas_buenas = esquinas
                rechazados_muestra = rechazados

    return hallazgos, esquinas_buenas, rechazados_muestra or []


def texto(imagen, cadena, posicion, color, escala=0.5):
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala,
                color, 1, cv2.LINE_AA)


def anotar(frame, hallazgos, esquinas, rechazados):
    anotado = frame.copy()

    for candidato in rechazados:
        contorno = candidato.reshape(-1, 1, 2).astype(int)
        cv2.polylines(anotado, [contorno], True, (0, 0, 255), 2)

    for puntos in esquinas:
        contorno = puntos.reshape(-1, 1, 2).astype(int)
        cv2.polylines(anotado, [contorno], True, (0, 255, 0), 3)

    if hallazgos:
        texto(anotado, "RECONOCIDO por:", (10, 25), (0, 255, 0), 0.6)
        for fila, (nombre, ids) in enumerate(sorted(hallazgos.items())):
            corto = nombre.replace("DICT_", "").lower()
            texto(anotado, f"  {corto}  ids={ids}", (10, 50 + 20 * fila),
                  (0, 255, 0))
    elif len(rechazados):
        texto(anotado, f"Veo {len(rechazados)} cuadrilatero(s) pero no puedo "
                       f"decodificar ninguno", (10, 25), (0, 200, 255), 0.55)
        texto(anotado, "acercar el marcador, mejorar el foco o mirarlo "
                       "mas de frente", (10, 48), (0, 200, 255), 0.5)
    else:
        texto(anotado, "No veo ningun cuadrilatero", (10, 25), (0, 0, 255), 0.6)
        texto(anotado, "revisar luz, reflejos y margen blanco alrededor",
              (10, 48), (0, 0, 255), 0.5)

    return anotado


def main(argumentos=None):
    parser = argparse.ArgumentParser(
        description="Prueba todos los diccionarios Aruco sobre la cámara.")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args(argumentos)

    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}.")
        return 1

    detectores = crear_detectores()
    print(__doc__)
    print(f"Diccionarios a probar: {len(detectores)}")

    hallazgos, esquinas, rechazados = {}, [], []
    cuadro = 0
    ultimo_reporte = None

    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                print("La cámara dejó de entregar cuadros.")
                break

            if cuadro % CADA_CUANTOS_CUADROS == 0:
                hallazgos, esquinas, rechazados = barrer(frame, detectores)

                reporte = tuple(sorted(hallazgos))
                if reporte and reporte != ultimo_reporte:
                    print("Reconocido por:", ", ".join(
                        nombre.replace("DICT_", "").lower()
                        for nombre in reporte))
                    ultimo_reporte = reporte
            cuadro += 1

            cv2.imshow(VENTANA, anotar(frame, hallazgos, esquinas, rechazados))

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == TECLA_ESC:
                break
            if tecla in (ord("g"), ord("G")):
                # El cuadro sin anotar: es el que hay que mirar para
                # entender por qué el detector no encuentra nada.
                cv2.imwrite(str(ARCHIVO_CAPTURA), frame)
                print(f"Cuadro guardado en {ARCHIVO_CAPTURA.resolve()}")
    finally:
        camara.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
