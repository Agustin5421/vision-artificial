"""
Diagnóstico: por qué la cámara no reconoce el marcador.

Prueba todos los diccionarios Aruco sobre la cámara y muestra qué ve:

  contorno verde   lo reconoce; arriba dice con qué diccionario
  contorno rojo    ve el cuadrado pero no puede leer el código: marcador
                   movido, borroso, muy chico o muy inclinado
  nada             no ve ni el cuadrado: poca luz, reflejo sobre el papel
                   o falta margen blanco alrededor

Teclas:
    g     guarda el frame actual en captura_diagnostico.png
    ESC   salir

Ejemplo:
    python src/diagnostico.py --camara 0
"""

import argparse
import sys
from pathlib import Path

import cv2

# Probar todos los diccionarios es lento, así que se hace cada 5 frames
# y en el medio se sigue mostrando el último resultado.
CADA_CUANTOS_FRAMES = 5

VENTANA = "Diagnostico"
TECLA_ESC = 27
ARCHIVO_CAPTURA = Path("captura_diagnostico.png")


def nombres_de_diccionarios():
    """Los diccionarios Aruco de esta versión de OpenCV"""
    return sorted(nombre for nombre in dir(cv2.aruco)
                  if nombre.startswith("DICT_") and "APRILTAG" not in nombre)


def crear_detectores():
    """Un detector por cada diccionario"""
    detectores = {}
    for nombre in nombres_de_diccionarios():
        diccionario = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, nombre))
        parametros = cv2.aruco.DetectorParameters()
        detectores[nombre] = cv2.aruco.ArucoDetector(diccionario, parametros)
    return detectores


def barrer(frame, detectores):
    """
    Prueba todos los diccionarios sobre el frame. Devuelve:
      - hallazgos: qué diccionarios reconocieron algo, y qué ids
      - esquinas: los marcadores reconocidos (en verde)
      - rechazados: los cuadrados que vio pero no pudo leer (en rojo)
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
    """Texto con borde negro, para que se lea sobre cualquier fondo"""
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala,
                color, 1, cv2.LINE_AA)


def anotar(frame, hallazgos, esquinas, rechazados):
    """Dibuja los contornos y el mensaje que corresponde a lo que vio"""
    anotado = frame.copy()

    # Rojos: cuadrados que vio pero no pudo leer
    for candidato in rechazados:
        contorno = candidato.reshape(-1, 1, 2).astype(int)
        cv2.polylines(anotado, [contorno], True, (0, 0, 255), 2)

    # Verdes: marcadores reconocidos
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
    # 1. Lee las opciones: solo qué cámara usar
    parser = argparse.ArgumentParser(
        description="Prueba todos los diccionarios Aruco sobre la cámara.")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args(argumentos)

    # 2. Abre la cámara. Si no puede, termina
    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}.")
        return 1

    # 3. Crea un detector por diccionario
    detectores = crear_detectores()
    print(__doc__)
    print(f"Diccionarios a probar: {len(detectores)}")

    hallazgos, esquinas, rechazados = {}, [], []
    numero_frame = 0
    ultimo_reporte = None

    try:
        # Loop: una vuelta por cada frame de la cámara
        while True:
            # 4. Lee un frame
            ok, frame = camara.read()
            if not ok:
                print("La cámara dejó de entregar frames.")
                break

            # 5. Cada 5 frames prueba todos los diccionarios, y avisa en la
            #    terminal si cambió cuáles lo reconocen
            if numero_frame % CADA_CUANTOS_FRAMES == 0:
                hallazgos, esquinas, rechazados = barrer(frame, detectores)

                reporte = tuple(sorted(hallazgos))
                if reporte and reporte != ultimo_reporte:
                    print("Reconocido por:", ", ".join(
                        nombre.replace("DICT_", "").lower()
                        for nombre in reporte))
                    ultimo_reporte = reporte
            numero_frame += 1

            # 6. Muestra el frame con los contornos y el mensaje
            cv2.imshow(VENTANA, anotar(frame, hallazgos, esquinas, rechazados))

            # 7. Teclado: ESC sale; g guarda el frame sin dibujos, para
            #    mirar con calma qué está viendo la cámara
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == TECLA_ESC:
                break
            if tecla in (ord("g"), ord("G")):
                cv2.imwrite(str(ARCHIVO_CAPTURA), frame)
                print(f"Frame guardado en {ARCHIVO_CAPTURA.resolve()}")
    finally:
        # 8. Pase lo que pase, libera la cámara y cierra la ventana
        camara.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
