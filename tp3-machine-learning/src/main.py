"""
Localización homográfica en tiempo real.

Una cámara fija observa en perspectiva un plano (un escritorio, la
pantalla de un monitor) sobre el que se desplaza un marcador Aruco. El
sistema determina y muestra en tiempo real la pose 2D del marcador
(coordenadas en mm y orientación) en el sistema de referencia métrico
registrado.

Ventanas:
    Cam   feed de la cámara con los marcadores detectados anotados
    W2D   vista cenital del mundo 2D con la pose de cada marcador

Teclas:
    r     registro del plano métrico (usa el marcador que haya en escena)
    ESC   salir

Ejemplo:
    python src/main.py --camara 0 --lado-mm 100 --diccionario 4x4_50
"""

import argparse
import sys

import cv2

from deteccion import (AUTOMATICO, DICCIONARIOS, Detector,
                       dibujar_detecciones)
from localizacion import localizar
from registro import registrar_plano
from vista_cenital import dibujar_w2d, lienzo_sin_registro, texto

VENTANA_CAM = "Cam"
VENTANA_W2D = "W2D"
TECLA_ESC = 27


def parsear_argumentos(argumentos=None):
    parser = argparse.ArgumentParser(
        description="Localización 2D de un marcador Aruco por homografía "
                    "de plano a vista.")
    parser.add_argument("--camara", type=int, default=0,
                        help="índice de la cámara (default: 0)")
    parser.add_argument("--lado-mm", type=float, default=100.0,
                        help="lado del marcador Aruco en mm (default: 100)")
    parser.add_argument("--diccionario", default=AUTOMATICO,
                        choices=[AUTOMATICO] + sorted(DICCIONARIOS),
                        metavar="NOMBRE",
                        help="diccionario Aruco (default: auto, lo busca "
                             "solo). Nombres cortos: 4x4_50, 6x6_250, "
                             "apriltag_36h11, aruco_original, ...")
    parser.add_argument("--ancho-w2d", type=int, default=720,
                        help="ancho en px de la ventana W2D (default: 720)")
    parser.add_argument("--alto-w2d", type=int, default=720,
                        help="alto en px de la ventana W2D (default: 720)")
    parser.add_argument("--escala", type=float, default=1.2,
                        help="píxeles por mm en la ventana W2D (default: 1.2)")
    return parser.parse_args(argumentos)


def main(argumentos=None):
    args = parsear_argumentos(argumentos)

    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}.")
        return 1

    detector = Detector(args.diccionario)
    ultimo_diccionario = detector.nombre
    registro = None

    print(__doc__)

    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                print("La cámara dejó de entregar cuadros.")
                break

            esquinas, ids = detector.detectar(frame)

            # El recordatorio va sobre la ventana Cam y no sólo en la
            # consola porque la tecla r la recibe la ventana, no la
            # terminal: hay que tener el foco acá para que funcione.
            cam = dibujar_detecciones(frame, esquinas, ids)
            if detector.nombre != ultimo_diccionario:
                ultimo_diccionario = detector.nombre
                print(f"Diccionario encontrado: {detector.nombre}")

            if not ids and detector.buscando:
                ayuda = "buscando el diccionario del marcador..."
                color = (0, 200, 255)
            elif not ids:
                ayuda = f"{detector.nombre}: sin marcadores detectados"
                color = (0, 200, 255)
            elif registro is None:
                ayuda = (f"{detector.nombre}   "
                         f"r: registrar el plano (con esta ventana activa)")
                color = (0, 255, 0)
            else:
                ayuda = (f"{detector.nombre}   "
                         f"r: volver a registrar   ESC: salir")
                color = (200, 200, 200)
            texto(cam, ayuda, (10, cam.shape[0] - 12), color, 0.55)
            cv2.imshow(VENTANA_CAM, cam)

            # La ventana W2D no se puede actualizar sin homografías; con
            # homografías pero sin marcadores se dibuja igual, porque el
            # fondo y los ejes ya están determinados.
            if registro is None:
                w2d = lienzo_sin_registro(args.ancho_w2d, args.alto_w2d)
            else:
                poses = localizar(esquinas, ids, registro.H_img_mm)
                w2d = dibujar_w2d(registro, poses)
            cv2.imshow(VENTANA_W2D, w2d)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == TECLA_ESC:
                break
            if tecla in (ord("r"), ord("R")):
                # Se registra sobre el frame sin anotar: el fondo cenital
                # no debe llevar encima los dibujos de la ventana Cam.
                nuevo = registrar_plano(frame, esquinas, ids, args.lado_mm,
                                        args.ancho_w2d, args.alto_w2d,
                                        args.escala)
                if nuevo is None:
                    print("Registro cancelado: no hay marcadores en escena.")
                else:
                    registro = nuevo
                    print(f"Plano registrado con el marcador "
                          f"id={registro.id_referencia} "
                          f"({registro.lado_mm:.0f} mm de lado).")
    finally:
        camara.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
