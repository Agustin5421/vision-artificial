"""
Localización homográfica en tiempo real

Una cámara fija mira en perspectiva un plano sobre el que se mueve un
marcador Aruco, y el programa muestra en vivo su pose 2D (posición en mm
y orientación) respecto del plano registrado con r.
Con esc se sale del programa.

Ventanas:
    Cam   video de la cámara con los marcadores detectados
    W2D   vista cenital del plano con la pose de cada marcador

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
                             "aruco_original, ...")
    parser.add_argument("--ancho-w2d", type=int, default=720,
                        help="ancho en px de la ventana W2D (default: 720)")
    parser.add_argument("--alto-w2d", type=int, default=720,
                        help="alto en px de la ventana W2D (default: 720)")
    parser.add_argument("--escala", type=float, default=1.2,
                        help="píxeles por mm en la ventana W2D (default: 1.2)")
    return parser.parse_args(argumentos)


def main(argumentos=None):
    # 1. Lee las opciones de la línea de comandos, como --lado-mm
    args = parsear_argumentos(argumentos)

    # 2. Intenta abrir la cámara
    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}.")
        return 1

    # 3. Crea el detector de marcadores (deteccion.py)
    detector = Detector(args.diccionario)
    ultimo_diccionario = detector.nombre

    # 4. Todavía no hay sistema de referencia, se crea al apretar r
    registro = None

    print(__doc__)

    try:
        # Loop: una vuelta por cada frame de la cámara
        while True:
            # 5. Lee un frame
            ok, frame = camara.read()
            if not ok:
                print("La cámara dejó de entregar frames.")
                break

            # 6. Busca los marcadores: sus cuatro esquinas y su id
            esquinas, ids = detector.detectar(frame)

            # 7. Ventana Cam: el video con los contornos verdes y una ayuda abajo
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

            # 8. Ventana W2D: sólo muestra la instrucción; con
            #    registro calcula la pose de cada marcador (localizacion.py)
            #    y la dibuja en la vista cenital (vista_cenital.py)
            if registro is None:
                w2d = lienzo_sin_registro(args.ancho_w2d, args.alto_w2d)
            else:
                poses = localizar(esquinas, ids, registro.H_img_mm)
                w2d = dibujar_w2d(registro, poses)
            cv2.imshow(VENTANA_W2D, w2d)

            # 9. Teclado: ESC sale; r registra el plano (registro.py) con
                # el frame crudo, para que el fondo de W2D no lleve los dibujos de Cam
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == TECLA_ESC:
                break
            if tecla in (ord("r"), ord("R")):
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
        # 10. Al salir del loop (no importa la razon), libera la cámara y cierra las ventanas
        camara.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
