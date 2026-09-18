"""
Genera un marcador Aruco en PNG

El tamaño en píxeles se calcula a partir del lado en mm y de los dpi del
dispositivo, para que el marcador mida de verdad lo que después se le
pasa a main.py con --lado-mm

Ej de uso:
    python src/generar_marcador.py --id 0 --lado-mm 100
"""

import argparse
import sys
from pathlib import Path

import cv2

from deteccion import DICCIONARIOS

MM_POR_PULGADA = 25.4


def parsear_argumentos(argumentos=None):
    parser = argparse.ArgumentParser(description="Genera un marcador Aruco.")
    parser.add_argument("--id", type=int, default=0,
                        help="id del marcador dentro del diccionario")
    parser.add_argument("--lado-mm", type=float, default=100.0,
                        help="lado deseado en mm (default: 100)")
    parser.add_argument("--dpi", type=float, default=96.0,
                        help="puntos por pulgada del dispositivo (default: 96)")
    parser.add_argument("--diccionario", default="4x4_50",
                        choices=sorted(DICCIONARIOS))
    parser.add_argument("--borde-mm", type=float, default=10.0,
                        help="margen blanco alrededor, en mm (default: 10)")
    parser.add_argument("--salida", default=None,
                        help="archivo PNG de salida")
    return parser.parse_args(argumentos)


def main(argumentos=None):
    args = parsear_argumentos(argumentos)

    # Los dpi dicen cuántos píxeles entran en una pulgada: con eso se calcula
    # de cuántos píxeles tiene que ser la imagen para medir los mm pedidos.
    lado_px = max(1, int(round(args.lado_mm / MM_POR_PULGADA * args.dpi)))
    borde_px = max(0, int(round(args.borde_mm / MM_POR_PULGADA * args.dpi)))

    diccionario = cv2.aruco.getPredefinedDictionary(
        DICCIONARIOS[args.diccionario])
    if hasattr(cv2.aruco, "generateImageMarker"):
        imagen = cv2.aruco.generateImageMarker(diccionario, args.id, lado_px)
    else:  # OpenCV anterior a 4.7
        imagen = cv2.aruco.drawMarker(diccionario, args.id, lado_px)

    # Sin margen blanco alrededor, el detector no encuentra el marcador
    if borde_px:
        imagen = cv2.copyMakeBorder(imagen, borde_px, borde_px, borde_px,
                                    borde_px, cv2.BORDER_CONSTANT, value=255)

    salida = Path(args.salida) if args.salida else Path(
        f"marcador_{args.diccionario}_{args.id}_{int(args.lado_mm)}mm.png")
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), imagen)

    print(f"Marcador id={args.id} guardado en {salida} "
          f"({lado_px} px de lado a {args.dpi:.0f} dpi = "
          f"{args.lado_mm:.0f} mm).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
