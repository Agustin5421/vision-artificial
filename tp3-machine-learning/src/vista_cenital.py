"""
Ventana W2D: la mesa vista desde arriba.

El fondo es la foto que se sacó al registrar, no es video en vivo.
Encima se dibujan, en cada frame, la grilla, los ejes y los marcadores.
"""

import cv2
import numpy as np

from localizacion import transformar_puntos

FUENTE = cv2.FONT_HERSHEY_SIMPLEX

COLOR_EJE_X = (60, 60, 255)
COLOR_EJE_Y = (60, 220, 60)
COLOR_CONTORNO = (255, 255, 0)
COLOR_FLECHA = (0, 255, 255)
COLOR_GRILLA = (90, 90, 90)
COLOR_TEXTO = (255, 255, 255)

LARGO_EJES_PX = 70
PASO_GRILLA_MM = 50.0


def texto(imagen, cadena, posicion, color=COLOR_TEXTO, escala=0.5, grosor=1):
    """Texto con borde negro para que se lea sobre cualquier fondo, sin salirse de la ventana"""
    alto_imagen, ancho_imagen = imagen.shape[:2]
    (ancho_texto, alto_texto), _ = cv2.getTextSize(cadena, FUENTE, escala,
                                                   grosor + 2)
    x = int(min(max(posicion[0], 3), max(3, ancho_imagen - ancho_texto - 3)))
    y = int(min(max(posicion[1], alto_texto + 3), alto_imagen - 3))

    cv2.putText(imagen, cadena, (x, y), FUENTE, escala, (0, 0, 0),
                grosor + 2, cv2.LINE_AA)
    cv2.putText(imagen, cadena, (x, y), FUENTE, escala, color,
                grosor, cv2.LINE_AA)


def _a_pixel(punto_mm, H_mm_vis):
    """Pasa un punto de mm a píxeles de la ventana W2D"""
    return tuple(transformar_puntos([punto_mm], H_mm_vis)[0].astype(int))


def dibujar_grilla(lienzo, H_mm_vis, escala):
    """Líneas tenues cada 50 mm, para leer distancias de un vistazo"""
    alto, ancho = lienzo.shape[:2]
    paso_px = PASO_GRILLA_MM * escala
    if paso_px < 8:  # demasiado densa para verse
        return

    origen = _a_pixel((0.0, 0.0), H_mm_vis)
    centro_x, centro_y = float(origen[0]), float(origen[1])

    desplazamiento = paso_px
    while desplazamiento < max(centro_x, centro_y) + paso_px:
        for x in (centro_x - desplazamiento, centro_x + desplazamiento):
            if 0 <= x < ancho:
                cv2.line(lienzo, (int(x), 0), (int(x), alto), COLOR_GRILLA, 1)
        for y in (centro_y - desplazamiento, centro_y + desplazamiento):
            if 0 <= y < alto:
                cv2.line(lienzo, (0, int(y)), (ancho, int(y)), COLOR_GRILLA, 1)
        desplazamiento += paso_px


def dibujar_ejes(lienzo, H_mm_vis):
    """Flechas de los ejes x e y, saliendo del origen"""
    origen = _a_pixel((0.0, 0.0), H_mm_vis)
    fin_x = (origen[0] + LARGO_EJES_PX, origen[1])
    fin_y = (origen[0], origen[1] - LARGO_EJES_PX)

    cv2.arrowedLine(lienzo, origen, fin_x, COLOR_EJE_X, 2, tipLength=0.2)
    cv2.arrowedLine(lienzo, origen, fin_y, COLOR_EJE_Y, 2, tipLength=0.2)
    texto(lienzo, "x", (fin_x[0] + 6, fin_x[1] + 5), COLOR_EJE_X, 0.6, 2)
    texto(lienzo, "y", (fin_y[0] + 6, fin_y[1] - 4), COLOR_EJE_Y, 0.6, 2)
    cv2.circle(lienzo, origen, 4, (255, 255, 255), -1)


def dibujar_pose(lienzo, pose, H_mm_vis, lado_mm):
    """Dibuja un marcador: su contorno, la flecha y el texto con x, y y ángulo"""
    # 1. Contorno: las esquinas en mm, llevadas a la ventana
    contorno = transformar_puntos(pose.esquinas_mm, H_mm_vis)
    cv2.polylines(lienzo, [contorno.astype(np.int32).reshape(-1, 1, 2)],
                  True, COLOR_CONTORNO, 2)

    # 2. Flecha: se arma en mm y después se lleva a la ventana, así su
    #    largo es una medida real y no una cantidad de píxeles cualquiera.
    largo = lado_mm * 0.9
    angulo = np.radians(pose.angulo_grados)
    punta_mm = pose.centro_mm + largo * np.array([np.cos(angulo),
                                                  np.sin(angulo)])

    origen = _a_pixel(pose.centro_mm, H_mm_vis)
    punta = _a_pixel(punta_mm, H_mm_vis)
    cv2.arrowedLine(lienzo, origen, punta, COLOR_FLECHA, 2, tipLength=0.25)
    cv2.circle(lienzo, origen, 4, COLOR_FLECHA, -1)

    # 3. Texto con el id, la posición y el ángulo
    etiqueta = (f"id={pose.identificador}  "
                f"x={pose.x_mm:.1f} mm  y={pose.y_mm:.1f} mm  "
                f"ang={pose.angulo_grados:.1f} deg")
    texto(lienzo, etiqueta, (origen[0] + 10, origen[1] - 10))


def dibujar_w2d(registro, poses):
    """Arma la ventana W2D completa"""
    # 1. Parte de una copia de la foto del registro, para no ensuciar la original
    lienzo = registro.fondo.copy()
    escala = float(registro.H_mm_vis[0, 0])

    # 2. Grilla y ejes
    dibujar_grilla(lienzo, registro.H_mm_vis, escala)
    dibujar_ejes(lienzo, registro.H_mm_vis)

    # 3. Cada marcador detectado
    for pose in poses:
        dibujar_pose(lienzo, pose, registro.H_mm_vis, registro.lado_mm)

    # 4. Encabezado con los datos del registro
    encabezado = (f"referencia: id={registro.id_referencia}  "
                  f"lado={registro.lado_mm:.0f} mm  "
                  f"grilla={PASO_GRILLA_MM:.0f} mm")
    texto(lienzo, encabezado, (10, 22), COLOR_TEXTO, 0.5)

    # 5. Aviso si no hay marcadores a la vista
    if not poses:
        texto(lienzo, "sin marcadores detectados",
              (10, lienzo.shape[0] - 15), (0, 200, 255), 0.55)

    return lienzo


def lienzo_sin_registro(ancho, alto):
    """W2D antes de apretar r: sin homografías solo puede mostrar la instrucción"""
    lienzo = np.full((alto, ancho, 3), 40, dtype=np.uint8)
    texto(lienzo, "Plano no registrado", (30, alto // 2 - 20),
          (255, 255, 255), 0.8, 2)
    texto(lienzo, "Mostrar un marcador Aruco y presionar r",
          (30, alto // 2 + 15), (0, 200, 255), 0.55)
    return lienzo
