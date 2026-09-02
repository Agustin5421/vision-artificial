"""
Módulo común: pipeline de detección de contornos y cálculo de
invariantes de Hu, compartido por el generador de descriptores
y el clasificador.
"""

import cv2
import numpy as np

ETIQUETAS = {
    1: "luna",
    2: "cuadrado",
    3: "estrella",
}

AREA_MINIMA = 500


def a_escala_de_grises(frame):
    """Paso 1: convierte el frame BGR a escala de grises."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def aplicar_threshold(gris, umbral, invertido=True):
    """
    Paso 2: binariza la imagen con un umbral fijo (ajustable
    externamente mediante trackbar).
    invertido=True asume fondo claro y objeto oscuro, o viceversa
    según THRESH_BINARY_INV; ajustar según el ambiente controlado.
    """
    tipo = cv2.THRESH_BINARY_INV if invertido else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(gris, umbral, 255, tipo)
    return binaria


def aplicar_morfologia(binaria, tamano_kernel):
    """
    Paso 3 (opcional): apertura + cierre para eliminar ruido
    y rellenar pequeños huecos. tamano_kernel viene de una
    trackbar; se fuerza a impar y >= 1.
    """
    tamano_kernel = max(1, tamano_kernel)
    if tamano_kernel % 2 == 0:
        tamano_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (tamano_kernel, tamano_kernel))
    abierta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    cerrada = cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, kernel)
    return cerrada


def encontrar_contornos(binaria):
    """Paso 4: obtiene todos los contornos externos de la imagen binaria."""
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def filtrar_contornos(contornos, area_minima=AREA_MINIMA):
    """Paso 5 (opcional): descarta contornos espurios por área mínima."""
    return [c for c in contornos if cv2.contourArea(c) >= area_minima]


def calcular_descriptor(contorno):
    """
    Calcula los 7 invariantes de Hu de un contorno y les aplica
    la transformación logarítmica con signo, para llevarlos a un
    rango numérico manejable sin perder el signo original.

    IMPORTANTE: esta función se usa igual en el generador de
    descriptores y en el clasificador. No modificar una sin
    modificar la otra.
    """
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()  # array de 7 valores

    hu_log = np.zeros_like(hu)
    for i, valor in enumerate(hu):
        if valor == 0:
            hu_log[i] = 0
        else:
            hu_log[i] = -np.sign(valor) * np.log10(np.abs(valor))

    return hu_log


def dibujar_trackbars_threshold(nombre_ventana):
    """Crea las trackbars de threshold y tamaño de kernel morfológico."""
    cv2.namedWindow(nombre_ventana)
    cv2.createTrackbar("Umbral", nombre_ventana, 127, 255, lambda x: None)
    cv2.createTrackbar("Kernel", nombre_ventana, 3, 21, lambda x: None)