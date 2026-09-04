"""
Clasificador.

Carga el modelo entrenado (modelos/modelo.joblib) y lo usa para
clasificar en tiempo real los contornos detectados en la imagen
de la webcam.

Verde = objeto reconocido, con su nombre y la distancia a la muestra
conocida más parecida. Rojo = contorno cuya forma no se parece lo
suficiente a nada conocido.

Esa decisión se toma por distancia y no por la confianza del árbol:
el árbol termina con hojas puras, así que predict_proba() devuelve
1.0 para cualquier entrada, incluso para un objeto que nunca vio.
La distancia sí crece cuando la forma es nueva.

La trackbar "Distancia x10" ajusta el umbral en vivo (el valor de la
barra dividido 10). Subirlo es ser más permisivo, bajarlo es exigir
más parecido.

Presionar ESC para salir.
"""

import cv2
import numpy as np
from joblib import load

import comun

NOMBRE_VENTANA = "Clasificador"
RUTA_MODELO = comun.RAIZ_PROYECTO / "modelos" / "modelo.joblib"

# La trackbar trabaja con enteros; el umbral real es su valor / 10.
ESCALA_TRACKBAR = 10
MAXIMO_TRACKBAR = 200


def distancia_a_lo_conocido(descriptor_norm, muestras, etiquetas, etiqueta):
    """Distancia a la muestra de entrenamiento más parecida de esa clase."""
    grupo = muestras[etiquetas == etiqueta]
    return float(np.linalg.norm(grupo - descriptor_norm, axis=1).min())


def main():
    try:
        modelo = load(RUTA_MODELO)
    except FileNotFoundError:
        print(f"No se encontró el modelo en {RUTA_MODELO}. "
              f"Correr primero entrenador.py.")
        return

    if not isinstance(modelo, dict):
        print(f"El modelo en {RUTA_MODELO} tiene un formato viejo. "
              f"Volver a correr entrenador.py para regenerarlo.")
        return

    clasificador = modelo["clasificador"]
    media = modelo["media"]
    desvio = modelo["desvio"]
    muestras = modelo["muestras"]
    etiquetas = modelo["etiquetas"]

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("No se pudo abrir la webcam.")
        return

    comun.dibujar_trackbars_threshold(NOMBRE_VENTANA)
    cv2.createTrackbar("Distancia x10", NOMBRE_VENTANA,
                       int(modelo["umbral_distancia"] * ESCALA_TRACKBAR),
                       MAXIMO_TRACKBAR, lambda x: None)

    while True:
        ok, frame = captura.read()
        if not ok:
            print("No se pudo leer el frame de la webcam.")
            break

        umbral = cv2.getTrackbarPos("Umbral", NOMBRE_VENTANA)
        tam_kernel = cv2.getTrackbarPos("Kernel", NOMBRE_VENTANA)
        umbral_distancia = (cv2.getTrackbarPos("Distancia x10", NOMBRE_VENTANA)
                            / ESCALA_TRACKBAR)

        gris = comun.a_escala_de_grises(frame)
        binaria = comun.aplicar_threshold(gris, umbral)
        binaria = comun.aplicar_morfologia(binaria, tam_kernel)

        contornos = comun.encontrar_contornos(binaria)
        contornos = comun.filtrar_contornos(contornos, binaria.shape)

        anotado = frame.copy()

        for contorno in contornos:
            descriptor = comun.calcular_descriptor(contorno)
            descriptor_norm = (descriptor - media) / desvio

            etiqueta_predicha = clasificador.predict(descriptor.reshape(1, -1))[0]
            distancia = distancia_a_lo_conocido(descriptor_norm, muestras,
                                                etiquetas, etiqueta_predicha)

            x, y, w, h = cv2.boundingRect(contorno)

            if distancia <= umbral_distancia:
                nombre = comun.ETIQUETAS.get(etiqueta_predicha, "?")
                color = (0, 255, 0)  # verde: reconocido
                texto = f"{nombre} (d={distancia:.2f})"
            else:
                color = (0, 0, 255)  # rojo: desconocido
                texto = f"desconocido (d={distancia:.2f})"

            cv2.drawContours(anotado, [contorno], -1, color, 2)
            cv2.rectangle(anotado, (x, y), (x + w, y + h), color, 1)
            cv2.putText(anotado, texto, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(anotado, f"Umbral distancia: {umbral_distancia:.1f}  ESC=salir",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(NOMBRE_VENTANA, anotado)
        cv2.imshow("Binaria", binaria)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
