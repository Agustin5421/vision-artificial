"""
Clasificador.

Carga el modelo entrenado (modelos/modelo.joblib) y lo usa para
clasificar en tiempo real los contornos detectados en la imagen
de la webcam.

Verde = objeto reconocido, con su nombre y la confianza de la
predicción. Rojo = contorno detectado que no se pudo clasificar
con confianza suficiente.

Presionar ESC para salir.
"""

import cv2
import numpy as np
from joblib import load

import comun

NOMBRE_VENTANA = "Clasificador"
RUTA_MODELO = "../modelos/modelo.joblib"
UMBRAL_CONFIANZA = 0.6  # por debajo de esto, se marca como desconocido


def main():
    try:
        clasificador = load(RUTA_MODELO)
    except FileNotFoundError:
        print(f"No se encontró el modelo en {RUTA_MODELO}. "
              f"Correr primero entrenador.py.")
        return

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("No se pudo abrir la webcam.")
        return

    comun.dibujar_trackbars_threshold(NOMBRE_VENTANA)

    while True:
        ok, frame = captura.read()
        if not ok:
            print("No se pudo leer el frame de la webcam.")
            break

        umbral = cv2.getTrackbarPos("Umbral", NOMBRE_VENTANA)
        tam_kernel = cv2.getTrackbarPos("Kernel", NOMBRE_VENTANA)

        gris = comun.a_escala_de_grises(frame)
        binaria = comun.aplicar_threshold(gris, umbral)
        binaria = comun.aplicar_morfologia(binaria, tam_kernel)

        contornos = comun.encontrar_contornos(binaria)
        contornos = comun.filtrar_contornos(contornos)

        anotado = frame.copy()

        for contorno in contornos:
            descriptor = comun.calcular_descriptor(contorno).reshape(1, -1)

            etiqueta_predicha = clasificador.predict(descriptor)[0]

            # Confianza: proporción de muestras de la clase ganadora
            # en la hoja del árbol que recibió este descriptor.
            probabilidades = clasificador.predict_proba(descriptor)[0]
            confianza = np.max(probabilidades)

            x, y, w, h = cv2.boundingRect(contorno)

            if confianza >= UMBRAL_CONFIANZA:
                nombre = comun.ETIQUETAS.get(etiqueta_predicha, "?")
                color = (0, 255, 0)  # verde: reconocido
                texto = f"{nombre} ({confianza:.0%})"
            else:
                color = (0, 0, 255)  # rojo: desconocido
                texto = f"desconocido ({confianza:.0%})"

            cv2.drawContours(anotado, [contorno], -1, color, 2)
            cv2.rectangle(anotado, (x, y), (x + w, y + h), color, 1)
            cv2.putText(anotado, texto, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(anotado, "ESC=salir", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(NOMBRE_VENTANA, anotado)
        cv2.imshow("Binaria", binaria)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()