"""
Generador de descriptores.

Captura la webcam, detecta contornos en tiempo real según el
pipeline de comun.py, y al presionar la barra espaciadora imprime
en la terminal los invariantes de Hu de cada contorno sobreviviente.

Uso:
    - Mostrar el objeto a la cámara, ajustar Umbral/Kernel con las
      trackbars hasta que el contorno se vea limpio en la ventana
      "Binaria".
    - Presionar ESPACIO para imprimir los invariantes de Hu.
    - Copiar esos valores a la planilla/dataset.csv, agregando a
      mano la columna de etiqueta correspondiente.
    - Presionar ESC para salir.
"""

import cv2
import comun

NOMBRE_VENTANA = "Generador de descriptores"


def main():
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
        contornos = comun.filtrar_contornos(contornos, binaria.shape)

        anotado = frame.copy()
        cv2.drawContours(anotado, contornos, -1, (0, 255, 0), 2)
        cv2.putText(anotado, f"Contornos: {len(contornos)}  (ESPACIO=imprimir, ESC=salir)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(NOMBRE_VENTANA, anotado)
        cv2.imshow("Binaria", binaria)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:  # ESC
            break

        elif tecla == 32:  # ESPACIO
            if not contornos:
                print("No hay contornos para imprimir.")
            for i, c in enumerate(contornos):
                descriptor = comun.calcular_descriptor(c)
                valores = ", ".join(f"{v:.8e}" for v in descriptor)
                print(f"[{valores}]")

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()