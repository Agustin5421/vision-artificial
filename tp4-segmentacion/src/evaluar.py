"""
Evalúa el modelo entrenado sobre el conjunto de test (pacientes que nunca vio).

Guarda en resultados/:
- metricas_test.txt         Dice / IoU por corte y por paciente, detección
- predicciones_mejores.png  imagen | máscara real | predicción | superposición
- predicciones_peores.png
- predicciones_azar.png
- dice_por_paciente.png

Uso:
    python src/evaluar.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datos import (RAIZ_TP, dividir_por_paciente, leer_imagen, leer_mascara,
                   listar_cortes, preprocesar)
from metricas import UMBRAL, dice, iou
from unet import UNet

CARPETA_RESULTADOS = RAIZ_TP / "resultados"


def cargar_modelo(dispositivo):
    guardado = torch.load(CARPETA_RESULTADOS / "modelo.pt", map_location=dispositivo)
    modelo = UNet(base=guardado["base"]).to(dispositivo)
    modelo.load_state_dict(guardado["pesos"])
    modelo.eval()
    print(f"Modelo de la época {guardado['epoca']}, entrada {guardado['tamano']} px")
    return modelo, guardado["tamano"]


@torch.no_grad()
def predecir(modelo, cortes, tamano, dispositivo, batch=16):
    """Para cada corte: (imagen original, máscara real, probabilidad predicha)."""
    resultados = []
    for i in range(0, len(cortes), batch):
        grupo = cortes[i:i + batch]
        originales = [(leer_imagen(c.imagen), leer_mascara(c.mascara)) for c in grupo]
        entradas = np.stack([preprocesar(im, m, tamano)[0] for im, m in originales])
        tensor = torch.from_numpy(entradas.transpose(0, 3, 1, 2)).to(dispositivo)
        probabilidades = torch.sigmoid(modelo(tensor))[:, 0].cpu().numpy()
        for (imagen, mascara), probabilidad in zip(originales, probabilidades):
            resultados.append((imagen, mascara, probabilidad))
    return resultados


def superposicion(imagen, real, prediccion):
    """FLAIR en gris; verde = acierto, rojo = falso positivo, azul = falso negativo."""
    gris = imagen[..., 1].astype(np.float32) / 255
    rgb = np.stack([gris] * 3, axis=-1) * 0.8
    colores = {(True, True): (0.1, 0.9, 0.1),
               (False, True): (1.0, 0.15, 0.15),
               (True, False): (0.2, 0.5, 1.0)}
    for (en_real, en_pred), color in colores.items():
        zona = (real == en_real) & (prediccion == en_pred)
        rgb[zona] = 0.4 * rgb[zona] + 0.6 * np.array(color)
    return rgb


def grilla(filas, titulo, ruta):
    figura, ejes = plt.subplots(len(filas), 4, figsize=(11, 2.8 * len(filas)))
    ejes = np.atleast_2d(ejes)
    for fila, (corte, imagen, real, probabilidad, prediccion, d) in zip(ejes, filas):
        fila[0].imshow(imagen[..., 1], cmap="gray")
        fila[0].set_title(f"{corte.paciente} #{corte.numero}", fontsize=8)
        fila[1].imshow(real, cmap="gray")
        fila[1].set_title("máscara real", fontsize=9)
        fila[2].imshow(probabilidad, cmap="magma", vmin=0, vmax=1)
        fila[2].set_title("probabilidad predicha", fontsize=9)
        fila[3].imshow(superposicion(imagen, real, prediccion))
        fila[3].set_title(f"Dice {d:.2f}", fontsize=9)
        for eje in fila:
            eje.axis("off")
    figura.suptitle(f"{titulo}\nverde: acierto · rojo: falso positivo · azul: falso negativo",
                    fontsize=11)
    figura.tight_layout()
    figura.savefig(ruta, dpi=100)
    plt.close(figura)


def main():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, tamano = cargar_modelo(dispositivo)
    test = dividir_por_paciente(listar_cortes())["test"]

    filas = []
    for corte, (imagen, real, probabilidad) in zip(test, predecir(modelo, test, tamano, dispositivo)):
        # La predicción se lleva al tamaño original (256) para comparar con la máscara real
        if probabilidad.shape != real.shape:
            import cv2
            probabilidad = cv2.resize(probabilidad, real.shape[::-1], interpolation=cv2.INTER_LINEAR)
        prediccion = probabilidad > UMBRAL
        filas.append((corte, imagen, real, probabilidad, prediccion, dice(prediccion, real)))

    # --- Métricas por corte ---
    con_tumor = [f for f in filas if f[2].any()]
    sin_tumor = [f for f in filas if not f[2].any()]
    dice_con = np.mean([f[5] for f in con_tumor])
    iou_con = np.mean([iou(f[4], f[2]) for f in con_tumor])
    falsos_positivos = sum(f[4].any() for f in sin_tumor)
    detectados = sum(f[4].any() for f in con_tumor)

    # --- Métricas por paciente: se apilan los cortes como un volumen 3D ---
    pacientes = sorted({f[0].paciente for f in filas})
    dice_paciente = {p: dice(np.stack([f[4] for f in filas if f[0].paciente == p]),
                             np.stack([f[2] for f in filas if f[0].paciente == p]))
                     for p in pacientes}

    texto = (
        f"Test: {len(pacientes)} pacientes, {len(filas)} cortes "
        f"({len(con_tumor)} con tumor, {len(sin_tumor)} sin tumor)\n\n"
        f"Dice medio por corte, todos los cortes:     {np.mean([f[5] for f in filas]):.3f}\n"
        f"Dice medio por corte, cortes con tumor:     {dice_con:.3f}\n"
        f"IoU medio por corte, cortes con tumor:      {iou_con:.3f}\n"
        f"Dice volumétrico medio por paciente:        {np.mean(list(dice_paciente.values())):.3f}\n\n"
        f"Detección (¿hay tumor en el corte?)\n"
        f"  cortes con tumor detectados:   {detectados}/{len(con_tumor)} "
        f"({detectados / len(con_tumor):.0%})\n"
        f"  falsos positivos en cortes sin tumor: {falsos_positivos}/{len(sin_tumor)} "
        f"({falsos_positivos / len(sin_tumor):.0%})\n\n"
        "Dice por paciente:\n" +
        "".join(f"  {p}  {d:.3f}\n" for p, d in sorted(dice_paciente.items(), key=lambda x: x[1]))
    )
    print(texto)
    (CARPETA_RESULTADOS / "metricas_test.txt").write_text(texto, encoding="utf-8")

    # --- Figuras ---
    ordenados = sorted(con_tumor, key=lambda f: f[5])
    grilla(ordenados[-6:][::-1], "Mejores predicciones (cortes con tumor)",
           CARPETA_RESULTADOS / "predicciones_mejores.png")
    grilla(ordenados[:6], "Peores predicciones (cortes con tumor)",
           CARPETA_RESULTADOS / "predicciones_peores.png")
    rng = np.random.default_rng(0)
    azar = [con_tumor[i] for i in rng.choice(len(con_tumor), 4, replace=False)]
    azar += [sin_tumor[i] for i in rng.choice(len(sin_tumor), 2, replace=False)]
    grilla(azar, "Cortes al azar (4 con tumor, 2 sin tumor)",
           CARPETA_RESULTADOS / "predicciones_azar.png")

    figura, eje = plt.subplots(figsize=(10, 4))
    nombres, valores = zip(*sorted(dice_paciente.items(), key=lambda x: x[1]))
    eje.bar(range(len(nombres)), valores, color="#2e86ab")
    eje.set_xticks(range(len(nombres)), nombres, rotation=60, ha="right", fontsize=8)
    eje.set(title="Dice volumétrico por paciente de test", ylabel="Dice", ylim=(0, 1))
    eje.grid(axis="y", alpha=0.3)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "dice_por_paciente.png", dpi=110)
    plt.close(figura)
    print(f"Figuras en {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    main()
