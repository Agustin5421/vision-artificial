# Figuras del dataset y del preprocesamiento (en resultados/)
# uso: python src/visualizar_datos.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datos import (CANALES, RAIZ_TP, dividir_por_paciente, leer_imagen,
                   leer_mascara, listar_cortes, preprocesar)

RESULTADOS = RAIZ_TP / "resultados"


def dibujar_contorno(ax, mascara):
    if mascara.any():
        ax.contour(mascara, levels=[0.5], colors="lime", linewidths=1)


def figura_originales(cortes):
    fig, axes = plt.subplots(len(cortes), 5, figsize=(13, 2.7 * len(cortes)))
    for fila, corte in zip(axes, cortes):
        imagen = leer_imagen(corte.imagen)
        mascara = leer_mascara(corte.mascara)
        for i in range(3):
            fila[i].imshow(imagen[:, :, i], cmap="gray")
            fila[i].set_title(CANALES[i], fontsize=9)
        fila[3].imshow(imagen)
        fila[3].set_title("3 canales como RGB", fontsize=9)
        fila[4].imshow(mascara, cmap="gray")
        fila[4].set_title("máscara", fontsize=9)
        dibujar_contorno(fila[1], mascara)
        fila[0].text(-10, 128, f"{corte.paciente}\ncorte {corte.numero}",
                     fontsize=8, ha="right", va="center")
        for ax in fila:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "ejemplos_originales.png", dpi=110)
    plt.close(fig)


def figura_preprocesamiento(cortes):
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(len(cortes), 5, figsize=(13, 2.7 * len(cortes)))
    for fila, corte in zip(axes, cortes):
        imagen = leer_imagen(corte.imagen)
        mascara = leer_mascara(corte.mascara)

        fila[0].imshow(imagen[:, :, 1], cmap="gray")
        dibujar_contorno(fila[0], mascara)
        fila[0].set_title("original (FLAIR)", fontsize=9)

        norm, m = preprocesar(imagen, mascara, 256)
        fila[1].imshow(norm[:, :, 1], cmap="gray", vmin=-2.5, vmax=3)
        dibujar_contorno(fila[1], m)
        fila[1].set_title("normalizada", fontsize=9)

        for k in range(3):
            aum, m = preprocesar(imagen, mascara, 256, rng)
            fila[2 + k].imshow(aum[:, :, 1], cmap="gray", vmin=-2.5, vmax=3)
            dibujar_contorno(fila[2 + k], m)
            fila[2 + k].set_title(f"aumentada {k + 1}", fontsize=9)

        for ax in fila:
            ax.axis("off")
    fig.suptitle("Preprocesamiento (contorno verde: máscara)")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "preprocesamiento.png", dpi=110)
    plt.close(fig)


def figura_estadisticas(particion):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    nombres = list(particion.keys())
    con = [sum(c.tiene_tumor for c in particion[n]) for n in nombres]
    sin = [len(particion[n]) - con[i] for i, n in enumerate(nombres)]
    ax1.bar(nombres, sin, label="sin tumor", color="gray")
    ax1.bar(nombres, con, bottom=sin, label="con tumor", color="tab:red")
    for i, n in enumerate(nombres):
        pacientes = len(set(c.paciente for c in particion[n]))
        ax1.text(i, sin[i] + con[i], f"{pacientes} pacientes", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Cortes por conjunto")
    ax1.set_ylabel("cortes")
    ax1.legend()

    areas = []
    for cortes in particion.values():
        for c in cortes:
            if c.tiene_tumor:
                areas.append(leer_mascara(c.mascara).mean() * 100)
    ax2.hist(areas, bins=40, color="tab:red")
    ax2.axvline(np.median(areas), color="k", ls="--", label=f"mediana {np.median(areas):.1f}%")
    ax2.set_title("Área del tumor en los cortes que lo tienen")
    ax2.set_xlabel("% de la imagen")
    ax2.set_ylabel("cortes")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(RESULTADOS / "estadisticas.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    RESULTADOS.mkdir(exist_ok=True)
    cortes = listar_cortes()
    particion = dividir_por_paciente(cortes)

    con_tumor = [c for c in cortes if c.tiene_tumor]
    print(len(cortes), "cortes,", len(con_tumor), "con tumor")

    # ejemplos con tumores de distinto tamaño y uno sin tumor
    con_tumor.sort(key=lambda c: leer_mascara(c.mascara).sum())
    n = len(con_tumor)
    sin_tumor = [c for c in cortes[30:] if not c.tiene_tumor][0]
    ejemplos = [con_tumor[n // 10], con_tumor[n // 2], con_tumor[-n // 20], sin_tumor]

    figura_originales(ejemplos)
    figura_preprocesamiento(ejemplos[:3])
    figura_estadisticas(particion)
    print("figuras guardadas en", RESULTADOS)
