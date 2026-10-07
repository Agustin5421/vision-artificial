"""
Figuras del dataset y del preprocesamiento, en resultados/:

- ejemplos_originales.png   los 3 canales de algunos cortes y su máscara
- preprocesamiento.png      original -> redimensionado/normalizado -> aumentado
- estadisticas.png          cortes con/sin tumor por conjunto y tamaño de los tumores

Uso:
    python src/visualizar_datos.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datos import (CANALES, RAIZ_TP, dividir_por_paciente, leer_imagen,
                   leer_mascara, listar_cortes, preprocesar)

CARPETA_RESULTADOS = RAIZ_TP / "resultados"


def contorno(eje, mascara, color="lime"):
    if mascara.any():
        eje.contour(mascara, levels=[0.5], colors=color, linewidths=1)


def ejemplos_originales(cortes, ruta):
    figura, ejes = plt.subplots(len(cortes), 5, figsize=(13, 2.7 * len(cortes)))
    for fila, corte in zip(ejes, cortes):
        imagen, mascara = leer_imagen(corte.imagen), leer_mascara(corte.mascara)
        paneles = [(imagen[..., i], CANALES[i]) for i in range(3)]
        paneles += [(imagen, "3 canales como RGB"), (mascara, "máscara")]
        for eje, (panel, titulo) in zip(fila, paneles):
            eje.imshow(panel, cmap="gray")
            eje.set_title(titulo, fontsize=9)
            eje.axis("off")
        contorno(fila[1], mascara)
        fila[0].text(-10, 128, f"{corte.paciente}\ncorte {corte.numero}",
                     fontsize=8, ha="right", va="center")
    figura.tight_layout()
    figura.savefig(ruta, dpi=110)
    plt.close(figura)


def mostrar_normalizada(eje, imagen):
    """El canal FLAIR normalizado (media 0, desvío 1), entre -2.5 y 3 desvíos."""
    eje.imshow(imagen[..., 1], cmap="gray", vmin=-2.5, vmax=3)


def preprocesamiento(cortes, ruta, tamano=256, aumentos=3):
    rng = np.random.default_rng(1)
    columnas = 2 + aumentos
    figura, ejes = plt.subplots(len(cortes), columnas, figsize=(2.6 * columnas, 2.7 * len(cortes)))
    for fila, corte in zip(ejes, cortes):
        imagen, mascara = leer_imagen(corte.imagen), leer_mascara(corte.mascara)

        fila[0].imshow(imagen[..., 1], cmap="gray")
        contorno(fila[0], mascara)
        fila[0].set_title("original (FLAIR)", fontsize=9)

        normalizada, m = preprocesar(imagen, mascara, tamano)
        mostrar_normalizada(fila[1], normalizada)
        contorno(fila[1], m)
        fila[1].set_title("normalizada", fontsize=9)

        for k in range(aumentos):
            aumentada, m = preprocesar(imagen, mascara, tamano, rng)
            mostrar_normalizada(fila[2 + k], aumentada)
            contorno(fila[2 + k], m)
            fila[2 + k].set_title(f"aumentada {k + 1}", fontsize=9)

        for eje in fila:
            eje.axis("off")
    figura.suptitle("Preprocesamiento (contorno verde: máscara)", fontsize=11)
    figura.tight_layout()
    figura.savefig(ruta, dpi=110)
    plt.close(figura)


def estadisticas(particion, ruta):
    figura, (eje_barras, eje_hist) = plt.subplots(1, 2, figsize=(11, 4))

    nombres = list(particion)
    con = [sum(c.tiene_tumor for c in particion[n]) for n in nombres]
    sin = [len(particion[n]) - k for n, k in zip(nombres, con)]
    eje_barras.bar(nombres, sin, label="sin tumor", color="#9aa5b1")
    eje_barras.bar(nombres, con, bottom=sin, label="con tumor", color="#d1495b")
    for i, n in enumerate(nombres):
        pacientes = len({c.paciente for c in particion[n]})
        eje_barras.text(i, sin[i] + con[i], f"{pacientes} pacientes", ha="center", va="bottom", fontsize=9)
    eje_barras.set(title="Cortes por conjunto", ylabel="cortes")
    eje_barras.legend()

    areas = [leer_mascara(c.mascara).mean() * 100
             for cortes in particion.values() for c in cortes if c.tiene_tumor]
    eje_hist.hist(areas, bins=40, color="#d1495b")
    eje_hist.set(title="Área del tumor en los cortes que lo tienen",
                 xlabel="% de la imagen", ylabel="cortes")
    eje_hist.axvline(np.median(areas), color="k", ls="--", label=f"mediana {np.median(areas):.1f}%")
    eje_hist.legend()

    figura.tight_layout()
    figura.savefig(ruta, dpi=120)
    plt.close(figura)


def main():
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    cortes = listar_cortes()
    particion = dividir_por_paciente(cortes)

    con_tumor = [c for c in cortes if c.tiene_tumor]
    print(f"{len(cortes)} cortes de {len({c.paciente for c in cortes})} pacientes, "
          f"{len(con_tumor)} con tumor ({len(con_tumor) / len(cortes):.0%})")

    # Cortes con tumores de distinto tamaño, y uno sin tumor
    por_area = sorted(con_tumor, key=lambda c: leer_mascara(c.mascara).sum())
    elegidos = [por_area[len(por_area) // 10], por_area[len(por_area) // 2],
                por_area[-len(por_area) // 20], next(c for c in cortes[30:] if not c.tiene_tumor)]

    ejemplos_originales(elegidos, CARPETA_RESULTADOS / "ejemplos_originales.png")
    preprocesamiento(elegidos[:3], CARPETA_RESULTADOS / "preprocesamiento.png")
    estadisticas(particion, CARPETA_RESULTADOS / "estadisticas.png")
    print(f"Figuras en {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    main()
