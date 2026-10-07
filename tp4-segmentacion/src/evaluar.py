# Evaluación del modelo sobre test
# Guarda en resultados/: metricas_test.txt, predicciones_*.png, dice_por_paciente.png
#
# uso: python src/evaluar.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datos import (RAIZ_TP, dividir_por_paciente, leer_imagen, leer_mascara,
                   listar_cortes, preprocesar)
from metricas import UMBRAL, dice, iou
from unet import UNet

RESULTADOS = RAIZ_TP / "resultados"


def superposicion(imagen, real, pred):
    # FLAIR en gris, verde = acierto, rojo = falso positivo, azul = falso negativo
    gris = imagen[:, :, 1] / 255 * 0.8
    rgb = np.stack([gris, gris, gris], axis=2)
    rgb[real & pred] = rgb[real & pred] * 0.4 + np.array([0.1, 0.9, 0.1]) * 0.6
    rgb[~real & pred] = rgb[~real & pred] * 0.4 + np.array([1.0, 0.15, 0.15]) * 0.6
    rgb[real & ~pred] = rgb[real & ~pred] * 0.4 + np.array([0.2, 0.5, 1.0]) * 0.6
    return rgb


def figura_predicciones(filas, titulo, archivo):
    fig, axes = plt.subplots(len(filas), 4, figsize=(11, 2.8 * len(filas)))
    for ax, r in zip(axes, filas):
        ax[0].imshow(r["imagen"][:, :, 1], cmap="gray")
        ax[0].set_title(f"{r['corte'].paciente} #{r['corte'].numero}", fontsize=8)
        ax[1].imshow(r["real"], cmap="gray")
        ax[1].set_title("máscara real", fontsize=9)
        ax[2].imshow(r["prob"], cmap="magma", vmin=0, vmax=1)
        ax[2].set_title("probabilidad predicha", fontsize=9)
        ax[3].imshow(superposicion(r["imagen"], r["real"], r["pred"]))
        ax[3].set_title(f"Dice {r['dice']:.2f}", fontsize=9)
        for a in ax:
            a.axis("off")
    fig.suptitle(titulo + "\nverde: acierto, rojo: falso positivo, azul: falso negativo")
    fig.tight_layout()
    fig.savefig(RESULTADOS / archivo, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    guardado = torch.load(RESULTADOS / "modelo.pt", map_location=device)
    modelo = UNet(base=guardado["base"]).to(device)
    modelo.load_state_dict(guardado["pesos"])
    modelo.eval()
    tamano = guardado["tamano"]
    print("modelo de la época", guardado["epoca"])

    test = dividir_por_paciente(listar_cortes())["test"]

    resultados = []
    with torch.no_grad():
        for corte in test:
            imagen = leer_imagen(corte.imagen)
            real = leer_mascara(corte.mascara)
            x, _ = preprocesar(imagen, real, tamano)
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
            prob = torch.sigmoid(modelo(x))[0, 0].cpu().numpy()
            if prob.shape != real.shape:  # si se entrenó con otro tamaño
                import cv2
                prob = cv2.resize(prob, real.shape[::-1])
            pred = prob > UMBRAL
            resultados.append({"corte": corte, "imagen": imagen, "real": real,
                               "prob": prob, "pred": pred, "dice": dice(pred, real)})

    con_tumor = [r for r in resultados if r["real"].any()]
    sin_tumor = [r for r in resultados if not r["real"].any()]
    detectados = sum(r["pred"].any() for r in con_tumor)
    falsos_pos = sum(r["pred"].any() for r in sin_tumor)

    # dice por paciente: se juntan todos sus cortes como un volumen
    dice_paciente = {}
    for p in sorted(set(r["corte"].paciente for r in resultados)):
        rs = [r for r in resultados if r["corte"].paciente == p]
        dice_paciente[p] = dice(np.stack([r["pred"] for r in rs]), np.stack([r["real"] for r in rs]))

    texto = f"Test: {len(dice_paciente)} pacientes, {len(resultados)} cortes "
    texto += f"({len(con_tumor)} con tumor, {len(sin_tumor)} sin tumor)\n\n"
    texto += f"Dice medio por corte (todos):           {np.mean([r['dice'] for r in resultados]):.3f}\n"
    texto += f"Dice medio por corte (con tumor):       {np.mean([r['dice'] for r in con_tumor]):.3f}\n"
    texto += f"IoU medio por corte (con tumor):        {np.mean([iou(r['pred'], r['real']) for r in con_tumor]):.3f}\n"
    texto += f"Dice por paciente (promedio):           {np.mean(list(dice_paciente.values())):.3f}\n\n"
    texto += f"Cortes con tumor detectados:            {detectados}/{len(con_tumor)}\n"
    texto += f"Cortes sin tumor con falso positivo:    {falsos_pos}/{len(sin_tumor)}\n\n"
    texto += "Dice por paciente:\n"
    for p, d in sorted(dice_paciente.items(), key=lambda x: x[1]):
        texto += f"  {p}  {d:.3f}\n"
    print(texto)
    (RESULTADOS / "metricas_test.txt").write_text(texto, encoding="utf-8")

    con_tumor.sort(key=lambda r: r["dice"])
    figura_predicciones(con_tumor[::-1][:6], "Mejores predicciones", "predicciones_mejores.png")
    figura_predicciones(con_tumor[:6], "Peores predicciones", "predicciones_peores.png")
    rng = np.random.default_rng(0)
    azar = [con_tumor[i] for i in rng.choice(len(con_tumor), 4, replace=False)]
    azar += [sin_tumor[i] for i in rng.choice(len(sin_tumor), 2, replace=False)]
    figura_predicciones(azar, "Cortes al azar (4 con tumor, 2 sin tumor)", "predicciones_azar.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ordenados = sorted(dice_paciente.items(), key=lambda x: x[1])
    ax.bar(range(len(ordenados)), [d for _, d in ordenados])
    ax.set_xticks(range(len(ordenados)))
    ax.set_xticklabels([p for p, _ in ordenados], rotation=60, ha="right", fontsize=8)
    ax.set_title("Dice por paciente (test)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "dice_por_paciente.png", dpi=110)
    plt.close(fig)
    print("figuras guardadas en", RESULTADOS)
