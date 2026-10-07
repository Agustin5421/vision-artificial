"""
Entrena la U-Net y guarda en resultados/:

- modelo.pt         los pesos de la mejor época (según Dice de validación)
- historial.csv     pérdida y Dice por época
- curvas.png        evolución de la pérdida y del Dice
- particion.txt     qué pacientes quedaron en cada conjunto

Uso:
    python src/entrenar.py --epocas 40 --tamano 256 --batch 16
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datos import (DatasetLGG, RAIZ_TP, dividir_por_paciente,
                   inicializar_worker, listar_cortes)
from metricas import UMBRAL, perdida_bce_dice
from unet import UNet

CARPETA_RESULTADOS = RAIZ_TP / "resultados"


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--tamano", type=int, default=256, help="lado de la imagen en px")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32, help="filtros del primer nivel de la U-Net")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def una_epoca(modelo, cargador, dispositivo, optimizador=None, escalador=None):
    """Recorre el cargador una vez. Con optimizador entrena; sin él, evalúa."""
    entrenando = optimizador is not None
    modelo.train(entrenando)

    perdida_total, interseccion, suma = 0.0, 0.0, 0.0
    with torch.set_grad_enabled(entrenando):
        for imagenes, mascaras in cargador:
            imagenes = imagenes.to(dispositivo, non_blocking=True)
            mascaras = mascaras.to(dispositivo, non_blocking=True)

            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                logits = modelo(imagenes)
            perdida = perdida_bce_dice(logits.float(), mascaras)

            if entrenando:
                optimizador.zero_grad(set_to_none=True)
                escalador.scale(perdida).backward()
                escalador.step(optimizador)
                escalador.update()

            perdida_total += perdida.item() * len(imagenes)
            prediccion = torch.sigmoid(logits.float()) > UMBRAL
            interseccion += (prediccion & (mascaras > 0.5)).sum().item()
            suma += prediccion.sum().item() + mascaras.sum().item()

    # Dice "global": todos los píxeles del conjunto juntos, como si fuera un volumen
    return perdida_total / len(cargador.dataset), 2 * interseccion / max(suma, 1)


def graficar(historial, ruta):
    epocas = [h["epoca"] for h in historial]
    figura, (eje_perdida, eje_dice) = plt.subplots(1, 2, figsize=(11, 4))

    eje_perdida.plot(epocas, [h["perdida_entrenamiento"] for h in historial], label="entrenamiento")
    eje_perdida.plot(epocas, [h["perdida_validacion"] for h in historial], label="validación")
    eje_perdida.set(title="Pérdida (BCE + Dice)", xlabel="época")
    eje_perdida.legend()

    eje_dice.plot(epocas, [h["dice_entrenamiento"] for h in historial], label="entrenamiento")
    eje_dice.plot(epocas, [h["dice_validacion"] for h in historial], label="validación")
    eje_dice.set(title="Dice", xlabel="época", ylim=(0, 1))
    eje_dice.legend()

    for eje in (eje_perdida, eje_dice):
        eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(ruta, dpi=120)
    plt.close(figura)


def guardar_particion(particion, ruta):
    with open(ruta, "w", encoding="utf-8") as archivo:
        for nombre, cortes in particion.items():
            pacientes = sorted({c.paciente for c in cortes})
            con_tumor = sum(c.tiene_tumor for c in cortes)
            archivo.write(f"{nombre}: {len(pacientes)} pacientes, {len(cortes)} cortes, "
                          f"{con_tumor} con tumor ({con_tumor / len(cortes):.0%})\n")
            archivo.write("  " + " ".join(pacientes) + "\n\n")


def main():
    args = argumentos()
    torch.manual_seed(0)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    particion = dividir_por_paciente(listar_cortes())
    guardar_particion(particion, CARPETA_RESULTADOS / "particion.txt")
    print((CARPETA_RESULTADOS / "particion.txt").read_text(encoding="utf-8")
          .split("\n  ")[0])  # primera línea, de muestra

    opciones = dict(batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, worker_init_fn=inicializar_worker,
                    persistent_workers=args.workers > 0)
    entrenamiento = DataLoader(DatasetLGG(particion["entrenamiento"], args.tamano, aumentar=True),
                               shuffle=True, drop_last=True, **opciones)
    validacion = DataLoader(DatasetLGG(particion["validacion"], args.tamano), **opciones)

    modelo = UNet(base=args.base).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    # Si el Dice de validación se estanca 5 épocas, divide el learning rate por 2
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max", factor=0.5, patience=5)
    escalador = torch.amp.GradScaler("cuda", enabled=dispositivo.type == "cuda")

    historial, mejor_dice = [], -1.0
    for epoca in range(1, args.epocas + 1):
        inicio = time.time()
        perdida_ent, dice_ent = una_epoca(modelo, entrenamiento, dispositivo, optimizador, escalador)
        perdida_val, dice_val = una_epoca(modelo, validacion, dispositivo)
        planificador.step(dice_val)

        historial.append(dict(epoca=epoca,
                              perdida_entrenamiento=perdida_ent, perdida_validacion=perdida_val,
                              dice_entrenamiento=dice_ent, dice_validacion=dice_val,
                              lr=optimizador.param_groups[0]["lr"]))

        marca = ""
        if dice_val > mejor_dice:
            mejor_dice = dice_val
            torch.save({"pesos": modelo.state_dict(), "tamano": args.tamano,
                        "base": args.base, "epoca": epoca},
                       CARPETA_RESULTADOS / "modelo.pt")
            marca = "  * mejor"
        print(f"época {epoca:3d}  pérdida {perdida_ent:.4f} / {perdida_val:.4f}  "
              f"Dice {dice_ent:.3f} / {dice_val:.3f}  "
              f"({time.time() - inicio:.0f}s){marca}", flush=True)

        with open(CARPETA_RESULTADOS / "historial.csv", "w", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=historial[0].keys())
            escritor.writeheader()
            escritor.writerows(historial)
        graficar(historial, CARPETA_RESULTADOS / "curvas.png")

    print(f"Mejor Dice de validación: {mejor_dice:.3f}")


if __name__ == "__main__":
    main()
