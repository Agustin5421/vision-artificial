# Entrenamiento de la U-Net
# Guarda en resultados/: modelo.pt (mejor época), historial.csv, curvas.png, particion.txt
#
# uso: python src/entrenar.py --epocas 40

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datos import RAIZ_TP, DatasetLGG, dividir_por_paciente, listar_cortes
from metricas import UMBRAL, perdida_bce_dice
from unet import UNet

RESULTADOS = RAIZ_TP / "resultados"

parser = argparse.ArgumentParser()
parser.add_argument("--epocas", type=int, default=40)
parser.add_argument("--tamano", type=int, default=256)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--base", type=int, default=32)
parser.add_argument("--workers", type=int, default=4)


def correr_epoca(modelo, loader, device, optimizador=None, scaler=None):
    # si se pasa optimizador entrena, si no solo evalúa
    entrenar = optimizador is not None
    modelo.train(entrenar)
    perdida_total = 0
    interseccion = 0
    suma = 0

    with torch.set_grad_enabled(entrenar):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # precisión mixta para que sea más rápido en GPU
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = modelo(x)
            logits = logits.float()
            perdida = perdida_bce_dice(logits, y)

            if entrenar:
                optimizador.zero_grad()
                scaler.scale(perdida).backward()
                scaler.step(optimizador)
                scaler.update()

            perdida_total += perdida.item() * len(x)
            pred = torch.sigmoid(logits) > UMBRAL
            interseccion += (pred & (y > 0.5)).sum().item()
            suma += pred.sum().item() + y.sum().item()

    # dice sumando todos los píxeles del conjunto
    return perdida_total / len(loader.dataset), 2 * interseccion / max(suma, 1)


def graficar(historial):
    epocas = [h["epoca"] for h in historial]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epocas, [h["perdida_ent"] for h in historial], label="entrenamiento")
    ax1.plot(epocas, [h["perdida_val"] for h in historial], label="validación")
    ax1.set_title("Pérdida (BCE + Dice)")
    ax1.set_xlabel("época")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epocas, [h["dice_ent"] for h in historial], label="entrenamiento")
    ax2.plot(epocas, [h["dice_val"] for h in historial], label="validación")
    ax2.set_title("Dice")
    ax2.set_xlabel("época")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTADOS / "curvas.png", dpi=120)
    plt.close(fig)


def guardar_particion(particion):
    with open(RESULTADOS / "particion.txt", "w") as f:
        for nombre, cortes in particion.items():
            pacientes = sorted(set(c.paciente for c in cortes))
            con_tumor = sum(c.tiene_tumor for c in cortes)
            f.write(f"{nombre}: {len(pacientes)} pacientes, {len(cortes)} cortes, "
                    f"{con_tumor} con tumor ({con_tumor / len(cortes):.0%})\n")
            f.write("  " + " ".join(pacientes) + "\n\n")


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(0)
    RESULTADOS.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    particion = dividir_por_paciente(listar_cortes())
    guardar_particion(particion)
    for nombre, cortes in particion.items():
        print(nombre, len(cortes), "cortes")

    ds_ent = DatasetLGG(particion["entrenamiento"], args.tamano, aumentar=True)
    ds_val = DatasetLGG(particion["validacion"], args.tamano)
    loader_ent = DataLoader(ds_ent, batch_size=args.batch, shuffle=True, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0)
    loader_val = DataLoader(ds_val, batch_size=args.batch,
                            num_workers=args.workers, persistent_workers=args.workers > 0)

    modelo = UNet(base=args.base).to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    # si el dice de validación no mejora en 5 épocas, lr / 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max",
                                                           factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    historial = []
    mejor_dice = -1
    for epoca in range(1, args.epocas + 1):
        inicio = time.time()
        perdida_ent, dice_ent = correr_epoca(modelo, loader_ent, device, optimizador, scaler)
        perdida_val, dice_val = correr_epoca(modelo, loader_val, device)
        scheduler.step(dice_val)

        historial.append({"epoca": epoca, "perdida_ent": perdida_ent, "perdida_val": perdida_val,
                          "dice_ent": dice_ent, "dice_val": dice_val,
                          "lr": optimizador.param_groups[0]["lr"]})

        texto = (f"época {epoca:3d}  pérdida {perdida_ent:.4f} / {perdida_val:.4f}  "
                 f"Dice {dice_ent:.3f} / {dice_val:.3f}  ({time.time() - inicio:.0f}s)")
        if dice_val > mejor_dice:
            mejor_dice = dice_val
            torch.save({"pesos": modelo.state_dict(), "tamano": args.tamano,
                        "base": args.base, "epoca": epoca}, RESULTADOS / "modelo.pt")
            texto += "  * mejor"
        print(texto, flush=True)

        with open(RESULTADOS / "historial.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=historial[0].keys())
            writer.writeheader()
            writer.writerows(historial)
        graficar(historial)

    print("mejor dice de validación:", round(mejor_dice, 3))
