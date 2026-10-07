# Carga del dataset LGG, división por paciente y preprocesamiento

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

RAIZ_TP = Path(__file__).resolve().parent.parent
CARPETA_DATOS = RAIZ_TP / "datos" / "kaggle_3m"

CANALES = ("pre-contraste", "FLAIR", "post-contraste")
SEMILLA = 42


class Corte:
    def __init__(self, imagen):
        # ej: TCGA_CS_4941_19960909_12.tif -> paciente TCGA_CS_4941, corte 12
        self.imagen = imagen
        self.mascara = imagen.with_name(imagen.stem + "_mask.tif")
        partes = imagen.stem.split("_")
        self.paciente = "_".join(partes[:3])
        self.numero = int(partes[-1])
        self.tiene_tumor = leer_mascara(self.mascara).any()


def listar_cortes():
    if not CARPETA_DATOS.exists():
        raise FileNotFoundError("No está el dataset, correr src/descargar_datos.py")
    imagenes = [p for p in CARPETA_DATOS.glob("*/*.tif") if not p.stem.endswith("_mask")]
    cortes = [Corte(p) for p in imagenes]
    cortes.sort(key=lambda c: (c.paciente, c.numero))
    return cortes


def dividir_por_paciente(cortes):
    # 70% entrenamiento, 15% validación, 15% test, separando por paciente
    pacientes = sorted(set(c.paciente for c in cortes))
    random.Random(SEMILLA).shuffle(pacientes)

    n_ent = round(0.70 * len(pacientes))
    n_val = round(0.15 * len(pacientes))
    ent = pacientes[:n_ent]
    val = pacientes[n_ent:n_ent + n_val]
    test = pacientes[n_ent + n_val:]

    return {
        "entrenamiento": [c for c in cortes if c.paciente in ent],
        "validacion": [c for c in cortes if c.paciente in val],
        "test": [c for c in cortes if c.paciente in test],
    }


def leer_imagen(ruta):
    # opencv lee en BGR, lo paso a RGB para que quede (pre, FLAIR, post)
    return cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2RGB)


def leer_mascara(ruta):
    return cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE) > 127


def redimensionar(imagen, mascara, tamano):
    if imagen.shape[0] == tamano:
        return imagen, mascara
    imagen = cv2.resize(imagen, (tamano, tamano), interpolation=cv2.INTER_AREA)
    # vecino más cercano para que la máscara siga siendo binaria
    mascara = cv2.resize(mascara.astype(np.uint8), (tamano, tamano),
                         interpolation=cv2.INTER_NEAREST) > 0
    return imagen, mascara


def normalizar(imagen):
    # media 0 y desvío 1 por canal, calculados solo sobre la cabeza
    imagen = imagen.astype(np.float32) / 255
    cabeza = imagen[:, :, 1] > 0.08
    if cabeza.sum() < 50:
        return np.zeros_like(imagen)

    media = imagen[cabeza].mean(axis=0)
    desvio = imagen[cabeza].std(axis=0)
    imagen = (imagen - media) / np.maximum(desvio, 1e-3)

    # hay cortes con un canal constante (desvío 0) que daban valores enormes
    imagen[:, :, desvio < 0.01] = 0
    return np.clip(imagen, -5, 5)


def aumentar(imagen, mascara, rng):
    # espejo horizontal
    if rng.random() < 0.5:
        imagen = imagen[:, ::-1].copy()
        mascara = mascara[:, ::-1].copy()

    # rotación y escala
    h, w = mascara.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.9, 1.1))
    imagen = cv2.warpAffine(imagen, M, (w, h), flags=cv2.INTER_LINEAR)
    mascara = cv2.warpAffine(mascara.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0

    # brillo y contraste (sin tocar el fondo)
    fondo = imagen.max(axis=2) == 0
    imagen = imagen.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    imagen = np.clip(imagen, 0, 255)
    imagen[fondo] = 0
    return imagen.astype(np.uint8), mascara


def preprocesar(imagen, mascara, tamano, rng=None):
    # si se pasa rng se aplica augmentation
    imagen, mascara = redimensionar(imagen, mascara, tamano)
    if rng is not None:
        imagen, mascara = aumentar(imagen, mascara, rng)
    return normalizar(imagen), mascara


class DatasetLGG(Dataset):
    def __init__(self, cortes, tamano=256, aumentar=False):
        self.cortes = cortes
        self.tamano = tamano
        self.aumentar = aumentar

    def __len__(self):
        return len(self.cortes)

    def __getitem__(self, i):
        corte = self.cortes[i]
        imagen = leer_imagen(corte.imagen)
        mascara = leer_mascara(corte.mascara)
        rng = np.random.default_rng() if self.aumentar else None
        imagen, mascara = preprocesar(imagen, mascara, self.tamano, rng)

        x = torch.from_numpy(imagen.transpose(2, 0, 1).copy())
        y = torch.from_numpy(mascara[None].astype(np.float32))
        return x, y
