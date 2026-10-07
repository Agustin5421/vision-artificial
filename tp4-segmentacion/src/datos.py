"""
Carga, partición y preprocesamiento del dataset LGG.

1. Lista los cortes: cada corte es una imagen .tif de 3 canales
   (pre-contraste, FLAIR, post-contraste) con su máscara binaria.
2. Divide en entrenamiento / validación / test **por paciente**, para que
   cortes vecinos de un mismo cerebro no queden de los dos lados.
3. Preprocesa cada corte: redimensiona, normaliza y, sólo en entrenamiento,
   aplica data augmentation.
"""

import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

RAIZ_TP = Path(__file__).resolve().parent.parent
CARPETA_DATOS = RAIZ_TP / "datos" / "kaggle_3m"

CANALES = ("pre-contraste", "FLAIR", "post-contraste")

FRACCIONES = (0.70, 0.15, 0.15)  # entrenamiento, validación, test
SEMILLA = 42


class Corte:
    """Un corte axial de resonancia: rutas a la imagen y a su máscara."""

    def __init__(self, imagen, mascara):
        self.imagen = imagen
        self.mascara = mascara
        # TCGA_CS_4941_19960909_12.tif -> paciente TCGA_CS_4941, corte 12
        partes = imagen.stem.split("_")
        self.paciente = "_".join(partes[:3])
        self.numero = int(partes[-1])
        self._tiene_tumor = None

    @property
    def tiene_tumor(self):
        if self._tiene_tumor is None:
            self._tiene_tumor = bool(leer_mascara(self.mascara).any())
        return self._tiene_tumor


def listar_cortes(carpeta=CARPETA_DATOS):
    """Todos los cortes del dataset, ordenados por paciente y número de corte."""
    if not carpeta.exists():
        raise FileNotFoundError(
            f"No está el dataset en {carpeta}. Correr src/descargar_datos.py")

    cortes = [Corte(imagen, imagen.with_name(imagen.stem + "_mask.tif"))
              for imagen in carpeta.glob("*/*.tif")
              if not re.search(r"_mask$", imagen.stem)]
    return sorted(cortes, key=lambda c: (c.paciente, c.numero))


def dividir_por_paciente(cortes, fracciones=FRACCIONES, semilla=SEMILLA):
    """Reparte los pacientes (no los cortes) en entrenamiento, validación y test."""
    pacientes = sorted({c.paciente for c in cortes})
    random.Random(semilla).shuffle(pacientes)

    n_entrenamiento = round(fracciones[0] * len(pacientes))
    n_validacion = round(fracciones[1] * len(pacientes))
    grupos = {
        "entrenamiento": set(pacientes[:n_entrenamiento]),
        "validacion": set(pacientes[n_entrenamiento:n_entrenamiento + n_validacion]),
        "test": set(pacientes[n_entrenamiento + n_validacion:]),
    }
    return {nombre: [c for c in cortes if c.paciente in grupo]
            for nombre, grupo in grupos.items()}


# --- Lectura -----------------------------------------------------------------

def leer_imagen(ruta):
    """Imagen uint8 HxWx3 con los canales en el orden del dataset."""
    # OpenCV lee en BGR: se invierte para recuperar (pre, FLAIR, post)
    return cv2.cvtColor(cv2.imread(str(ruta), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def leer_mascara(ruta):
    """Máscara booleana HxW."""
    return cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE) > 127


# --- Preprocesamiento --------------------------------------------------------

def redimensionar(imagen, mascara, tamano):
    if imagen.shape[0] == tamano:
        return imagen, mascara
    imagen = cv2.resize(imagen, (tamano, tamano), interpolation=cv2.INTER_AREA)
    # Vecino más cercano: la máscara tiene que seguir siendo binaria
    mascara = cv2.resize(mascara.astype(np.uint8), (tamano, tamano),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return imagen, mascara


def normalizar(imagen):
    """
    Estandariza cada canal a media 0 y desvío 1, imagen por imagen.

    En resonancia la intensidad no tiene unidades físicas: depende del equipo y
    del protocolo de adquisición. Normalizar por imagen hace comparables los
    cortes de distintos hospitales. La media y el desvío se calculan sólo sobre
    la cabeza (FLAIR no oscuro), para que el fondo negro no los domine; después
    se aplican a la imagen entera.

    Hay cortes donde un canal es constante (desvío 0): dividir por el desvío
    daría valores de cientos de miles, que arruinan las estadísticas de
    BatchNorm. Esos canales quedan en 0, y todo se recorta a ±5 desvíos.
    """
    imagen = imagen.astype(np.float32) / 255.0
    cabeza = imagen[..., 1] > 0.08
    if cabeza.sum() < 50:  # corte casi vacío (extremos del volumen)
        return np.zeros_like(imagen)
    valores = imagen[cabeza]  # N x 3
    media, desvio = valores.mean(axis=0), valores.std(axis=0)
    normalizada = (imagen - media) / np.maximum(desvio, 1e-3)
    normalizada[..., desvio < 0.01] = 0
    return np.clip(normalizada, -5, 5)


def aumentar(imagen, mascara, rng):
    """
    Data augmentation: transformaciones plausibles de un corte de cerebro.

    - espejo horizontal (el cerebro es aproximadamente simétrico)
    - rotación de hasta ±15° y escala de 0.9 a 1.1 (posición de la cabeza)
    - cambio de brillo y contraste (variación entre equipos)

    Las transformaciones geométricas se aplican igual a imagen y máscara.
    """
    if rng.random() < 0.5:
        imagen = imagen[:, ::-1]
        mascara = mascara[:, ::-1]

    alto, ancho = mascara.shape
    angulo = rng.uniform(-15, 15)
    escala = rng.uniform(0.9, 1.1)
    matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, escala)
    imagen = cv2.warpAffine(np.ascontiguousarray(imagen), matriz, (ancho, alto),
                            flags=cv2.INTER_LINEAR, borderValue=0)
    mascara = cv2.warpAffine(np.ascontiguousarray(mascara).astype(np.uint8), matriz,
                             (ancho, alto), flags=cv2.INTER_NEAREST,
                             borderValue=0).astype(bool)

    contraste = rng.uniform(0.85, 1.15)
    brillo = rng.uniform(-15, 15)
    fondo = imagen.max(axis=2) == 0
    imagen = np.clip(imagen.astype(np.float32) * contraste + brillo, 0, 255)
    imagen[fondo] = 0  # el fondo sigue negro
    return imagen.astype(np.uint8), mascara


def preprocesar(imagen, mascara, tamano, rng=None):
    """Pipeline completo. Con rng aplica data augmentation; sin rng, no."""
    imagen, mascara = redimensionar(imagen, mascara, tamano)
    if rng is not None:
        imagen, mascara = aumentar(imagen, mascara, rng)
    return normalizar(imagen), mascara


class DatasetLGG(Dataset):
    """Dataset de PyTorch: devuelve (imagen 3xHxW float, máscara 1xHxW float)."""

    def __init__(self, cortes, tamano=256, aumentar=False):
        self.cortes = cortes
        self.tamano = tamano
        self.aumentar = aumentar
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.cortes)

    def __getitem__(self, i):
        corte = self.cortes[i]
        imagen, mascara = preprocesar(leer_imagen(corte.imagen),
                                      leer_mascara(corte.mascara),
                                      self.tamano,
                                      self._rng if self.aumentar else None)
        return (torch.from_numpy(imagen.transpose(2, 0, 1).copy()),
                torch.from_numpy(mascara[None].astype(np.float32)))


def inicializar_worker(id_worker):
    """Cada worker del DataLoader necesita su propio generador aleatorio."""
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(torch.initial_seed() % 2**32)
