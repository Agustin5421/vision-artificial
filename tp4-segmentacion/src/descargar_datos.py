# Descarga el dataset de Kaggle (no hace falta cuenta) y lo copia a datos/kaggle_3m
# https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

import shutil
from pathlib import Path

import kagglehub

DESTINO = Path(__file__).resolve().parent.parent / "datos" / "kaggle_3m"

if DESTINO.exists():
    print("Ya está descargado en", DESTINO)
else:
    ruta = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    # el zip trae dos copias iguales, uso kaggle_3m
    shutil.copytree(Path(ruta) / "kaggle_3m", DESTINO)
    print("Descargado en", DESTINO)
