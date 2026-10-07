"""
Descarga el dataset LGG MRI Segmentation de Kaggle y lo deja en datos/kaggle_3m.

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

El dataset es público: kagglehub lo baja sin credenciales (~700 MB). El zip
trae dos copias idénticas; sólo se usa kaggle_3m.
"""

import shutil
from pathlib import Path

import kagglehub

RAIZ_TP = Path(__file__).resolve().parent.parent
DESTINO = RAIZ_TP / "datos" / "kaggle_3m"


def main():
    if DESTINO.exists():
        print(f"Ya está descargado en {DESTINO}")
        return

    cache = Path(kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation"))
    print(f"Copiando a {DESTINO} ...")
    shutil.copytree(cache / "kaggle_3m", DESTINO)
    print("Listo.")


if __name__ == "__main__":
    main()
