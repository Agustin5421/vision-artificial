"""
Entrenador.

Lee el dataset de invariantes de Hu (dataset/dataset.csv), entrena
un clasificador (árbol de decisión por defecto), reporta exactitud
y matriz de confusión sobre un conjunto de prueba, y guarda el
modelo entrenado en modelos/modelo.joblib para que lo use
clasificador.py.

Junto al clasificador se guardan las estadísticas de normalización,
las muestras de entrenamiento y un umbral de distancia. Con eso
clasificador.py decide si un contorno se parece lo suficiente a algo
conocido o hay que marcarlo como desconocido; el árbol por sí solo no
alcanza, porque siempre devuelve alguna de las clases que conoce.

No usa la webcam: opera exclusivamente sobre el dataset.
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

import comun

RUTA_DATASET = comun.RAIZ_PROYECTO / "dataset" / "dataset.csv"
RUTA_MODELO = comun.RAIZ_PROYECTO / "modelos" / "modelo.joblib"

# Percentil de las distancias intra-clase que se toma como umbral por
# defecto: deja pasar a casi todas las muestras genuinas y rechaza lo
# que cae notoriamente más lejos. Se puede reajustar en vivo con la
# trackbar de clasificador.py.
PERCENTIL_UMBRAL = 95


def normalizar(X, media, desvio):
    """
    Lleva los descriptores a media 0 y desvío 1 por componente.

    Los 7 invariantes de Hu tienen escalas muy distintas entre sí. Sin
    normalizar, la distancia euclídea queda dominada por las componentes
    de mayor magnitud y las demás casi no influyen.
    """
    return (X - media) / desvio


def calcular_umbral_distancia(X_norm, y):
    """
    Calcula cuán separadas están las muestras de una misma clase.

    Para cada muestra busca la más parecida de su propia clase y devuelve
    el percentil PERCENTIL_UMBRAL de esas distancias. Un contorno que
    quede más lejos que eso de todo lo conocido es candidato a desconocido.
    """
    distancias = []
    for etiqueta in np.unique(y):
        grupo = X_norm[y == etiqueta]
        if len(grupo) < 2:
            continue
        for i, muestra in enumerate(grupo):
            otras = np.delete(grupo, i, axis=0)
            distancias.append(np.linalg.norm(otras - muestra, axis=1).min())
    return float(np.percentile(distancias, PERCENTIL_UMBRAL))


def main():
    # --- Cargar dataset ---
    df = pd.read_csv(RUTA_DATASET)
    columnas_hu = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]

    X = df[columnas_hu].values
    y = df["etiqueta"].values

    print(f"Dataset cargado: {len(df)} muestras")
    print("Distribución por etiqueta:")
    print(df["etiqueta"].value_counts().sort_index())

    # --- Separar train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Entrenar ---
    clasificador = tree.DecisionTreeClassifier(random_state=42)
    clasificador.fit(X_train, y_train)

    # --- Evaluar ---
    y_pred = clasificador.predict(X_test)
    exactitud = accuracy_score(y_test, y_pred)

    print(f"\nExactitud sobre conjunto de prueba: {exactitud:.2%}")
    print("\nMatriz de confusión:")
    print("(filas = real, columnas = predicho, orden de etiquetas:",
          sorted(comun.ETIQUETAS.keys()), ")")
    print(confusion_matrix(y_test, y_pred, labels=sorted(comun.ETIQUETAS.keys())))

    print("\nReporte de clasificación:")
    nombres = [comun.ETIQUETAS[e] for e in sorted(comun.ETIQUETAS.keys())]
    print(classification_report(y_test, y_pred,
                                 labels=sorted(comun.ETIQUETAS.keys()),
                                 target_names=nombres))

    # --- Preparar el rechazo de formas desconocidas ---
    media = X_train.mean(axis=0)
    desvio = X_train.std(axis=0)
    desvio[desvio == 0] = 1.0  # una componente constante no aporta distancia

    X_train_norm = normalizar(X_train, media, desvio)
    umbral_distancia = calcular_umbral_distancia(X_train_norm, y_train)

    print(f"\nUmbral de distancia (percentil {PERCENTIL_UMBRAL} "
          f"intra-clase): {umbral_distancia:.3f}")
    print("Un contorno más lejano que esto de toda muestra conocida "
          "se marca como desconocido.")

    # --- Guardar modelo ---
    modelo = {
        "clasificador": clasificador,
        "media": media,
        "desvio": desvio,
        "muestras": X_train_norm,
        "etiquetas": y_train,
        "umbral_distancia": umbral_distancia,
    }

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    dump(modelo, RUTA_MODELO)
    print(f"\nModelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
