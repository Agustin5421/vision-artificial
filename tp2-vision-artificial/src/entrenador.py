"""
Entrenador.

Lee el dataset de invariantes de Hu (dataset/dataset.csv), entrena
un clasificador (árbol de decisión por defecto), reporta exactitud
y matriz de confusión sobre un conjunto de prueba, y guarda el
modelo entrenado en modelos/modelo.joblib para que lo use
clasificador.py.

No usa la webcam: opera exclusivamente sobre el dataset.
"""

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

import comun

RUTA_DATASET = "../dataset/dataset.csv"
RUTA_MODELO = "../modelos/modelo.joblib"


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

    # --- Guardar modelo ---
    dump(clasificador, RUTA_MODELO)
    print(f"\nModelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()