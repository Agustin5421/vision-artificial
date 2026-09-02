# TP2 — Detección y clasificación de formas

Detección y clasificación de contornos en tiempo real desde la webcam, en un
ambiente controlado. El sistema encuentra los contornos de la imagen, calcula
los invariantes de Hu de cada uno y los clasifica con un árbol de decisión
entrenado previamente, marcando como desconocidas las formas que no se parecen
a nada conocido.

## Objetos

| Etiqueta | Nombre   |
|----------|----------|
| 1        | luna     |
| 2        | cuadrado |
| 3        | estrella |

Definidos en `ETIQUETAS`, en `src/comun.py`.

## Ambiente controlado

- Fondo liso, de color uniforme y contrastante con los objetos.
- Iluminación pareja, sin sombras marcadas ni degradado sobre el fondo.
- Cámara fija y perpendicular al plano de la escena.
- La escena ocupa la imagen completa: no se recorta región de interés.

## Requisitos

```
pip install -r requirements.txt
```

Los scripts se pueden correr desde cualquier directorio: las rutas al dataset y
al modelo se arman a partir de la ubicación del propio archivo, no del directorio
actual. Los ejemplos de abajo asumen que estás parado en la raíz del proyecto.

## Uso

### 1. Generar descriptores

```
python src/generador_descriptores.py
```

Mostrar el objeto a la cámara y ajustar las trackbars **Umbral** y **Kernel**
hasta que el contorno se vea limpio en la ventana "Binaria".

- `ESPACIO` imprime en la terminal los invariantes de Hu de cada contorno detectado.
- `ESC` sale.

Los valores impresos se copian a `dataset/dataset.csv`, agregando a mano la
columna `etiqueta` con el número del objeto.

### 2. Entrenar

```
python src/entrenador.py
```

Lee `dataset/dataset.csv`, entrena el árbol, imprime exactitud y matriz de
confusión, y guarda `modelos/modelo.joblib`. No usa la webcam.

El modelo no se versiona (está en `.gitignore`), así que hay que correr este
paso al menos una vez antes de clasificar.

### 3. Clasificar

```
python src/clasificador.py
```

Muestra el video anotado en tiempo real:

- **Verde**: objeto reconocido, con su nombre y la distancia a la muestra
  conocida más parecida.
- **Rojo**: forma que no se parece lo suficiente a nada conocido.

Trackbars disponibles:

| Trackbar        | Qué ajusta                                                  |
|-----------------|-------------------------------------------------------------|
| `Umbral`        | Nivel de binarización de la imagen en escala de grises.      |
| `Kernel`        | Tamaño del elemento estructural de las operaciones morfológicas. |
| `Distancia x10` | Umbral de validez, dividido 10. Más alto = más permisivo.    |

`ESC` sale.

## Cómo funciona

### Pipeline de detección (`comun.py`)

1. Conversión del frame a escala de grises.
2. Binarización con umbral fijo ajustable por trackbar.
3. Apertura + cierre morfológico para eliminar ruido y rellenar huecos.
4. Búsqueda de todos los contornos externos.
5. Descarte de contornos espurios por área mínima (`AREA_MINIMA = 500`).

### Descriptor

De cada contorno se calculan los 7 invariantes de Hu y se les aplica una
transformación logarítmica con signo, para llevarlos a un rango numérico
manejable sin perder el signo original. Los invariantes de Hu no cambian ante
traslación, rotación ni escala, así que el objeto se reconoce en cualquier
posición del cuadro.

El cálculo vive en una sola función (`comun.calcular_descriptor`) que usan tanto
el generador como el clasificador: si se modifica, el modelo entrenado deja de
ser válido y hay que reentrenar.

### Clasificación y formas desconocidas

El árbol de decisión responde *qué* clase es, pero no sirve para decidir *si*
la forma es conocida: sus hojas quedan puras, así que `predict_proba()` devuelve
1.0 para cualquier entrada, incluso para un objeto que nunca vio.

Por eso la validez se decide por distancia. El descriptor se normaliza a media 0
y desvío 1 por componente (los invariantes de Hu tienen escalas muy distintas
entre sí, y sin normalizar la distancia euclídea queda dominada por unas pocas
componentes) y se mide cuán lejos está de la muestra de entrenamiento más
parecida de la clase predicha. Si esa distancia supera el umbral, la forma es
desconocida.

El umbral por defecto sale del percentil 95 de las distancias entre muestras de
una misma clase, calculado en el entrenamiento, y se puede reajustar en vivo con
la trackbar `Distancia x10`.

## Resultados

Con 91 muestras (35 luna, 27 cuadrado, 29 estrella) y 25% reservado para prueba:

```
Exactitud sobre conjunto de prueba: 91.30%

Matriz de confusión
(filas = real, columnas = predicho, orden: luna, cuadrado, estrella)
[[8 0 1]
 [0 7 0]
 [1 0 6]]

Umbral de distancia (percentil 95 intra-clase): 1.686
```

Las confusiones que quedan son entre luna y estrella. El cuadrado se separa sin
error, que es lo esperable: es la forma con los invariantes de Hu más distintos
de las otras dos.

## Decisiones de diseño

La consigna sugiere comparar cada contorno contra imágenes de referencia con
`cv2.matchShapes()`. Acá se usa en cambio un clasificador entrenado sobre los
invariantes de Hu de varias muestras por objeto.

`matchShapes()` compara contra una única imagen de referencia por objeto, así
que la decisión depende por completo de qué tan representativa sea esa captura.
Tomar varias muestras de cada objeto, en distintas posiciones y rotaciones, hace
que el criterio cubra la variación real con la que aparece cada forma frente a
la cámara.

El umbral de distancia máxima de validez que pide la consigna se mantiene: es la
trackbar `Distancia x10`, aplicada sobre la distancia en el espacio de
descriptores en lugar de sobre la métrica de `matchShapes()`.

## Estructura

```
tp2-vision-artificial/
├── dataset/
│   └── dataset.csv                 muestras: 7 invariantes de Hu + etiqueta
├── modelos/                        se crea al entrenar (no versionado)
│   └── modelo.joblib
├── src/
│   ├── comun.py                    pipeline de detección y descriptor
│   ├── generador_descriptores.py   captura de muestras para el dataset
│   ├── entrenador.py               entrenamiento y evaluación
│   └── clasificador.py             clasificación en tiempo real
└── requirements.txt
```
