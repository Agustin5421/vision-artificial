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
5. Descarte de contornos espurios, por dos criterios:
   - área mínima (`AREA_MINIMA = 500`);
   - contornos cortados por el borde del cuadro.

El filtro de borde importa más de lo que parece. Un contorno cortado por el
marco no describe al objeto sino a su intersección con la imagen, así que sus
invariantes de Hu no corresponden a ninguna forma real. Es además el caso
típico de una persona entrando en escena: el cuerpo siempre sale del cuadro por
algún lado.

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

El umbral por defecto sale de un percentil de las distancias entre muestras de
una misma clase, calculado en el entrenamiento, y se puede reajustar en vivo con
la trackbar `Distancia x10`. El percentil elegido es 90; el compromiso, medido
sobre este dataset y contra siluetas que el modelo nunca vio, está en la sección
de resultados.

## Resultados

Con 91 muestras (35 luna, 27 cuadrado, 29 estrella) y 25% reservado para prueba:

```
Comparación de clasificadores (validación cruzada de 5 particiones,
promediada sobre 5 semillas):
  DecisionTree     0.956 +/- 0.000
  RandomForest     0.958 +/- 0.005
  KNeighbors(3)    0.800 +/- 0.019
  SVC(rbf)         0.906 +/- 0.015

Exactitud sobre conjunto de prueba: 91.30%

Matriz de confusión
(filas = real, columnas = predicho, orden: luna, cuadrado, estrella)
[[8 0 1]
 [0 7 0]
 [1 0 6]]

Umbral de distancia (percentil 90 intra-clase): 1.059
```

El árbol de decisión y el random forest empatan; los otros dos quedan atrás. Se
guarda el árbol, que es además el que sugiere la consigna.

Las confusiones que quedan son entre luna y estrella. El cuadrado se separa sin
error: es la forma con los invariantes de Hu más distintos de las otras dos.

### Elección del umbral de rechazo

Medido contra siluetas que el modelo nunca vio (manos con distinta cantidad de
dedos y apertura, torsos con brazos, y formas orgánicas al azar, cada una
también en versión cortada por el borde):

| Percentil | Reconoce bien | Falso "desconocido" | Intrusos aceptados |
|-----------|---------------|---------------------|--------------------|
| 99 | 91.2% | 3.3% | 59.7% |
| 95 | 86.8% | 7.7% | 40.0% |
| **90** | **83.5%** | **12.1%** | **19.2%** |

Se elige 90: el problema a evitar es aceptar como figura algo que no lo es.

### Efecto del filtro de borde

| Intruso | Contornos | Descartados por borde | Aceptados |
|---------|-----------|------------------------|-----------|
| Mano | 200 | 0 | 46.5% |
| Torso | 400 | 50 | 28.0% |
| Forma orgánica | 200 | 0 | 41.0% |
| Cortado por el borde | 593 | 444 | 2.5% |
| **Total** | **1393** | **494** | **21.7%** |

Antes de estos ajustes el total aceptado era 43.0%.

### Límite conocido

Queda un 21.7% de siluetas ajenas aceptadas como figura conocida, casi todas
como "estrella". No es un defecto de implementación sino del descriptor, y
conviene tenerlo presente al mostrar el sistema.

Las tres figuras son simétricas o casi. Para el cuadrado y la estrella, los
invariantes de Hu de orden 2 en adelante valen **teóricamente cero**: lo que se
mide en ellos es el piso de ruido numérico del cálculo de momentos, no la
forma. Comparando una figura de referencia limpia contra las capturas de webcam
se ve directo:

| | figura de referencia | capturas de webcam |
|---|---|---|
| cuadrado `hu2,hu3,hu4` | 11.3, 9.72, 10.66 | 4.87, 4.95, 7.07 |
| estrella `hu2,hu3,hu4` | 6.65, 6.94, 7.84 | 4.03, 4.87, 4.60 |
| luna `hu2,hu3,hu4` | 0.93, 1.18, 1.96 | 0.81, 1.30, 1.84 |

La luna coincide, porque es asimétrica y sus invariantes son genuinamente
distintos de cero. El cuadrado y la estrella no coinciden: su valor depende del
ruido de captura. Separando la señal del ruido por clase (dispersión entre
clases dividida dispersión dentro de cada clase), `hu1..hu4` dan entre 1.7 y
2.6, y `hu5..hu7` dan 0.2.

El resultado es que en el espacio de Hu una mano cae encima de la estrella:

```
conocidos:  luna 0.24-0.35   estrella 0.66-0.68   cuadrado 0.77-0.78
intrusos:                    mano 0.67-0.74       forma orgánica 0.73-0.79
```

Tres caminos para mejorarlo, todos dentro de la consigna:

1. **Recortar una región de interés.** La consigna la contempla como parte del
   ambiente controlado: si sólo se procesa el rectángulo donde se apoyan las
   figuras, una persona que pasa por detrás no entra en el proceso. Es la
   solución más efectiva para este caso puntual.
2. **Más muestras y más variadas**, que es lo que pide la consigna. Las 91
   actuales son ráfagas del mismo objeto quieto: los 27 cuadrados tienen `hu1`
   entre 0.7767 y 0.7795. Capturar cada figura en distintas rotaciones,
   distancias y posiciones ensancha lo que el modelo reconoce sin aflojar el
   umbral.
3. **Bajar el umbral con la trackbar** durante la demostración, aceptando más
   falsos "desconocido" a cambio de menos falsos aciertos.

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
