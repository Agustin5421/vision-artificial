# TP4 — Segmentación de tumores cerebrales con U-Net

Entrenamiento completo de una U-Net, implementada desde cero en PyTorch, para
segmentar gliomas de bajo grado en cortes de resonancia magnética cerebral.

## Temática

**Segmentación de tumores cerebrales en resonancia magnética.** Los gliomas de
bajo grado son tumores que nacen en las células de soporte del cerebro (la
glía). Crecen lento y sus bordes son difusos, así que delimitarlos a mano, corte
por corte, es un trabajo lento y que varía según quién lo haga. Esa delimitación
se usa para planificar cirugías y radioterapia y para seguir cómo evoluciona el
tumor.

El objetivo es que, dado un corte de resonancia, el modelo marque píxel a píxel
dónde está la anomalía visible en la secuencia FLAIR, donde el tumor y el edema
que lo rodea se ven más brillantes. Es segmentación semántica binaria:
tumor / no tumor.

## Cómo probarlo

Todo se corre desde la carpeta `tp4-segmentacion`, con el `venv` de la raíz del
repo. En Git Bash:

```bash
cd tp4-segmentacion

# 0. Dependencias (una sola vez). PyTorch con CUDA para usar la GPU:
../venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
../venv/Scripts/python.exe -m pip install -r requirements.txt

# 1. Bajar el dataset (~700 MB, no hace falta cuenta de Kaggle)
../venv/Scripts/python.exe src/descargar_datos.py

# 2. Figuras del dataset y del preprocesamiento
../venv/Scripts/python.exe src/visualizar_datos.py

# 3. Entrenar (~30 min en una RTX 3060 Laptop; en CPU, horas)
../venv/Scripts/python.exe src/entrenar.py --epocas 40

# 4. Evaluar sobre test y generar las figuras de predicciones
../venv/Scripts/python.exe src/evaluar.py
```

En PowerShell es lo mismo con `..\venv\Scripts\python.exe`.

Pruebas rápidas sin entrenar todo:

```bash
../venv/Scripts/python.exe src/unet.py                  # arma la red e imprime parámetros y forma de salida
../venv/Scripts/python.exe src/entrenar.py --epocas 2   # entrenamiento corto, para ver que todo corre
../venv/Scripts/python.exe src/evaluar.py               # funciona con cualquier modelo.pt guardado
```

`entrenar.py` acepta además `--tamano 128` (más rápido, menos detalle),
`--batch`, `--lr`, `--base` (filtros del primer nivel) y `--workers` (poner 0
si el DataLoader da problemas en Windows).

Todo lo que se genera queda en `resultados/`:

| Archivo | Lo genera | Qué es |
|---|---|---|
| `ejemplos_originales.png` | visualizar_datos | los 3 canales de RM de algunos cortes y su máscara |
| `preprocesamiento.png` | visualizar_datos | original → normalizada → 3 versiones aumentadas |
| `estadisticas.png` | visualizar_datos | cortes con/sin tumor por conjunto, tamaño de los tumores |
| `particion.txt` | entrenar | qué pacientes quedaron en entrenamiento / validación / test |
| `historial.csv`, `curvas.png` | entrenar | pérdida y Dice por época |
| `entrenamiento.log` | (redirección de la salida) | la salida de consola del entrenamiento |
| `modelo.pt` | entrenar | pesos de la mejor época (no se sube al repo) |
| `metricas_test.txt` | evaluar | Dice, IoU y detección sobre test |
| `predicciones_*.png` | evaluar | imagen, máscara real, predicción y superposición |
| `dice_por_paciente.png` | evaluar | Dice volumétrico de cada paciente de test |

## 1. El problema y el dataset

**LGG MRI Segmentation** (Buda et al., 2019), en Kaggle:
<https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>

- 110 pacientes con glioma de bajo grado (The Cancer Imaging Archive).
- 3.929 cortes axiales de 256×256, en `.tif`.
- Cada corte tiene **3 canales**, cada uno una secuencia de resonancia distinta:
  pre-contraste, FLAIR y post-contraste. Cuando falta una secuencia, el dataset
  la reemplaza por FLAIR.
- Cada corte tiene una **máscara binaria** de la anomalía visible en FLAIR,
  dibujada a mano.
- Sólo el 35% de los cortes tiene tumor. En los que lo tienen, ocupa una mediana
  del 2,3% de la imagen. Las clases están muy desbalanceadas.

Ver `resultados/ejemplos_originales.png` y `resultados/estadisticas.png`.

Detalle técnico: OpenCV lee los `.tif` como BGR, así que `leer_imagen` invierte
los canales para recuperar el orden (pre, FLAIR, post).

## 2. Preprocesamiento — `src/datos.py`

1. **Redimensionamiento.** Las imágenes ya son de 256×256, que es el tamaño por
   defecto; con `--tamano 128` se reducen. La imagen se reduce con `INTER_AREA`
   y la máscara con vecino más cercano, para que siga siendo binaria.
2. **Normalización por imagen y por canal** (media 0, desvío 1). En resonancia
   la intensidad no tiene unidades físicas: depende del equipo y del protocolo.
   Normalizar cada imagen hace comparables los cortes de distintos hospitales.
   La media y el desvío se calculan sólo sobre la cabeza (FLAIR > 8% del
   máximo), para que el fondo negro no los domine.
3. **Data augmentation**, sólo en entrenamiento y distinto en cada época:
   - espejo horizontal (el cerebro es aproximadamente simétrico),
   - rotación de ±15° y escala de 0,9 a 1,1 (posición de la cabeza),
   - brillo y contraste (variación entre equipos).

   Las transformaciones geométricas se aplican igual a la imagen y a la máscara.

Ver `resultados/preprocesamiento.png`.

### Un problema que apareció: canales constantes

En la primera corrida, el Dice de validación quedó **exactamente en 0** durante
6 épocas, mientras el de entrenamiento subía a 0,67. Al diagnosticarlo:

- El modelo segmentaba bien en modo `train()` y no predecía nada en modo
  `eval()`, incluso sobre imágenes de entrenamiento. El problema no eran los
  datos de validación sino **BatchNorm**, que en `eval()` usa las estadísticas
  acumuladas durante el entrenamiento.
- La primera capa BatchNorm tenía una varianza acumulada de ~68.000, contra ~1
  en un batch normal.
- La causa: en 23 cortes el canal post-contraste es **constante**
  (desvío 0). Dividir por ese desvío producía valores de hasta 200.000, y bastaba
  uno de esos cortes en un batch para arruinar las estadísticas.

La solución: un canal constante queda en 0, y el resultado se recorta a ±5
desvíos. Es un buen ejemplo de por qué conviene revisar los extremos de los
datos después de normalizar.

## 3. División de los datos

70% / 15% / 15% **por paciente**, no por corte (`dividir_por_paciente`):

| Conjunto | Pacientes | Cortes | Con tumor |
|---|---|---|---|
| entrenamiento | 77 | 2.719 | 34% |
| validación | 16 | 621 | 34% |
| test | 17 | 589 | 39% |


Si se dividiera por corte, cortes vecinos del mismo cerebro, casi idénticos,
caerían en entrenamiento y en test, y el resultado de test saldría inflado
(*data leakage*). La semilla es fija (42), así que la partición es siempre la
misma y `evaluar.py` la reconstruye sin guardar nada.

- **Entrenamiento**: ajusta los pesos.
- **Validación**: elige la mejor época y ajusta el learning rate.
- **Test**: se usa una sola vez, al final, en `evaluar.py`.

## 4. El modelo — `src/unet.py`

U-Net clásica (Ronneberger et al., 2015), escrita desde cero:

- **Encoder**: 4 niveles de (conv 3×3 → BatchNorm → ReLU) × 2 + max pooling, con
  32, 64, 128 y 256 filtros.
- **Cuello**: 512 filtros, a 1/16 de la resolución.
- **Decoder**: convolución transpuesta para subir la resolución, concatenación
  con el mapa del encoder del mismo nivel (*skip connection*) y bloque doble.
- **Salida**: conv 1×1 → 1 canal (logit de "tumor").
- 7,8 M de parámetros.

Diferencias con el paper original: convoluciones con padding (la salida tiene el
mismo tamaño que la entrada) y BatchNorm.

## 5. Entrenamiento — `src/entrenar.py`, `src/metricas.py`

- **Pérdida: BCE + Dice.** Con sólo entropía cruzada, predecir "todo fondo" da
  una pérdida baja, porque el tumor es ~1% de los píxeles. El término Dice mide
  la superposición con el tumor y no se deja engañar por el fondo.
- **Optimizador**: Adam, lr 1e-3, batch 16. `ReduceLROnPlateau` divide el lr por
  2 si el Dice de validación no mejora en 5 épocas.
- **Precisión mixta** (`torch.autocast`) en GPU, para entrenar más rápido.
- Se guarda el modelo de la época con mejor **Dice de validación**. Ese Dice es
  "global": se suman todos los píxeles del conjunto, como si fuera un volumen.

Ver `resultados/curvas.png`.

## 6. Evaluación — `src/evaluar.py`

Sobre los pacientes de test, con umbral 0,5:

- **Dice por corte**, sobre todos los cortes y sobre los que tienen tumor. Si un
  corte no tiene tumor y el modelo no predice nada, el Dice es 1.
- **IoU** en los cortes con tumor.
- **Dice volumétrico por paciente**: se apilan todos los cortes del paciente y
  se calcula un solo Dice 3D. Es la métrica más usada en este dataset.
- **Detección**: qué fracción de cortes con tumor tiene alguna predicción, y
  cuántos cortes sin tumor tienen falsos positivos.

Figuras: las 6 mejores y las 6 peores predicciones, y 6 cortes al azar. En la
superposición, verde es acierto, rojo es falso positivo y azul es falso
negativo.

## 7. Resultados

_(se completa al terminar el entrenamiento)_

## Archivos

```
tp4-segmentacion/
├── README.md
├── requirements.txt
├── src/
│   ├── descargar_datos.py    baja el dataset a datos/kaggle_3m
│   ├── datos.py              lectura, partición por paciente, preprocesamiento, Dataset
│   ├── unet.py               la U-Net
│   ├── metricas.py           pérdida BCE+Dice, Dice, IoU
│   ├── visualizar_datos.py   figuras del dataset y del preprocesamiento
│   ├── entrenar.py           entrenamiento
│   └── evaluar.py            evaluación sobre test
├── datos/                    el dataset (no se sube al repo)
└── resultados/               figuras, métricas y modelo
```
