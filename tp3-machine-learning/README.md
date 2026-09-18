# TP3 — Localización homográfica

Demostración de localización robótica en 2D en tiempo real. Una cámara fija
observa en perspectiva un plano por el que se desplaza un marcador Aruco, y el
sistema determina y visualiza su pose 2D —coordenadas en mm y orientación— en un
sistema de referencia métrico registrado sobre ese plano.

Todo se resuelve con una homografía de plano a vista: como el movimiento del
marcador está confinado a un plano, no hace falta calibrar la cámara ni estimar
una pose 3D.

## El mundo

El mundo es un plano con un marcador Aruco encima. Por ejemplo, un escritorio
con un marcador de 100 mm de lado impreso en papel, o un monitor mostrando un
marcador móvil. La cámara mira ese plano desde un costado, en perspectiva, y no
se mueve: si se mueve, hay que volver a registrar.

## Ventanas

**Cam** — el feed de la cámara, anotando el contorno de cada marcador Aruco
detectado y su etiqueta. El punto magenta marca la esquina 0 del marcador, que
es de donde sale la orientación.

**W2D** — la vista cenital del mundo 2D:

- flecha que expresa la localización y orientación del marcador
- contorno cuadrado del marcador
- etiquetas con las coordenadas en mm y el ángulo en grados
- ejes x, y del sistema de referencia, centrados en la imagen, x horizontal
  hacia la derecha e y vertical hacia arriba
- de fondo, la vista cenital **no viva**: la imagen rectificada que se capturó
  en el momento del registro
- una grilla cada 50 mm, para leer distancias de un vistazo

## Las dos operaciones

### Registro del plano métrico

Es instantáneo y se dispara con la tecla `r`. Toma el marcador Aruco que haya en
escena y lo usa como sistema de referencia métrico del mundo 2D: su centro pasa
a ser el origen, su lado en mm fija la escala y su eje propio fija la dirección
de x. Si no hay ningún marcador detectado, el registro no se lleva a cabo. Si
hay varios, se elige el de id más chico (el criterio es indistinto; éste es
reproducible).

En ese momento se determinan las dos homografías:

| Homografía  | De → a                    | Para qué |
|-------------|---------------------------|----------|
| `H_img_mm`  | imagen (px) → mundo (mm)  | el resultado buscado: las coordenadas métricas |
| `H_img_vis` | imagen (px) → vista W2D   | anotación: transporta puntos de la cámara a la vista cenital |

La segunda se arma componiendo la primera con una semejanza `H_mm_vis` de mm a
píxeles de la ventana, que ubica el origen en el centro de la imagen e invierte
el eje vertical (en el mundo y crece hacia arriba, en la imagen hacia abajo).

También en ese momento se produce la vista cenital de fondo, con una
transformación de perspectiva del frame actual. Esa rectificación se hace una
sola vez, acá, y **no** en el bucle de cámara.

### Localización

Corre en bucle, en tiempo real, y actualiza las dos ventanas en cada frame.
Para cada marcador detectado se transportan sus cuatro esquinas al mundo con
`H_img_mm`, y de ahí salen:

- **posición**: el promedio de las cuatro esquinas ya en mm. El promedio se hace
  en el mundo y no en la imagen, porque la perspectiva no conserva el punto medio.
- **orientación**: la dirección del eje x propio del marcador, o sea del punto
  medio del lado izquierdo al del lado derecho. Medido desde el eje x del mundo,
  positivo antihorario.

Con esa convención, el marcador que registró el plano queda en la pose
(0, 0, 0°) en el instante del registro.

La ventana W2D no se puede dibujar antes del registro: sin homografías no hay ni
fondo ni ejes. Hasta entonces muestra la instrucción. Ya registrado, se sigue
dibujando aunque no haya marcadores en escena, porque el fondo y los ejes quedan
determinados.

## Requisitos

```
pip install -r requirements.txt
```

## Uso

### 1. Generar un marcador (opcional)

```
python src/generar_marcador.py --id 0 --lado-mm 100 --dpi 96
```

Genera el PNG del marcador con el tamaño físico pedido, para imprimir o mostrar
en pantalla. Lo importante es que el lado real coincida con el que después se le
declara al sistema con `--lado-mm`: de ese número sale toda la escala métrica.

### 2. Localizar

```
python src/main.py --lado-mm 100
```

1. Apuntar la cámara al plano, en perspectiva, y dejarla fija.
2. Poner el marcador de referencia donde se quiera el origen del mundo.
3. Presionar `r` para registrar el plano.
4. Mover el marcador: la ventana W2D muestra su pose en tiempo real.
5. `ESC` para salir.

### Opciones

| Opción          | Default  | Qué hace |
|-----------------|----------|----------|
| `--camara`      | `0`      | índice de la cámara |
| `--lado-mm`     | `100`    | lado real del marcador, en mm |
| `--diccionario` | `auto`   | diccionario Aruco. Por defecto lo busca solo; si se quiere fijar, va el nombre corto (`4x4_50`, `6x6_250`, `aruco_original`, ...) |
| `--ancho-w2d`   | `720`    | ancho de la ventana W2D, en px |
| `--alto-w2d`    | `720`    | alto de la ventana W2D, en px |
| `--escala`      | `1.2`    | píxeles por mm en la ventana W2D |

`--escala` define cuánto mundo entra en W2D: con los valores por defecto, la
ventana cubre ±300 mm alrededor del origen. Bajarla es alejarse.

## El diccionario del marcador

Un marcador sólo se detecta con el diccionario al que pertenece, y los marcadores
que uno se baja de internet casi nunca son del que uno supone: pueden ser de
4x4, de 6x6, `aruco_original`, etc.

Por eso el default de `--diccionario` es `auto`: el detector prueba todos los
diccionarios Aruco que trae OpenCV hasta que alguno reconozca un marcador, y a
partir de ahí se queda con ése. La búsqueda cuesta dieciocho detecciones por frame,
así que sólo se paga hasta encontrarlo; el nombre encontrado aparece abajo en la
ventana `Cam` y en la terminal, y se le puede pasar después con `--diccionario`
para saltear la búsqueda.

Un mismo marcador suele dar positivo en varios diccionarios emparentados (4x4_50
está contenido en 4x4_100, y así). Cualquiera de ellos sirve: la geometría de las
esquinas, que es lo único que usa la localización, es la misma.

El diccionario fijado no es definitivo: si pasan 15 frames seguidos sin detectar
nada, se vuelve a barrer. Sin eso, un falso positivo sobre un frame con ruido
dejaría al detector fijado en un diccionario equivocado y no encontraría el
marcador nunca más.

## Si aun así el marcador no se detecta

Cuando en la ventana `Cam` no aparece ningún contorno verde:

```
python src/diagnostico.py
```

Prueba todos los diccionarios de OpenCV sobre cada frame y reporta todos los que
reconocen al marcador, no sólo el primero. Dibuja en rojo los candidatos rechazados, los cuadriláteros que encontró
pero no pudo decodificar, y con eso se distinguen los tres casos:

| Lo que se ve | Qué pasa |
|--------------|----------|
| contorno verde | se reconoce: el problema no es el marcador |
| contorno rojo | ve el cuadrado pero no lee el código: diccionario equivocado, imagen borrosa o movida, marcador muy chico o muy oblicuo |
| nada | no distingue ni el cuadrado: falta contraste, hay reflejo sobre el papel, o falta margen blanco alrededor |

## Estructura

```
src/deteccion.py         detección de marcadores Aruco y anotación de la ventana Cam
src/registro.py          registro del plano métrico: las dos homografías y el fondo cenital
src/localizacion.py      pose 2D de un marcador a partir de la homografía imagen → mm
src/vista_cenital.py     dibujo de la ventana W2D
src/main.py              bucle de cámara, ventanas y teclas
src/generar_marcador.py  utilidad para generar marcadores imprimibles
src/diagnostico.py       utilidad para averiguar por qué no se detecta un marcador
tests/                   tests de la localización con una cámara sintética
```

## Tests

```
python -m pytest tests
```

Los tests no necesitan webcam: arman a mano una homografía plano → imagen que
hace de cámara en perspectiva, proyectan con ella marcadores de pose conocida y
verifican que el registro y la localización devuelvan esa pose.

## Consejos de uso

- Dejar un margen blanco alrededor del marcador: sin zona de silencio el
  detector no lo encuentra. Cuidado también con que dos marcadores se pisen entre
  sí, porque el borde blanco de uno puede comerle el contorno negro al otro.
- Cuanto más rasante la mirada de la cámara, peor condicionada queda la
  homografía y más ruidosas las coordenadas lejos del origen.
- El registro se puede repetir en cualquier momento con `r`: es la manera de
  recuperarse si se movió la cámara o si se quiere cambiar el origen de lugar.

## Referencias

- [`perspectiveTransform`](https://docs.opencv.org/5.0/main_modules/core_array.html#perspectivetransform)
- [Transformaciones geométricas](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#perspective-transformation)
