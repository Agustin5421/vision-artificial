# TP3 — Documentación interna

Este documento explica **qué hace cada archivo y cada función** del proyecto, y
cómo se encadenan entre sí. El `README.md` cuenta *qué es* el sistema y *cómo
usarlo*; esto cuenta *cómo está hecho por dentro*.

---

## 1. La idea en un párrafo

Una cámara fija mira en perspectiva un plano (un escritorio, un monitor) por el
que se mueve un marcador Aruco. Como el movimiento está confinado a ese plano,
la relación entre "lo que ve la cámara" y "el mundo real en milímetros" es una
**homografía**: una matriz 3×3 que se calcula una sola vez, en el momento del
*registro*, a partir de las cuatro esquinas de un marcador cuyo lado real
conocemos. Después, localizar consiste sólo en multiplicar las esquinas
detectadas por esa matriz.

No hace falta calibrar la cámara ni estimar pose 3D. Ese es el atajo que hace
que todo el proyecto quepa en siete archivos cortos.

---

## 2. Mapa de archivos

| Archivo | Rol | ¿Lo usa el programa principal? |
|---|---|---|
| `src/main.py` | Orquestador: abre la cámara, corre el bucle, maneja teclas y ventanas | es el punto de entrada |
| `src/deteccion.py` | Encuentra marcadores Aruco en un frame y anota la ventana `Cam` | sí |
| `src/registro.py` | Calcula las homografías y el fondo cenital (la tecla `r`) | sí |
| `src/localizacion.py` | Convierte esquinas en píxeles → pose en mm y grados | sí |
| `src/vista_cenital.py` | Dibuja la ventana `W2D` | sí |
| `src/generar_marcador.py` | Utilidad suelta: genera el PNG de un marcador para imprimir | no, se corre aparte |
| `src/diagnostico.py` | Utilidad suelta: averigua por qué no se detecta un marcador | no, se corre aparte |
| `tests/conftest.py` | Hace importable `src/` desde los tests | sólo pytest |
| `tests/test_localizacion.py` | Tests con una cámara sintética, sin webcam | sólo pytest |
| `requirements.txt` | Dependencias: `opencv-python`, `numpy` | — |

### Quién importa a quién

```
main.py
 ├── deteccion.py      (Detector, dibujar_detecciones)
 ├── registro.py       (registrar_plano)
 ├── localizacion.py   (localizar)
 └── vista_cenital.py  (dibujar_w2d, lienzo_sin_registro, texto)
                        └── localizacion.py (transformar_puntos)

generar_marcador.py ── deteccion.py (sólo la tabla DICCIONARIOS)
diagnostico.py      ── nada del proyecto (es autónomo)
```

Las dependencias van en una sola dirección y no hay ciclos. `localizacion.py` y
`registro.py` no importan nada del proyecto: son los módulos "de abajo", puro
cálculo, y por eso son los que se pueden testear sin cámara.

---

## 3. El recorrido de un cuadro

Lo que pasa en cada vuelta del bucle, en orden:

```
cámara ──frame(px)──> deteccion.Detector.detectar()
                            │
                            ├──> esquinas (4,2) px + ids
                            │
              ┌─────────────┴──────────────┐
              │                            │
   dibujar_detecciones()      localizacion.localizar(esquinas, ids, H_img_mm)
        ventana Cam                        │
                                           └──> lista de Pose (x mm, y mm, áng)
                                                       │
                                     vista_cenital.dibujar_w2d(registro, poses)
                                                  ventana W2D
```

Y aparte, una sola vez cuando se aprieta `r`:

```
frame + esquinas + ids ──> registro.registrar_plano() ──> Registro
                                                           ├── H_img_mm
                                                           ├── H_mm_vis
                                                           ├── H_img_vis
                                                           └── fondo (rectificado)
```

El objeto `Registro` es el estado que el bucle guarda entre cuadros. Mientras
vale `None`, la ventana `W2D` sólo muestra el cartel de "presionar r".

---

## 4. Archivo por archivo

### 4.1 `src/deteccion.py` — detección de marcadores

**Para qué existe.** Envuelve `cv2.aruco` por dos motivos: (a) la API de OpenCV
cambió en la 4.7 (antes el detector era una función suelta, ahora es un objeto)
y no queremos que ese detalle contamine el resto del programa; (b) resuelve solo
*a qué diccionario pertenece* el marcador que uno tiene a mano, que casi nunca
es el que uno supone.

**Constantes y tablas**

- `AUTOMATICO = "auto"` — el valor centinela de `--diccionario`.
- `COLOR_CONTORNO`, `COLOR_ETIQUETA`, `COLOR_PRIMERA_ESQUINA` — colores BGR de
  la anotación sobre la ventana `Cam`.
- `_diccionarios_predefinidos()` — arma por **introspección** (`dir(cv2.aruco)`)
  la tabla `{"4x4_50": cv2.aruco.DICT_4X4_50, ...}`. Se hace así y no a mano
  para no dejar afuera ningún diccionario de la versión instalada de OpenCV: el
  marcador bajado de internet bien puede ser un AprilTag.
- `DICCIONARIOS` — el resultado de lo anterior, calculado al importar.
- `ORDEN_BUSQUEDA` — los nombres ordenados alfabéticamente, con `apriltag_16h5`
  forzado al final: tiene poca redundancia y es famoso por reconocer ruido, así
  que si otro diccionario también da positivo conviene quedarse con ese otro.
- `CUADROS_ANTES_DE_REBUSCAR = 15` — ver `Detector.detectar`.

**`_DetectorDeUnDiccionario`** (privada)

Un detector de un solo diccionario. Todo el código de compatibilidad vive acá:
el `__init__` prueba con `hasattr` si existen `getPredefinedDictionary` y
`ArucoDetector` (API nueva) y si no cae a `Dictionary_get` / `detectMarkers(...)`
(API vieja). Su método `detectar(frame)` devuelve `([], [])` si no hay nada, y
si no normaliza la salida de OpenCV a una lista de arrays `(4, 2)` float32 y una
lista de ids enteros.

**`Detector`** (la clase pública)

- `__init__(nombre_diccionario)` — con un nombre concreto crea un único
  `_DetectorDeUnDiccionario`; con `"auto"` crea uno por cada diccionario
  disponible y deja `self.nombre = None`. Con un nombre inválido levanta
  `ValueError` listando las opciones.
- `buscando` (property) — `True` mientras todavía no sepa con qué diccionario
  trabaja. `main.py` la usa para mostrar "buscando el diccionario...".
- `detectar(frame)` — el método que usa el bucle. Devuelve `(esquinas, ids)`.
  La lógica es:
  1. Si todavía no hay diccionario fijado → barrer todos.
  2. Si hay uno fijado y detecta algo → devolverlo y resetear el contador.
  3. Si hay uno fijado y no detecta nada → sumar 1 al contador de cuadros
     vacíos; al llegar a 15, volver a barrer.

  El paso 3 es lo que evita que el sistema quede enganchado para siempre en un
  diccionario mal elegido por un falso positivo sobre un cuadro de ruido. De
  paso, permite cambiar de marcador en plena corrida.

  **El orden de las esquinas es el contrato importante de este módulo**: Aruco
  siempre las devuelve en el mismo orden respecto del marcador (0 = superior
  izquierda, 1 = superior derecha, 2 = inferior derecha, 3 = inferior
  izquierda). De ese orden sale después la orientación.

- `_buscar_diccionario(frame)` — prueba todos los diccionarios y se queda con el
  que **más marcadores** haya reconocido. Un mismo marcador suele dar positivo
  en varios diccionarios emparentados (`4x4_50` está contenido en `4x4_100`);
  cualquiera sirve, porque la geometría de las esquinas —lo único que usa la
  localización— es idéntica.

**`dibujar_detecciones(frame, esquinas, ids)`**

Devuelve una **copia** del frame con el contorno verde de cada marcador, la
etiqueta `id=N` y un círculo magenta sobre la esquina 0. El círculo no es
decorativo: es la referencia visual de dónde sale el ángulo. El texto se dibuja
dos veces (negro grueso y color fino) para que se lea sobre cualquier fondo.

---

### 4.2 `src/registro.py` — el corazón geométrico

**Para qué existe.** Fija el sistema de referencia métrico del mundo. Es
instantáneo: se dispara con `r` y produce todo lo que el resto del programa
necesita.

**`Registro`** (dataclass) — el estado que devuelve el registro:

| Campo | Qué es |
|---|---|
| `H_img_mm` | homografía imagen (px) → mundo (mm). **Es el resultado buscado** |
| `H_mm_vis` | semejanza mm → píxeles de la ventana W2D |
| `H_img_vis` | `H_mm_vis @ H_img_mm`, imagen → vista cenital. Sólo para anotar |
| `fondo` | la imagen rectificada congelada en el instante del registro |
| `id_referencia` | qué marcador definió el sistema de referencia |
| `lado_mm` | el lado real del marcador: la unidad métrica de todo |

**`esquinas_en_mm(lado_mm)`**

Devuelve las cuatro esquinas del marcador de referencia *en coordenadas del
mundo*: un cuadrado centrado en el origen, de lado `lado_mm`, con x hacia la
derecha e y hacia arriba. El orden es el mismo que devuelve Aruco, así que la
correspondencia punto a punto con las esquinas de la imagen es directa. **Esta
función es la que define la convención de todo el sistema**: por eso el marcador
que registra queda en (0, 0, 0°).

**`matriz_mm_a_vista(ancho, alto, escala)`**

```
[ escala    0     ancho/2 ]
[   0    -escala  alto/2  ]
[   0       0        1    ]
```

Apenas una semejanza: escala de mm a píxeles, traslada el origen al centro de la
ventana, y el signo negativo invierte el eje vertical (en el mundo la y crece
hacia arriba, en una imagen hacia abajo). `escala` está en píxeles por
milímetro; con el default 1.2 y una ventana de 720 px, la vista cubre ±300 mm.

**`elegir_marcador(ids)`**

Con varios marcadores en escena hay que elegir uno como referencia. Devuelve el
**índice** del id más chico. El criterio da igual —cualquiera sirve—, pero éste
es reproducible entre corridas.

**`registrar_plano(frame, esquinas, ids, lado_mm, ancho_vista, alto_vista, escala)`**

La función principal del módulo. Paso a paso:

1. Si no hay ningún marcador → devuelve `None` (el registro no se lleva a cabo).
2. Elige el marcador de referencia.
3. Calcula `H_img_mm` con `cv2.getPerspectiveTransform(esquinas_img,
   destino_mm)`. Son **cuatro puntos exactos**, así que resuelve el sistema
   directamente; no se usa `findHomography` porque no hay nada que ajustar por
   mínimos cuadrados.
4. Compone `H_img_vis = H_mm_vis @ H_img_mm`.
5. Produce el fondo con `cv2.warpPerspective(frame, H_img_vis, ...)`: la vista
   cenital del plano. **Esta rectificación se hace una sola vez, acá, y no en el
   bucle de cámara** — por eso el fondo de `W2D` es una foto congelada y no
   video en vivo.

---

### 4.3 `src/localizacion.py` — de píxeles a milímetros

**Para qué existe.** Es el módulo que da el resultado del TP. Es corto porque
todo el trabajo pesado ya lo hizo la homografía.

**`Pose`** (dataclass) — `identificador`, `x_mm`, `y_mm`, `angulo_grados`
(respecto del eje x del mundo, positivo antihorario) y `esquinas_mm`, el array
(4, 2) de las esquinas ya en milímetros, en el mismo orden que las de la imagen.
La property `centro_mm` devuelve `[x_mm, y_mm]` como array, cómodo para hacer
cuentas vectoriales al dibujar.

**`transformar_puntos(puntos, homografia)`**

Envoltorio de `cv2.perspectiveTransform`, que exige la forma `(N, 1, 2)` y
float32. Toma `(N, 2)`, devuelve `(N, 2)` en float64. Es la única función del
proyecto que toca la API de transformación, y la usan tanto la localización como
el dibujo de la vista cenital.

**`pose_en_mm(esquinas_img, identificador, H_img_mm)`**

Las dos decisiones que importan:

- **Posición** = promedio de las cuatro esquinas *ya llevadas a milímetros*. El
  promedio se hace en el mundo y no en la imagen porque **la perspectiva no
  conserva el punto medio**: promediar en píxeles y después transformar daría un
  centro corrido.
- **Orientación** = dirección del eje x propio del marcador, del punto medio del
  lado izquierdo (esquinas 0 y 3) al del lado derecho (esquinas 1 y 2), con
  `arctan2`. Se usan los puntos medios de los lados y no una sola esquina porque
  promedia el ruido de detección.

Con esta convención, el marcador que registró el plano queda exactamente en la
pose (0, 0, 0°) en el instante del registro.

**`localizar(esquinas, ids, H_img_mm)`**

Una línea: aplica `pose_en_mm` a todos los marcadores detectados y devuelve la
lista de `Pose`.

---

### 4.4 `src/vista_cenital.py` — la ventana W2D

**Para qué existe.** Todo el dibujo de la ventana de resultado. No calcula nada
de geometría del mundo: recibe el `Registro` y la lista de `Pose` y pinta.

- **Constantes** — colores de ejes, contorno, flecha, grilla y texto;
  `LARGO_EJES_PX = 70` y `PASO_GRILLA_MM = 50`.

- **`texto(imagen, cadena, posicion, ...)`** — texto con contorno negro, legible
  sobre cualquier fondo. Además **corrige la posición** con `getTextSize` para
  que la etiqueta no quede cortada por el borde, cosa que pasa apenas el
  marcador se acerca a un extremo de la ventana. La usa también `main.py` para
  el cartel de ayuda de la ventana `Cam`.

- **`_a_pixel(punto_mm, H_mm_vis)`** — un punto en mm llevado a coordenadas
  enteras de la ventana. Atajo sobre `transformar_puntos`.

- **`dibujar_grilla(lienzo, H_mm_vis, escala)`** — líneas cada 50 mm a partir
  del origen, hacia los cuatro lados. Si el paso quedara de menos de 8 px
  (escala muy chica) no dibuja nada: sería una mancha. Se pinta tenue para no
  tapar el fondo.

- **`dibujar_ejes(lienzo, H_mm_vis)`** — las dos flechas del sistema de
  referencia (x roja a la derecha, y verde hacia arriba) con sus etiquetas y un
  punto blanco en el origen.

- **`dibujar_pose(lienzo, pose, H_mm_vis, lado_mm)`** — por cada marcador: el
  contorno cuadrado (transportando `pose.esquinas_mm`), la flecha de orientación
  y la etiqueta con `x`, `y` y ángulo. Detalle: **la flecha se arma en
  milímetros** (largo = 90 % del lado del marcador) y recién después se lleva a
  píxeles, para que su longitud sea una medida del mundo y no un número
  arbitrario de pantalla.

- **`dibujar_w2d(registro, poses)`** — arma la ventana completa: copia el fondo,
  dibuja grilla, ejes y cada pose, pone el encabezado con el id de referencia y
  el lado, y si no hay marcadores agrega el aviso abajo. Nótese que recupera la
  escala leyendo `registro.H_mm_vis[0, 0]`, que por construcción es exactamente
  el valor de `--escala`.

- **`lienzo_sin_registro(ancho, alto)`** — la pantalla gris con la instrucción,
  para antes del registro. Existe porque **sin homografías no hay ni fondo ni
  ejes**: no hay nada que se pueda dibujar.

---

### 4.5 `src/main.py` — el programa

**`parsear_argumentos(argumentos=None)`** — define las seis opciones de línea de
comando (`--camara`, `--lado-mm`, `--diccionario`, `--ancho-w2d`, `--alto-w2d`,
`--escala`). Las opciones válidas de `--diccionario` salen de la tabla
`DICCIONARIOS`, así que se adaptan solas a la versión de OpenCV instalada. El
parámetro `argumentos=None` permite llamarla desde un test sin tocar `sys.argv`.

**`main(argumentos=None)`** — el bucle. Su estructura:

1. Abre la cámara; si falla, imprime y devuelve 1.
2. Crea el `Detector` y deja `registro = None`.
3. En cada vuelta:
   - lee el frame; si la cámara deja de entregar, corta;
   - detecta marcadores;
   - arma la ventana `Cam` con `dibujar_detecciones` más un **cartel de ayuda
     contextual** (buscando diccionario / sin marcadores / podés registrar / ya
     registrado). El cartel va sobre la ventana y no sólo en la consola porque
     la tecla `r` la recibe la ventana: hay que tener el foco ahí;
   - si el diccionario recién se descubrió, lo avisa por consola una sola vez;
   - arma `W2D`: `lienzo_sin_registro` si todavía no hay registro, y si no
     `localizar` + `dibujar_w2d`. Con registro pero sin marcadores igual se
     dibuja, porque el fondo y los ejes ya están determinados;
   - lee la tecla: `ESC` sale, `r` registra.
4. `r` llama a `registrar_plano` **sobre el frame sin anotar** — detalle
   importante: si se le pasara `cam`, el fondo cenital quedaría con los
   contornos verdes pintados encima. Si devuelve `None`, avisa que no había
   marcadores y deja el registro anterior intacto.
5. El `finally` libera la cámara y cierra las ventanas pase lo que pase.

---

### 4.6 `src/generar_marcador.py` — utilidad de marcadores

Genera el PNG de un marcador para imprimir o mostrar en pantalla. Lo único no
trivial es la conversión de tamaño:

```
lado_px = lado_mm / 25.4 * dpi
```

o sea, el marcador sale con el tamaño físico pedido para el dpi del dispositivo.
Eso importa porque **el lado real del marcador es la unidad métrica de todo el
sistema**: si se declara `--lado-mm 100` pero el papel mide 87 mm, todas las
coordenadas salen escaladas por 87/100.

Agrega además un margen blanco (`--borde-mm`, 10 mm por defecto) con
`copyMakeBorder`. Sin esa zona de silencio el detector no encuentra el marcador.

También tiene su rama de compatibilidad: `generateImageMarker` en OpenCV nuevo,
`drawMarker` en el viejo.

---

### 4.7 `src/diagnostico.py` — utilidad de depuración

Herramienta aparte, autónoma (no importa nada del proyecto), para cuando la
ventana `Cam` no muestra ningún contorno verde. La diferencia con el `auto` de
`deteccion.py` es que acá **no se corta en el primero que funciona**: reporta
*todos* los diccionarios que reconocen el marcador, y dibuja en rojo los
candidatos rechazados.

- `nombres_de_diccionarios()` / `crear_detectores()` — arman un detector por
  cada diccionario disponible.
- `barrer(frame, detectores)` — prueba todos y devuelve `{diccionario: [ids]}`,
  más las esquinas y los rechazados del primero que haya reconocido algo. Se
  corre cada 5 cuadros (`CADA_CUANTOS_CUADROS`) porque son más de veinte
  detecciones y sería inusable en cada frame.
- `texto` / `anotar` — pintan el resultado y, según el caso, el consejo
  correspondiente.
- `main` — bucle propio, con `g` para guardar el cuadro **sin anotar** en
  `captura_diagnostico.png` y `ESC` para salir.

La tabla de lectura: verde = se reconoce, el problema no es el marcador; rojo =
ve el cuadrado pero no lee el código (diccionario equivocado, foco, marcador
chico u oblicuo); nada = ni el cuadrado (contraste, reflejo, falta de margen
blanco).

---

### 4.8 `tests/`

**`conftest.py`** — cuatro líneas que agregan `src/` al `sys.path`, para poder
escribir `from localizacion import ...` en los tests igual que lo hace `main.py`.

**`test_localizacion.py`** — la parte conceptualmente linda del proyecto: los
tests **no necesitan webcam**. Se arma a mano una matriz `H_MM_IMG` que hace de
cámara en perspectiva (una semejanza más los términos proyectivos de la última
fila, que son los que producen el escorzo), se proyectan con ella marcadores de
pose conocida, y se verifica que el registro y la localización devuelvan esa
misma pose.

Helpers:
- `proyectar(puntos_mm)` — aplica la cámara sintética en coordenadas homogéneas.
- `esquinas_de_marcador(x, y, angulo)` — las esquinas en mm de un marcador
  puesto en una pose dada (rota `esquinas_en_mm` y traslada).
- `frame_vacio()`, `registrar(...)` — andamiaje.

Qué se verifica:

| Test | Qué garantiza |
|---|---|
| `test_sin_marcadores_no_hay_registro` | sin marcador, `registrar_plano` devuelve `None` |
| `test_el_marcador_de_referencia_queda_en_el_origen` | la convención (0, 0, 0°) |
| `test_se_recupera_la_pose_de_un_marcador_movil` | 4 poses distintas se recuperan con error < 0.05 mm / 0.05° |
| `test_el_registro_elige_el_id_mas_chico` | el criterio de `elegir_marcador` |
| `test_la_escala_metrica_la_fija_el_lado_del_marcador` | los cuatro lados miden `LADO` |
| `test_localizar_devuelve_una_pose_por_marcador` | `localizar` respeta el orden y los ids |
| `test_la_vista_cenital_...` | `matriz_mm_a_vista`: origen al centro, y hacia arriba, escala correcta |

Se corren con `python -m pytest tests`.

---

## 5. La matemática, en limpio

Un punto del plano del mundo `(X, Y)` se ve en la imagen en `(u, v)`. Como todo
ocurre sobre un plano, la relación en coordenadas homogéneas es lineal:

```
[ u' ]       [ X ]
[ v' ]  =  H [ Y ]        u = u'/w',  v = v'/w'
[ w' ]       [ 1 ]
```

`H` es 3×3 con 8 grados de libertad (la escala global no importa), y cada
correspondencia de puntos aporta 2 ecuaciones → **4 puntos alcanzan y sobran**.
Esas 4 correspondencias son las 4 esquinas del marcador de referencia: en la
imagen las da Aruco, en el mundo las da `esquinas_en_mm(lado_mm)`.

El registro calcula directamente el sentido que interesa, `H_img_mm`, y a partir
de ahí localizar es sólo una multiplicación por cuadro. El término proyectivo
(la última fila, distinta de `[0 0 1]`) es lo que hace que el punto medio **no**
se conserve, y por eso el centro del marcador se promedia en mm y no en px.

---

## 6. Dónde tocar si querés cambiar algo

| Quiero... | Archivo / función |
|---|---|
| cambiar la convención del origen o del eje x | `registro.esquinas_en_mm` |
| cambiar cómo se elige el marcador de referencia | `registro.elegir_marcador` |
| cambiar la definición del ángulo | `localizacion.pose_en_mm` |
| cambiar colores, grilla, etiquetas de W2D | constantes de `vista_cenital.py` |
| que el fondo cenital sea en vivo | mover el `warpPerspective` de `registro.registrar_plano` al bucle de `main` (cuesta caro por cuadro) |
| agregar una tecla | el bloque de `waitKey` en `main.main` |
| agregar una opción de línea de comando | `main.parsear_argumentos` |
| que el barrido de diccionarios sea más o menos tolerante | `deteccion.CUADROS_ANTES_DE_REBUSCAR` |
