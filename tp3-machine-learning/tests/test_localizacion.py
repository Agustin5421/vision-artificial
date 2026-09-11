"""
Tests de la localización homográfica, con una cámara sintética.

La idea es no depender de una webcam: se arma a mano una homografía
plano -> imagen que hace las veces de cámara en perspectiva, se
proyectan con ella marcadores de pose conocida, y se verifica que el
registro y la localización devuelvan esa pose.
"""

import numpy as np
import pytest

from localizacion import localizar, pose_en_mm
from registro import esquinas_en_mm, matriz_mm_a_vista, registrar_plano

LADO = 100.0
ANCHO_CAM, ALTO_CAM = 640, 480

# Cámara sintética: semejanza (escala + traslación) más los términos
# proyectivos de la última fila, que son los que producen la perspectiva.
H_MM_IMG = np.array([
    [1.8, 0.10, 320.0],
    [0.05, -1.7, 240.0],
    [0.0008, 0.0012, 1.0],
], dtype=np.float64)


def proyectar(puntos_mm, homografia=H_MM_IMG):
    """Proyecta puntos del plano métrico a la imagen de la cámara."""
    puntos = np.asarray(puntos_mm, dtype=np.float64)
    homogeneos = np.hstack([puntos, np.ones((len(puntos), 1))])
    proyectados = homogeneos @ homografia.T
    return (proyectados[:, :2] / proyectados[:, 2:3]).astype(np.float32)


def esquinas_de_marcador(x_mm, y_mm, angulo_grados, lado=LADO):
    """Esquinas en mm de un marcador ubicado en una pose dada."""
    angulo = np.radians(angulo_grados)
    rotacion = np.array([[np.cos(angulo), -np.sin(angulo)],
                         [np.sin(angulo), np.cos(angulo)]])
    return esquinas_en_mm(lado) @ rotacion.T + np.array([x_mm, y_mm])


def frame_vacio():
    return np.zeros((ALTO_CAM, ANCHO_CAM, 3), dtype=np.uint8)


def registrar(ids=(7,), poses=((0.0, 0.0, 0.0),)):
    esquinas = [proyectar(esquinas_de_marcador(*pose)) for pose in poses]
    return registrar_plano(frame_vacio(), esquinas, list(ids), LADO,
                           400, 400, 1.0)


def test_sin_marcadores_no_hay_registro():
    assert registrar_plano(frame_vacio(), [], [], LADO, 400, 400, 1.0) is None


def test_el_marcador_de_referencia_queda_en_el_origen():
    registro = registrar()
    esquinas = proyectar(esquinas_de_marcador(0.0, 0.0, 0.0))

    pose = pose_en_mm(esquinas, 7, registro.H_img_mm)

    assert pose.x_mm == pytest.approx(0.0, abs=1e-6)
    assert pose.y_mm == pytest.approx(0.0, abs=1e-6)
    assert pose.angulo_grados == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("x, y, angulo", [
    (120.0, -80.0, 30.0),
    (-45.5, 210.0, -115.0),
    (0.0, 150.0, 90.0),
    (300.0, 300.0, 179.0),
])
def test_se_recupera_la_pose_de_un_marcador_movil(x, y, angulo):
    registro = registrar()
    esquinas = proyectar(esquinas_de_marcador(x, y, angulo))

    pose = pose_en_mm(esquinas, 3, registro.H_img_mm)

    assert pose.x_mm == pytest.approx(x, abs=0.05)
    assert pose.y_mm == pytest.approx(y, abs=0.05)
    assert pose.angulo_grados == pytest.approx(angulo, abs=0.05)


def test_el_registro_elige_el_id_mas_chico():
    registro = registrar(ids=(9, 4), poses=((50.0, 20.0, 10.0),
                                            (0.0, 0.0, 0.0)))
    assert registro.id_referencia == 4

    # El marcador elegido es el que queda en el origen, no el otro.
    esquinas = proyectar(esquinas_de_marcador(0.0, 0.0, 0.0))
    pose = pose_en_mm(esquinas, 4, registro.H_img_mm)
    assert pose.centro_mm == pytest.approx([0.0, 0.0], abs=1e-6)


def test_la_escala_metrica_la_fija_el_lado_del_marcador():
    registro = registrar()
    esquinas = proyectar(esquinas_de_marcador(0.0, 0.0, 0.0))

    pose = pose_en_mm(esquinas, 7, registro.H_img_mm)
    lados = [np.linalg.norm(pose.esquinas_mm[i] - pose.esquinas_mm[(i + 1) % 4])
             for i in range(4)]

    assert lados == pytest.approx([LADO] * 4, abs=0.01)


def test_localizar_devuelve_una_pose_por_marcador():
    registro = registrar()
    esquinas = [proyectar(esquinas_de_marcador(0.0, 0.0, 0.0)),
                proyectar(esquinas_de_marcador(-120.0, 60.0, 45.0))]

    poses = localizar(esquinas, [7, 2], registro.H_img_mm)

    assert [p.identificador for p in poses] == [7, 2]
    assert poses[1].x_mm == pytest.approx(-120.0, abs=0.05)
    assert poses[1].angulo_grados == pytest.approx(45.0, abs=0.05)


def test_la_vista_cenital_tiene_el_origen_al_centro_y_la_y_hacia_arriba():
    H = matriz_mm_a_vista(400, 300, 2.0)

    origen = H @ np.array([0.0, 0.0, 1.0])
    arriba = H @ np.array([0.0, 50.0, 1.0])
    derecha = H @ np.array([50.0, 0.0, 1.0])

    assert origen[:2] == pytest.approx([200.0, 150.0])
    assert arriba[1] < origen[1]       # +y del mundo sube en la imagen
    assert derecha[0] > origen[0]      # +x del mundo va a la derecha
    assert derecha[0] - origen[0] == pytest.approx(100.0)  # 2 px por mm
