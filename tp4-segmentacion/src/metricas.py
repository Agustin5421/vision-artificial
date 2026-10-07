"""
Función de pérdida y métricas de segmentación.

El tumor ocupa en promedio ~1% de los píxeles de un corte, y dos de cada tres
cortes no tienen tumor. Con sólo entropía cruzada, predecir "todo fondo" ya da
una pérdida baja. Por eso la pérdida suma un término Dice, que mide la
superposición con el tumor y no se deja engañar por el fondo.
"""

import torch
from torch.nn import functional as F

UMBRAL = 0.5


def dice_suave(logits, mascaras, eps=1.0):
    """Dice diferenciable, calculado sobre todos los píxeles del batch."""
    probabilidades = torch.sigmoid(logits)
    interseccion = (probabilidades * mascaras).sum()
    return (2 * interseccion + eps) / (probabilidades.sum() + mascaras.sum() + eps)


def perdida_bce_dice(logits, mascaras):
    return (F.binary_cross_entropy_with_logits(logits, mascaras)
            + 1 - dice_suave(logits, mascaras))


def dice(prediccion, real):
    """
    Dice de máscaras binarias (arrays o tensores booleanos).

    Si ninguna de las dos tiene tumor, el modelo acertó: Dice = 1.
    """
    suma = prediccion.sum() + real.sum()
    if suma == 0:
        return 1.0
    return float(2 * (prediccion & real).sum() / suma)


def iou(prediccion, real):
    union = (prediccion | real).sum()
    if union == 0:
        return 1.0
    return float((prediccion & real).sum() / union)
