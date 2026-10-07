import torch
import torch.nn.functional as F

UMBRAL = 0.5


def perdida_bce_dice(logits, mascaras):
    # BCE sola puede dar baja prediciendo todo fondo (el tumor es muy chico),
    # por eso se le suma 1 - Dice
    bce = F.binary_cross_entropy_with_logits(logits, mascaras)
    p = torch.sigmoid(logits)
    dice = (2 * (p * mascaras).sum() + 1) / (p.sum() + mascaras.sum() + 1)
    return bce + 1 - dice


def dice(pred, real):
    # si no hay tumor en ninguna de las dos, cuenta como acierto
    suma = pred.sum() + real.sum()
    if suma == 0:
        return 1.0
    return float(2 * (pred & real).sum() / suma)


def iou(pred, real):
    union = (pred | real).sum()
    if union == 0:
        return 1.0
    return float((pred & real).sum() / union)
