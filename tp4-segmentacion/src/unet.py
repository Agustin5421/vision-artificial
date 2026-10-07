"""
U-Net (Ronneberger et al., 2015), implementada desde cero.

    entrada 3x256x256
      │
      ├─ bloque 32  ───────────────────────────────┐  (skip)
      │  maxpool                                   │
      ├─ bloque 64  ─────────────────────────┐     │
      │  maxpool                             │     │
      ├─ bloque 128 ───────────────────┐     │     │
      │  maxpool                       │     │     │
      ├─ bloque 256 ─────────────┐     │     │     │
      │  maxpool                 │     │     │     │
      └─ bloque 512 (cuello)     │     │     │     │
         subir + concat ─────────┘     │     │     │
         bloque 256                    │     │     │
         subir + concat ───────────────┘     │     │
         bloque 128                          │     │
         subir + concat ─────────────────────┘     │
         bloque 64                                 │
         subir + concat ───────────────────────────┘
         bloque 32
         conv 1x1 -> 1 canal (logit de "tumor")

Diferencias con el paper original: convoluciones con padding (la salida tiene
el mismo tamaño que la entrada, no hace falta recortar) y BatchNorm después de
cada convolución, que estabiliza el entrenamiento.
"""

import torch
from torch import nn


class BloqueDoble(nn.Module):
    """(conv 3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, entrada, salida):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(salida, salida, 3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.capas(x)


class UNet(nn.Module):
    def __init__(self, canales_entrada=3, canales_salida=1, base=32, niveles=4):
        super().__init__()
        filtros = [base * 2 ** i for i in range(niveles + 1)]  # 32 64 128 256 512

        self.bajada = nn.ModuleList()
        entrada = canales_entrada
        for f in filtros[:-1]:
            self.bajada.append(BloqueDoble(entrada, f))
            entrada = f
        self.pool = nn.MaxPool2d(2)

        self.cuello = BloqueDoble(filtros[-2], filtros[-1])

        self.subida = nn.ModuleList()
        self.decodificadores = nn.ModuleList()
        for f in reversed(filtros[:-1]):
            self.subida.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decodificadores.append(BloqueDoble(f * 2, f))

        self.salida = nn.Conv2d(filtros[0], canales_salida, 1)

    def forward(self, x):
        saltos = []
        for bloque in self.bajada:
            x = bloque(x)
            saltos.append(x)
            x = self.pool(x)

        x = self.cuello(x)

        for subir, decodificar, salto in zip(self.subida, self.decodificadores,
                                             reversed(saltos)):
            x = subir(x)
            x = decodificar(torch.cat([salto, x], dim=1))

        return self.salida(x)  # logits: aplicar sigmoid para obtener probabilidad


if __name__ == "__main__":
    modelo = UNet()
    parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Parámetros: {parametros / 1e6:.1f} M")
    print(modelo(torch.zeros(1, 3, 256, 256)).shape)
