# U-Net (Ronneberger et al. 2015)
# Diferencias con el paper: padding en las convoluciones y BatchNorm

import torch
from torch import nn


class BloqueDoble(nn.Module):
    # conv 3x3 -> BN -> ReLU, dos veces
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
    def __init__(self, canales_entrada=3, canales_salida=1, base=32):
        super().__init__()
        filtros = [base, base * 2, base * 4, base * 8]  # 32, 64, 128, 256

        self.bajada = nn.ModuleList()
        anterior = canales_entrada
        for f in filtros:
            self.bajada.append(BloqueDoble(anterior, f))
            anterior = f
        self.pool = nn.MaxPool2d(2)

        self.cuello = BloqueDoble(filtros[-1], filtros[-1] * 2)

        self.subida = nn.ModuleList()
        self.decodificadores = nn.ModuleList()
        for f in reversed(filtros):
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

        for i in range(len(self.subida)):
            x = self.subida[i](x)
            x = torch.cat([saltos[-1 - i], x], dim=1)
            x = self.decodificadores[i](x)

        return self.salida(x)  # logits, sin sigmoid


if __name__ == "__main__":
    modelo = UNet()
    print("parámetros:", sum(p.numel() for p in modelo.parameters()))
    print(modelo(torch.zeros(1, 3, 256, 256)).shape)
