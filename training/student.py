"""Browser-native student U-Net for nnLive distillation.

Uses ONLY ops that ORT-Web's WebGPU EP covers (verified against the teacher export):
  - Conv3d (group=1), strided Conv3d for downsampling (NO AvgPool, NO grouped conv)
  - Resize/Upsample (nearest) + Conv3d for upsampling (NO ConvTranspose3d)
  - InstanceNorm3d, LeakyReLU, Concat
So the exported ONNX runs fully on WebGPU by construction. Same 8-in / 2-out contract as the
teacher, fully convolutional (any patch size divisible by 2**stages).
"""
import torch, torch.nn as nn, torch.nn.functional as F


def cbr(ci, co, stride=1):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm3d(co, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class Up(nn.Module):
    """Resize(nearest, 2x) + Conv — the WebGPU-safe replacement for ConvTranspose3d."""
    def __init__(self, ci, co):
        super().__init__()
        self.block = nn.Sequential(cbr(ci, co), cbr(co, co))

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class Student(nn.Module):
    def __init__(self, cin=8, cout=2, base=32, stages=4):
        super().__init__()
        feats = [base * (2 ** i) for i in range(stages + 1)]  # 32,64,128,256,512
        self.stem = nn.Sequential(cbr(cin, feats[0]), cbr(feats[0], feats[0]))
        self.downs = nn.ModuleList(
            nn.Sequential(cbr(feats[i], feats[i + 1], stride=2), cbr(feats[i + 1], feats[i + 1]))
            for i in range(stages)
        )
        self.ups = nn.ModuleList(
            Up(feats[i] + feats[i - 1], feats[i - 1]) for i in range(stages, 0, -1)
        )
        self.head = nn.Conv3d(feats[0], cout, 1)

    def forward(self, x):
        skips = [self.stem(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        x = skips[-1]
        for i, up in enumerate(self.ups):
            x = up(x, skips[-2 - i])
        return self.head(x)


if __name__ == "__main__":
    m = Student()
    n = sum(p.numel() for p in m.parameters())
    print(f"student params {n/1e6:.1f} M")
    y = m(torch.zeros(1, 8, 128, 128, 128))
    print("out", list(y.shape))
