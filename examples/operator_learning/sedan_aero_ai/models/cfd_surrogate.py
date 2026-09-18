"""
AI surrogate model definitions for sedan external aerodynamics.

Uses PaddleScience's TFNO3dNet (3D Fourier Neural Operator) as the primary
architecture, which provides resolution-invariant operator learning with
3+ orders of magnitude speedup over traditional CFD.

Architecture choices:
  1. TFNO3dNet — Primary: FNO for 3D flow fields, 100-1000x speedup
  2. Transolver — Optional: Transformer-based PDE solver
  3. DeepONet — Optional: For parameter-varying boundary conditions
"""

import paddle
import paddle.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


class SedanAeroFNO(nn.Layer):
    """
    3D Fourier Neural Operator for sedan external aerodynamics.

    Maps: (3D domain + SDF of car geometry) -> (u, v, w, p) flow fields

    This is a standalone implementation that mirrors PaddleScience's TFNO3dNet
    but is self-contained for portability. When PaddleScience is installed,
    use ppsci.arch.TFNO3dNet directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        n_modes: Tuple[int, int, int] = (12, 12, 12),
        hidden_channels: int = 64,
        n_layers: int = 4,
        domain_padding: float = 0.1,
        fno_skip: str = "linear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.domain_padding = domain_padding
        self.fno_skip = fno_skip

        # Input projection
        self.fc0 = nn.Linear(in_channels, hidden_channels)

        # Fourier layers
        self.fourier_layers = nn.LayerList()
        self.conv_layers = nn.LayerList()
        self.w_layers = nn.LayerList()

        for _ in range(n_layers):
            self.fourier_layers.append(
                SpectralConv3d(
                    hidden_channels,
                    hidden_channels,
                    n_modes[0],
                    n_modes[1],
                    n_modes[2],
                )
            )
            self.conv_layers.append(
                nn.Conv3D(hidden_channels, hidden_channels, kernel_size=1)
            )
            self.w_layers.append(
                nn.Conv3D(hidden_channels, hidden_channels, kernel_size=1)
            )

        # Output projection
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.act = nn.GELU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, D, H, W)

        Returns:
            Output tensor of shape (N, C_out, D, H, W)
        """
        # Domain padding
        if self.domain_padding > 0:
            x = self._pad_domain(x)

        # Input projection
        x = x.transpose([0, 2, 3, 4, 1])  # (N, D, H, W, C)
        x = self.fc0(x)
        x = x.transpose([0, 4, 1, 2, 3])  # (N, C, D, H, W)

        # Fourier layers with skip connections
        x_skip = self.w_layers[0](x)
        for i, (fourier, conv, w) in enumerate(
            zip(self.fourier_layers, self.conv_layers, self.w_layers)
        ):
            x_fourier = fourier(x)
            x_conv = conv(x)
            x = x_fourier + x_conv
            if i > 0:
                x = x + w(x_skip)
            x = self.act(x)

        # Remove domain padding
        if self.domain_padding > 0:
            x = self._unpad_domain(x)

        # Output projection
        x = x.transpose([0, 2, 3, 4, 1])  # (N, D, H, W, C)
        x = self.fc2(self.act(self.fc1(x)))
        x = x.transpose([0, 4, 1, 2, 3])  # (N, C_out, D, H, W)

        return x

    def _pad_domain(self, x):
        pad = int(self.domain_padding * x.shape[-1])
        pad_d = int(self.domain_padding * x.shape[-3])
        # Remember pad amounts so _unpad_domain crops symmetrically
        # (recomputing from the padded shape would over-crop).
        self._pad_sizes = (pad_d, pad)
        return nn.functional.pad(x, [pad, pad, pad, pad, pad_d, pad_d], mode="constant")

    def _unpad_domain(self, x):
        pad_d, pad = self._pad_sizes
        # Explicit end indices: `pad:-pad` would yield an empty slice when pad == 0
        return x[
            ...,
            pad_d : x.shape[-3] - pad_d,
            pad : x.shape[-2] - pad,
            pad : x.shape[-1] - pad,
        ]


class SpectralConv3d(nn.Layer):
    """3D Spectral Convolution layer for FNO using matmul for complex mult."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_channels * out_channels)
        init = nn.initializer.Normal(std=scale)
        w_shape = [in_channels, out_channels, modes_d, modes_h, modes_w]
        self.weights_real = paddle.create_parameter(
            shape=w_shape, dtype="float32", default_initializer=init,
        )
        self.weights_imag = paddle.create_parameter(
            shape=w_shape, dtype="float32", default_initializer=init,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """x: (N, C_in, D, H, W) -> (N, C_out, D, H, W)"""
        N, C_in, D, H, W = tuple(x.shape)

        # rFFT: (N, C_in, D, H, W//2+1) as complex
        x_ft = paddle.fft.rfftn(x, axes=[-3, -2, -1])

        md = min(self.modes_d, D)
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, x_ft.shape[-1])

        # Work with real/imag parts separately
        out_real = paddle.zeros_like(x_ft.real())
        out_imag = paddle.zeros_like(x_ft.imag())

        # For the low-frequency modes, apply complex linear transform
        # (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
        # x_ft: (N, C_in, md, mh, mw) complex
        # weights: (C_in, C_out, md, mh, mw) complex
        # out_ft[c_out] = sum_{c_in} x_ft[c_in] * w[c_in, c_out]
        slice_ft = x_ft[:, :, :md, :mh, :mw]  # (N, C_in, md, mh, mw)
        w_r = self.weights_real[:, :, :md, :mh, :mw]  # (C_in, C_out, md, mh, mw)
        w_i = self.weights_imag[:, :, :md, :mh, :mw]  # (C_in, C_out, md, mh, mw)

        # einsum: N,Cin,D,H,W * Cin,Cout,D,H,W -> N,Cout,D,H,W
        out_real[:, :, :md, :mh, :mw] = (
            paddle.einsum("ncdhw,codhw->nodhw", slice_ft.real(), w_r) -
            paddle.einsum("ncdhw,codhw->nodhw", slice_ft.imag(), w_i)
        )
        out_imag[:, :, :md, :mh, :mw] = (
            paddle.einsum("ncdhw,codhw->nodhw", slice_ft.real(), w_i) +
            paddle.einsum("ncdhw,codhw->nodhw", slice_ft.imag(), w_r)
        )

        out_ft = paddle.complex(out_real, out_imag)
        x_out = paddle.fft.irfftn(out_ft, s=(D, H, W), axes=[-3, -2, -1])
        return x_out


class SedanAeroCNN(nn.Layer):
    """
    Alternative 3D CNN-based surrogate model.

    Simpler architecture using 3D convolutions with residual connections.
    Good for quick prototyping when FNO is overkill.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        hidden_channels: int = 64,
        n_layers: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.enc_conv1 = nn.Conv3D(in_channels, hidden_channels, 3, padding=1)
        self.enc_conv2 = nn.Conv3D(hidden_channels, hidden_channels * 2, 3, padding=1, stride=2)
        self.enc_conv3 = nn.Conv3D(hidden_channels * 2, hidden_channels * 4, 3, padding=1, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3D(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
            nn.GELU(),
            nn.Conv3D(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
        )

        # Decoder
        self.dec_conv1 = nn.Conv3DTranspose(hidden_channels * 4, hidden_channels * 2, 3, padding=1, stride=2)
        self.dec_conv2 = nn.Conv3DTranspose(hidden_channels * 2, hidden_channels, 3, padding=1, stride=2)
        self.dec_conv3 = nn.Conv3D(hidden_channels, out_channels, 3, padding=1)

        self.act = nn.GELU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # Encoder
        e1 = self.act(self.enc_conv1(x))
        e2 = self.act(self.enc_conv2(e1))
        e3 = self.act(self.enc_conv3(e2))

        # Bottleneck
        b = self.bottleneck(e3) + e3

        # Decoder with skip connections
        d1 = self.act(self.dec_conv1(b))
        # Strided convs can make spatial dims non-divisible by 2; align the
        # transposed-conv output to the encoder feature map before adding.
        d1 = nn.functional.interpolate(d1, size=e2.shape[-3:], mode="trilinear")
        d1 = d1 + e2
        d2 = self.act(self.dec_conv2(d1))
        d2 = nn.functional.interpolate(d2, size=e1.shape[-3:], mode="trilinear")
        d2 = d2 + e1
        out = self.dec_conv3(d2)

        return out


def build_model(config: dict) -> nn.Layer:
    """
    Build the AI surrogate model based on configuration.

    Args:
        config: Model configuration dictionary.

    Returns:
        PaddlePaddle model instance.
    """
    model_cfg = config["MODEL"]
    arch = model_cfg.get("arch", "fno")

    if arch == "fno":
        return SedanAeroFNO(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            n_modes=(
                model_cfg["n_modes_depth"],
                model_cfg["n_modes_height"],
                model_cfg["n_modes_width"],
            ),
            hidden_channels=model_cfg["hidden_channels"],
            n_layers=model_cfg["n_layers"],
            domain_padding=model_cfg.get("domain_padding", 0.0),
        )
    elif arch == "cnn":
        return SedanAeroCNN(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            hidden_channels=model_cfg["hidden_channels"],
            n_layers=model_cfg.get("n_layers", 5),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Layer) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)


def load_pretrained(model: nn.Layer, path: str):
    """Load pretrained weights."""
    state_dict = paddle.load(path)
    model.set_state_dict(state_dict)
    print(f"Loaded pretrained model from {path}")
