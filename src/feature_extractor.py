"""Configurable CNN feature extractor for stacked image observations."""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """Nature-DQN style CNN: conv stack -> flatten -> linear -> ReLU.

    SB3 transposes channels-last uint8 images to channels-first and (because
    ``normalize_images=True``) rescales them to [0, 1] *before* they reach this
    module, so the extractor receives float tensors of shape (B, C, H, W).

    Args:
        observation_space: image Box space, channels-first after SB3 preprocessing.
        features_dim: size of the output feature vector.
        channels / kernels / strides: per-layer conv configuration.
        augment_pad: if > 0, random-shift augmentation (pixels of padding) applied only in training mode.
        norm: ``"none"`` (Linear->ReLU head) or ``"layernorm"`` (Linear->LayerNorm->Tanh head).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        channels: Sequence[int] = (32, 64, 64),
        kernels: Sequence[int] = (8, 4, 3),
        strides: Sequence[int] = (4, 2, 1),
        norm: str = "none",
        augment_pad: int = 0,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 3:
            raise ValueError(f"CustomCNN expects an image space, got shape {observation_space.shape}.")
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("channels, kernels and strides must have the same length.")
        in_ch = observation_space.shape[0]
        layers: list[nn.Module] = []
        for out_ch, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s), nn.ReLU()]
            in_ch = out_ch
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        self.augment_pad = int(augment_pad)

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            try:
                n_flatten = self.cnn(sample).shape[1]
            except RuntimeError as exc:
                raise ValueError(
                    f"CNN kernels/strides {list(kernels)}/{list(strides)} are too large for the "
                    f"observation shape {observation_space.shape}. Use a larger env.resize or "
                    "smaller kernels/strides."
                ) from exc
        if norm not in ("none", "layernorm"):
            raise ValueError(f"norm must be 'none' or 'layernorm', got '{norm}'.")
        # 'layernorm': Linear -> LayerNorm -> Tanh (SAC-AE / DrQ style). Keeps features bounded and
        # input-dependent; the plain Linear -> ReLU head was observed to collapse (see README).
        head: list[nn.Module] = [nn.Linear(n_flatten, features_dim)]
        head += [nn.LayerNorm(features_dim), nn.Tanh()] if norm == "layernorm" else [nn.ReLU()]
        self.linear = nn.Sequential(*head)

    def _random_shift(self, x: torch.Tensor) -> torch.Tensor:
        """DrQ random shift: replicate-pad by ``augment_pad`` px, then crop a random HxW window per sample."""
        n, _, h, w = x.shape
        p = self.augment_pad
        x = torch.nn.functional.pad(x, (p, p, p, p), mode="replicate")
        ox = torch.randint(0, 2 * p + 1, (n,), device=x.device)
        oy = torch.randint(0, 2 * p + 1, (n,), device=x.device)
        ar = torch.arange(h, device=x.device)
        rows = (oy[:, None] + ar[None, :])[:, :, None].expand(n, h, w)
        cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])[:, None, :].expand(n, h, w)
        idx_n = torch.arange(n, device=x.device)[:, None, None].expand(n, h, w)
        return x.permute(0, 2, 3, 1)[idx_n, rows, cols].permute(0, 3, 1, 2).contiguous()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.augment_pad > 0 and self.training:  # SB3 sets training mode only inside gradient updates
            observations = self._random_shift(observations)
        return self.linear(self.cnn(observations))
