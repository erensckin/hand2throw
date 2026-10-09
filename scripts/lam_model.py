"""Small latent action model that doubles as a world model (LAPO-style).

    encoder:  (frame t, frame t+k) -> 4 tokens of 3 numbers -> FSQ, 5 levels each
    decoder:  frame t + quantised latent -> predicted frame t+k

The latent is tiny (125 codes per token, about 28 bits per frame pair), so it cannot carry
the image. The decoder gets the scene from frame t through skip connections, and the latent
only has room for what changed. Quantisation is FSQ (Mentzer et al. 2023), which has no
codebook that could collapse. An earlier version with an EMA VQ codebook and zero-initialised
FiLM layers did collapse: the decoder ignored the latent and the encoder became constant.

References: LAPO, Learning to Act without Actions (Schmidt & Jiang, ICLR 2024); LAPA, Latent
Action Pretraining from Videos (Ye et al., ICLR 2025), which scales the idea to VLA
pretraining.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class Encoder(nn.Module):
    """(B, cin, S, S) -> features at S/2, S/4, S/8, S/16."""

    def __init__(self, cin: int, widths=(32, 64, 128, 256)):
        super().__init__()
        chans = [cin, *widths]
        self.levels = nn.ModuleList(conv_block(chans[i], chans[i + 1], 2) for i in range(len(widths)))

    def forward(self, x):
        feats = []
        for level in self.levels:
            x = level(x)
            feats.append(x)
        return feats


class FiLM(nn.Module):
    """Condition a feature map on the latent action: h * (1 + gamma) + beta. Default (non-zero)
    initialisation, so the latent shapes the prediction from the first step."""

    def __init__(self, zdim: int, ch: int):
        super().__init__()
        self.lin = nn.Linear(zdim, 2 * ch)

    def forward(self, h, z):
        gamma, beta = self.lin(z).chunk(2, dim=-1)
        return h * (1 + gamma[..., None, None]) + beta[..., None, None]


class Decoder(nn.Module):
    """U-Net decoder over the context frame's features, FiLM-conditioned on the latent at
    every scale; outputs the change to add to the context frame (tanh, in [-1, 1])."""

    def __init__(self, zdim: int, widths=(32, 64, 128, 256)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.mid, self.film3 = conv_block(w3, w3, 1), FiLM(zdim, w3)
        self.up2, self.film2 = conv_block(w3 + w2, w2, 1), FiLM(zdim, w2)
        self.up1, self.film1 = conv_block(w2 + w1, w1, 1), FiLM(zdim, w1)
        self.up0, self.film0 = conv_block(w1 + w0, w0, 1), FiLM(zdim, w0)
        self.out = nn.Conv2d(w0, 3, 3, 1, 1)

    def forward(self, feats, z):
        f0, f1, f2, f3 = feats
        h = self.film3(self.mid(f3), z)
        for skip, block, film in ((f2, self.up2, self.film2), (f1, self.up1, self.film1), (f0, self.up0, self.film0)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = film(block(torch.cat([h, skip], dim=1)), z)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        return torch.tanh(self.out(h))


class FSQ(nn.Module):
    """Finite scalar quantisation with odd level counts: z -> tanh -> round to `levels` values,
    straight-through gradient. Returns values in [-1, 1] and an integer code per token."""

    def __init__(self, levels=(5, 5, 5)):
        super().__init__()
        assert all(n % 2 == 1 for n in levels), "use odd level counts"
        self.n_codes = math.prod(levels)
        # (not named "half": nn.Module.half() exists)
        self.register_buffer("half_range", (torch.tensor(levels, dtype=torch.float32) - 1) / 2)
        basis = [1]
        for n in levels[:-1]:
            basis.append(basis[-1] * n)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer("levels_t", torch.tensor(levels, dtype=torch.long))

    def forward(self, z):
        bounded = torch.tanh(z) * self.half_range
        rounded = torch.round(bounded)
        q = bounded + (rounded - bounded).detach()
        idx = ((rounded + self.half_range).long() * self.basis).sum(-1)
        return q / self.half_range, idx

    def codes_to_values(self, idx):
        digits = (idx[..., None] // self.basis) % self.levels_t
        return (digits.float() - self.half_range) / self.half_range


class LatentActionModel(nn.Module):
    def __init__(self, img: int = 96, n_tokens: int = 4, levels=(5, 5, 5)):
        super().__init__()
        assert img % 16 == 0
        levels = tuple(levels)
        self.cfg = {"img": img, "n_tokens": n_tokens, "levels": list(levels)}
        self.n_tokens, self.token_dim = n_tokens, len(levels)
        self.action_enc = Encoder(6)
        self.to_z = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * (img // 16) ** 2, 512), nn.SiLU(), nn.Linear(512, n_tokens * len(levels))
        )
        self.fsq = FSQ(levels)
        self.ctx_enc = Encoder(3)
        self.dec = Decoder(n_tokens * len(levels))

    @property
    def n_codes(self) -> int:
        return self.fsq.n_codes

    def encode(self, x0, x1):
        """Continuous latent (B, T, d), quantised latent (B, T, d), code per token (B, T)."""
        z = self.to_z(self.action_enc(torch.cat([x0, x1], dim=1))[-1]).view(-1, self.n_tokens, self.token_dim)
        q, idx = self.fsq(z)
        return z, q, idx

    def decode(self, x0, q):
        """World-model step: predicted next frame from the current frame and a latent action."""
        return x0 + self.dec(self.ctx_enc(x0), q.flatten(1))

    def codes_to_latent(self, idx):
        """Codes (B, T) -> quantised latent (B, T, d), e.g. to roll out chosen latent actions."""
        return self.fsq.codes_to_values(idx)

    def forward(self, x0, x1):
        z, q, idx = self.encode(x0, x1)
        return self.decode(x0, q), idx
