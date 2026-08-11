from __future__ import annotations

import math

import torch

from .config import (
    ParameterDomain,
    PhysicalConstants,
)


def lognormal_permeability_ratio(
    xy: torch.Tensor,
    parameters: torch.Tensor,
    domain: ParameterDomain,
    constants: PhysicalConstants,
) -> torch.Tensor:
    """Four-mode differentiable lognormal permeability field.

    The xi columns define a realization.  The local variance correction
    gives unit ensemble mean for untruncated standard-normal xi; the
    experiment uses the declared bounded truncation.
    """

    x = xy[:, 0] / constants.half_width
    y = xy[:, 1]
    modes = torch.stack(
        (
            torch.cos(0.5 * math.pi * x) * torch.cos(math.pi * y),
            torch.sin(math.pi * x) * torch.cos(0.5 * math.pi * y),
            torch.cos(1.5 * math.pi * x) * torch.sin(math.pi * y),
            torch.sin(2.0 * math.pi * x) * torch.sin(1.5 * math.pi * y),
        ),
        dim=1,
    )
    xi = parameters[:, 4 : 4 + domain.random_modes]
    scale = math.sqrt(float(domain.random_modes))
    gaussian = (xi * modes).sum(dim=1) / scale
    local_variance = modes.square().sum(dim=1) / float(
        domain.random_modes
    )
    sigma = math.sqrt(
        math.log(1.0 + domain.permeability_cov**2)
    )
    log_ratio = (
        sigma * gaussian
        - 0.5 * sigma**2 * local_variance
    )
    return torch.exp(log_ratio)
