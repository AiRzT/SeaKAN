from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CubicBSplineBasis(nn.Module):
    """Open-uniform cubic B-spline basis with autograd-compatible evaluation."""

    def __init__(self, n_basis: int = 10, degree: int = 3):
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be positive")
        if n_basis <= degree:
            raise ValueError("n_basis must be greater than degree")
        n_internal = n_basis - degree - 1
        if n_internal > 0:
            internal = torch.linspace(0.0, 1.0, n_internal + 2, dtype=torch.float64)[1:-1]
        else:
            internal = torch.empty(0, dtype=torch.float64)
        knots = torch.cat(
            (
                torch.zeros(degree + 1, dtype=torch.float64),
                internal,
                torch.ones(degree + 1, dtype=torch.float64),
            )
        )
        self.n_basis = n_basis
        self.degree = degree
        self.register_buffer("knots", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # KAN layer inputs are bounded to [-1, 1]. Mapping to the open interval
        # avoids the ambiguous right endpoint in the zeroth-order basis.
        u = ((x + 1.0) * 0.5).clamp(0.0, 1.0 - 1.0e-12)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        n_zero = knots.numel() - 1
        basis = (
            (u.unsqueeze(-1) >= knots[:-1])
            & (u.unsqueeze(-1) < knots[1:])
        ).to(x.dtype)

        for order in range(1, self.degree + 1):
            n_now = n_zero - order
            left_den = knots[order : order + n_now] - knots[:n_now]
            right_den = knots[order + 1 : order + n_now + 1] - knots[1 : n_now + 1]

            left_num = u.unsqueeze(-1) - knots[:n_now]
            right_num = knots[order + 1 : order + n_now + 1] - u.unsqueeze(-1)

            left = torch.where(
                left_den.abs() > 0,
                left_num / torch.where(left_den.abs() > 0, left_den, torch.ones_like(left_den)),
                torch.zeros_like(left_num),
            )
            right = torch.where(
                right_den.abs() > 0,
                right_num
                / torch.where(right_den.abs() > 0, right_den, torch.ones_like(right_den)),
                torch.zeros_like(right_num),
            )
            basis = left * basis[..., :n_now] + right * basis[..., 1 : n_now + 1]

        return basis


class KANLayer(nn.Module):
    """
    Smooth KAN layer.

    Each edge contains a learnable base branch and a cubic B-spline branch:
        phi_ji(x_i) = w_ji * silu(x_i) + sum_g c_jig B_g(x_i)
    and each output node sums its incoming edge functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_basis: int = 10,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis = CubicBSplineBasis(n_basis=n_basis, degree=3)
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        self.spline_weight = nn.Parameter(
            torch.zeros(out_features, in_features, n_basis, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        nn.init.xavier_uniform_(self.base_weight, gain=0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        base = F.silu(bounded) @ self.base_weight.t()
        basis = self.basis(bounded)
        spline = torch.einsum("nig,oig->no", basis, self.spline_weight)
        return base + spline + self.bias

    def spline_roughness(self) -> torch.Tensor:
        second_difference = (
            self.spline_weight[..., 2:]
            - 2.0 * self.spline_weight[..., 1:-1]
            + self.spline_weight[..., :-2]
        )
        return torch.mean(second_difference.square())


class KAN(nn.Module):
    def __init__(
        self,
        widths: list[int],
        n_basis: int = 10,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain an input and an output width")
        self.layers = nn.ModuleList(
            [
                KANLayer(widths[i], widths[i + 1], n_basis=n_basis, dtype=dtype)
                for i in range(len(widths) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def spline_roughness(self) -> torch.Tensor:
        return torch.stack([layer.spline_roughness() for layer in self.layers]).mean()

