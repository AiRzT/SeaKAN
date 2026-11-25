from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Geometry


@dataclass
class MaterialPointGrid2D:
    """Bilinear background grid and midpoint material quadrature."""

    nx: int
    ny: int
    xy: torch.Tensor
    volume: torch.Tensor
    node_ids: torch.Tensor
    shape: torch.Tensor
    grad_x: torch.Tensor
    grad_y: torch.Tensor
    nodal_volume: torch.Tensor
    node_xy: torch.Tensor
    momentum_nodes: torch.Tensor
    fluid_nodes: torch.Tensor
    top_node_ids: torch.Tensor
    top_shape: torch.Tensor
    top_weight: torch.Tensor
    top_xy: torch.Tensor

    @classmethod
    def create(
        cls,
        nx: int,
        ny: int,
        geometry: Geometry,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "MaterialPointGrid2D":
        if nx < 16 or ny < 12:
            raise ValueError("nx >= 16 and ny >= 12 are required")
        half = geometry.half_width_over_height
        dx = 2.0 * half / nx
        dy = 1.0 / ny

        ix = torch.arange(nx, device=device)
        iy = torch.arange(ny, device=device)
        cell_x, cell_y = torch.meshgrid(ix, iy, indexing="xy")
        x = -half + (cell_x.reshape(-1).to(dtype) + 0.5) * dx
        y = (cell_y.reshape(-1).to(dtype) + 0.5) * dy
        outside = (
            x.square() + (y - geometry.center_y).square()
            > geometry.radius_over_height**2
        )
        x = x[outside]
        y = y[outside]
        cell_x = cell_x.reshape(-1)[outside]
        cell_y = cell_y.reshape(-1)[outside]
        xy = torch.stack((x, y), dim=1)
        volume = torch.full_like(x, dx * dy)

        npx = nx + 1
        n00 = cell_y * npx + cell_x
        node_ids = torch.stack((n00, n00 + 1, n00 + npx, n00 + npx + 1), dim=1)
        shape = torch.full((x.numel(), 4), 0.25, dtype=dtype, device=device)
        grad_x = torch.tensor(
            [-0.5 / dx, 0.5 / dx, -0.5 / dx, 0.5 / dx],
            dtype=dtype,
            device=device,
        ).repeat(x.numel(), 1)
        grad_y = torch.tensor(
            [-0.5 / dy, -0.5 / dy, 0.5 / dy, 0.5 / dy],
            dtype=dtype,
            device=device,
        ).repeat(x.numel(), 1)

        node_x = torch.linspace(-half, half, nx + 1, dtype=dtype, device=device)
        node_y = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype, device=device)
        xx, yy = torch.meshgrid(node_x, node_y, indexing="xy")
        node_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        nodal_volume = torch.zeros(
            node_xy.shape[0], dtype=dtype, device=device
        ).index_add(
            0,
            node_ids.reshape(-1),
            (volume[:, None] * shape).reshape(-1),
        )

        tol = 0.51 * max(dx, dy)
        is_outer_essential = (
            (node_xy[:, 1] <= 1.0e-8)
            | (node_xy[:, 0].abs() >= half - 1.0e-8)
        )
        distance = torch.sqrt(
            node_xy[:, 0].square()
            + (node_xy[:, 1] - geometry.center_y).square()
        )
        near_pipe = distance <= geometry.radius_over_height + tol
        active = nodal_volume > 1.0e-10
        momentum_nodes = active & ~is_outer_essential & ~near_pipe
        fluid_nodes = active & (node_xy[:, 1] < 1.0 - 1.0e-8)

        top_ix = torch.arange(nx, device=device)
        top_left = ny * npx + top_ix
        top_node_ids = torch.stack((top_left, top_left + 1), dim=1)
        top_shape = torch.full((nx, 2), 0.5, dtype=dtype, device=device)
        top_weight = torch.full((nx,), dx, dtype=dtype, device=device)
        top_xy = torch.stack(
            (
                -half + (top_ix.to(dtype) + 0.5) * dx,
                torch.ones(nx, dtype=dtype, device=device),
            ),
            dim=1,
        )
        return cls(
            nx=nx,
            ny=ny,
            xy=xy,
            volume=volume,
            node_ids=node_ids,
            shape=shape,
            grad_x=grad_x,
            grad_y=grad_y,
            nodal_volume=nodal_volume,
            node_xy=node_xy,
            momentum_nodes=momentum_nodes,
            fluid_nodes=fluid_nodes,
            top_node_ids=top_node_ids,
            top_shape=top_shape,
            top_weight=top_weight,
            top_xy=top_xy,
        )

    @property
    def n_particles(self) -> int:
        return self.xy.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.node_xy.shape[0]

    def p2g(self, value: torch.Tensor, derivative: str | None = None) -> torch.Tensor:
        basis = (
            self.shape
            if derivative is None
            else self.grad_x
            if derivative == "x"
            else self.grad_y
            if derivative == "y"
            else None
        )
        if basis is None:
            raise ValueError("derivative must be None, 'x', or 'y'")
        contribution = self.volume[:, None] * basis * value[:, None]
        out = torch.zeros(
            self.n_nodes, dtype=value.dtype, device=value.device
        )
        return out.index_add(
            0, self.node_ids.reshape(-1), contribution.reshape(-1)
        )

    def top_p2g(self, value: torch.Tensor) -> torch.Tensor:
        contribution = self.top_weight[:, None] * self.top_shape * value[:, None]
        out = torch.zeros(
            self.n_nodes, dtype=value.dtype, device=value.device
        )
        return out.index_add(
            0, self.top_node_ids.reshape(-1), contribution.reshape(-1)
        )
