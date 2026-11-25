from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Geometry, WaveParameters


@dataclass
class MaterialPointCloud2D:
    """Uniform material points with bilinear particle-to-grid connectivity."""

    xy: torch.Tensor
    volume: torch.Tensor
    node_ids: torch.Tensor
    shape: torch.Tensor
    n_nodes: int
    node_xy: torch.Tensor
    cells_x: int
    cells_y: int
    length_x: float

    @classmethod
    def create(
        cls,
        wave: WaveParameters,
        geometry: Geometry,
        device: torch.device,
        dtype: torch.dtype,
        *,
        cells_x: int = 32,
        cells_y: int = 64,
    ) -> "MaterialPointCloud2D":
        if cells_x < 8 or cells_y < 16:
            raise ValueError("MPM grid must be at least 8 by 16")
        length_x = wave.wavelength / geometry.seabed_thickness
        dx = length_x / cells_x
        dy = 1.0 / cells_y
        ix = torch.arange(cells_x, device=device, dtype=dtype)
        iy = torch.arange(cells_y, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(
            (ix + 0.5) * dx,
            (iy + 0.5) * dy,
            indexing="xy",
        )
        xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        volume = torch.full(
            (xy.shape[0],),
            dx * dy,
            dtype=dtype,
            device=device,
        )
        node_ids, shape, node_xy = _p2g_connectivity(
            xy, length_x, cells_x, cells_y
        )
        return cls(
            xy=xy,
            volume=volume,
            node_ids=node_ids,
            shape=shape,
            n_nodes=(cells_x + 1) * (cells_y + 1),
            node_xy=node_xy,
            cells_x=cells_x,
            cells_y=cells_y,
            length_x=length_x,
        )

    @property
    def n_particles(self) -> int:
        return int(self.xy.shape[0])

    def sample_indices(
        self,
        requested: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        requested = min(requested, self.n_particles)
        if requested == self.n_particles:
            return torch.arange(self.n_particles, device=self.xy.device)
        return torch.randperm(
            self.n_particles,
            device=self.xy.device,
            generator=generator,
        )[:requested]

    def weighted_mean(
        self,
        value: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        volume = self.volume[indices]
        normalized = volume / volume.sum().clamp_min(1.0e-15)
        return torch.sum(normalized * value)

    def p2g_residual(
        self,
        value: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        volume = self.volume[indices]
        shape = self.shape[indices]
        nodes = self.node_ids[indices]
        contribution = volume[:, None] * shape * value[:, None]
        nodal_value = torch.zeros(
            self.n_nodes,
            dtype=value.dtype,
            device=value.device,
        ).index_add(
            0,
            nodes.reshape(-1),
            contribution.reshape(-1),
        )
        nodal_volume = torch.zeros(
            self.n_nodes,
            dtype=volume.dtype,
            device=volume.device,
        ).index_add(
            0,
            nodes.reshape(-1),
            (volume[:, None] * shape).reshape(-1),
        )
        active = nodal_volume > 1.0e-12
        return nodal_value[active] / nodal_volume[active]

    def summary(self) -> dict[str, float | int]:
        return {
            "particles": self.n_particles,
            "cells_x": self.cells_x,
            "cells_y": self.cells_y,
            "p2g_nodes": self.n_nodes,
            "dimensionless_length_x": self.length_x,
            "integration_area": float(
                self.volume.sum().detach().cpu()
            ),
        }


def _p2g_connectivity(
    xy: torch.Tensor,
    length_x: float,
    nx: int,
    ny: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = xy.dtype
    device = xy.device
    dx = length_x / nx
    dy = 1.0 / ny
    fx = (xy[:, 0] / dx).clamp(0.0, nx - 1.0e-6)
    fy = (xy[:, 1] / dy).clamp(0.0, ny - 1.0e-6)
    ix = torch.floor(fx).to(torch.long)
    iy = torch.floor(fy).to(torch.long)
    tx = fx - ix.to(dtype)
    ty = fy - iy.to(dtype)
    npx = nx + 1
    n00 = iy * npx + ix
    node_ids = torch.stack(
        (n00, n00 + 1, n00 + npx, n00 + npx + 1),
        dim=1,
    )
    shape = torch.stack(
        (
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ),
        dim=1,
    )
    node_x = torch.linspace(
        0.0, length_x, nx + 1, dtype=dtype, device=device
    )
    node_y = torch.linspace(
        0.0, 1.0, ny + 1, dtype=dtype, device=device
    )
    xx, yy = torch.meshgrid(node_x, node_y, indexing="xy")
    node_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return node_ids, shape, node_xy
