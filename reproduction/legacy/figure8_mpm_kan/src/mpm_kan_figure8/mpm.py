from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaterialPointGrid1D:
    """
    One-dimensional background grid with Gaussian material points per cell.

    The grid coordinate y ranges from 0 (rigid bottom) to 1 (mudline).
    """

    n_cells: int
    particles_per_cell: int
    y: torch.Tensor
    volume: torch.Tensor
    node_ids: torch.Tensor
    shape: torch.Tensor
    shape_gradient: torch.Tensor
    nodal_volume: torch.Tensor

    @classmethod
    def create(
        cls,
        n_cells: int,
        particles_per_cell: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "MaterialPointGrid1D":
        if n_cells < 8:
            raise ValueError("n_cells must be at least 8")
        if particles_per_cell not in (1, 2, 3):
            raise ValueError("particles_per_cell must be 1, 2, or 3")

        if particles_per_cell == 1:
            gauss_x = torch.tensor([0.0], dtype=dtype, device=device)
            gauss_w = torch.tensor([2.0], dtype=dtype, device=device)
        elif particles_per_cell == 2:
            point = 1.0 / (3.0**0.5)
            gauss_x = torch.tensor([-point, point], dtype=dtype, device=device)
            gauss_w = torch.ones(2, dtype=dtype, device=device)
        else:
            point = (3.0 / 5.0) ** 0.5
            gauss_x = torch.tensor([-point, 0.0, point], dtype=dtype, device=device)
            gauss_w = torch.tensor([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=dtype, device=device)

        dy = 1.0 / n_cells
        cell_ids = torch.arange(n_cells, device=device).repeat_interleave(particles_per_cell)
        local_xi = gauss_x.repeat(n_cells)
        local_weight = gauss_w.repeat(n_cells)
        y = (cell_ids.to(dtype) + 0.5 * (local_xi + 1.0)) * dy
        volume = 0.5 * dy * local_weight

        left_node = cell_ids
        right_node = cell_ids + 1
        node_ids = torch.stack((left_node, right_node), dim=1)
        local_y = 0.5 * (local_xi + 1.0)
        shape = torch.stack((1.0 - local_y, local_y), dim=1)
        shape_gradient = torch.empty_like(shape)
        shape_gradient[:, 0] = -1.0 / dy
        shape_gradient[:, 1] = 1.0 / dy

        nodal_volume = torch.zeros(n_cells + 1, dtype=dtype, device=device)
        flat_nodes = node_ids.reshape(-1)
        flat_weight = (volume[:, None] * shape).reshape(-1)
        nodal_volume = nodal_volume.index_add(0, flat_nodes, flat_weight)

        return cls(
            n_cells=n_cells,
            particles_per_cell=particles_per_cell,
            y=y[:, None],
            volume=volume,
            node_ids=node_ids,
            shape=shape,
            shape_gradient=shape_gradient,
            nodal_volume=nodal_volume,
        )

    @property
    def n_nodes(self) -> int:
        return self.n_cells + 1

    @property
    def n_particles(self) -> int:
        return self.y.shape[0]

    def p2g(self, particle_value: torch.Tensor, use_gradient: bool = False) -> torch.Tensor:
        """Integrate scalar/complex particle values against N or dN/dy."""
        basis = self.shape_gradient if use_gradient else self.shape
        contribution = self.volume[:, None] * basis * particle_value[:, None]
        nodes = self.node_ids.reshape(-1)
        values = contribution.reshape(-1)
        out = torch.zeros(
            self.n_nodes, dtype=particle_value.dtype, device=particle_value.device
        )
        return out.index_add(0, nodes, values)

