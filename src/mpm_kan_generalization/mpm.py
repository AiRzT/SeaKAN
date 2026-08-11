from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .cases import CaseTable
from .config import PhysicalConstants


@dataclass
class ParametricCloud:
    xy: torch.Tensor
    case_index: torch.Tensor
    volume: torch.Tensor
    level: torch.Tensor
    node_ids: torch.Tensor
    shape: torch.Tensor
    n_nodes: int
    n_cases: int

    @property
    def n_particles(self) -> int:
        return int(self.xy.shape[0])

    def sample(
        self,
        case_count: int,
        background_per_case: int,
        annulus_per_case: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_cases = torch.randperm(
            self.n_cases,
            generator=generator,
            device=self.xy.device,
        )[: min(case_count, self.n_cases)]
        selections: list[torch.Tensor] = []
        for case in selected_cases:
            for level, requested in (
                (0, background_per_case),
                (1, annulus_per_case),
            ):
                available = torch.nonzero(
                    (self.case_index == case)
                    & (self.level == level),
                    as_tuple=False,
                ).squeeze(1)
                if available.numel() > requested:
                    order = torch.randperm(
                        available.numel(),
                        generator=generator,
                        device=self.xy.device,
                    )[:requested]
                    available = available[order]
                selections.append(available)
        return torch.cat(selections), selected_cases

    def weighted_level_mean(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros(
            (), dtype=values.dtype, device=values.device
        )
        for level, level_weight in ((0, 0.55), (1, 0.45)):
            mask = self.level[indices] == level
            local_volume = self.volume[indices][mask]
            normalized = local_volume / local_volume.sum().clamp_min(
                1.0e-15
            )
            result = result + level_weight * torch.sum(
                normalized * values[mask]
            )
        return result

    def p2g(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        volume = self.volume[indices]
        shape = self.shape[indices]
        nodes = self.node_ids[indices]
        contribution = (
            volume[:, None] * shape * values[:, None]
        )
        nodal_value = torch.zeros(
            self.n_nodes,
            dtype=values.dtype,
            device=values.device,
        ).index_add_(
            0,
            nodes.reshape(-1),
            contribution.reshape(-1),
        )
        nodal_volume = torch.zeros(
            self.n_nodes,
            dtype=volume.dtype,
            device=volume.device,
        ).index_add_(
            0,
            nodes.reshape(-1),
            (volume[:, None] * shape).reshape(-1),
        )
        active = nodal_volume > 1.0e-12
        return nodal_value[active] / nodal_volume[active]


def _rectangular_points(
    constants: PhysicalConstants,
    nx: int,
    ny: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = constants.half_width
    dx = 2.0 * half / nx
    dy = 1.0 / ny
    x = torch.arange(nx, device=device, dtype=dtype)
    y = torch.arange(ny, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(
        -half + (x + 0.5) * dx,
        (y + 0.5) * dy,
        indexing="xy",
    )
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    volume = torch.full(
        (xy.shape[0],),
        dx * dy,
        device=device,
        dtype=dtype,
    )
    return xy, volume


def _annulus_points(
    center_y: float,
    constants: PhysicalConstants,
    radial: int,
    angular: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = constants.radius
    edges = torch.linspace(
        radius + 0.002,
        radius + 0.22,
        radial + 1,
        device=device,
        dtype=dtype,
    )
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        angular + 1,
        device=device,
        dtype=dtype,
    )
    r0, r1 = edges[:-1], edges[1:]
    rr = 0.5 * (r0 + r1)
    tt = 0.5 * (angles[:-1] + angles[1:])
    grid_r, grid_t = torch.meshgrid(
        rr, tt, indexing="ij"
    )
    x = grid_r * torch.cos(grid_t)
    y = center_y + grid_r * torch.sin(grid_t)
    area = (
        0.5
        * (r1.square() - r0.square())[:, None]
        * (2.0 * math.pi / angular)
        * torch.ones(
            (radial, angular),
            device=device,
            dtype=dtype,
        )
    )
    xy = torch.stack((x.flatten(), y.flatten()), dim=1)
    keep = (
        (xy[:, 1] > 0.0)
        & (xy[:, 1] < 1.0)
        & (xy[:, 0].abs() < constants.half_width)
    )
    return xy[keep], area.flatten()[keep]


def _connectivity(
    xy: torch.Tensor,
    constants: PhysicalConstants,
    nx: int,
    ny: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = constants.half_width
    dx = 2.0 * half / nx
    dy = 1.0 / ny
    fx = ((xy[:, 0] + half) / dx).clamp(
        0.0, nx - 1.0e-6
    )
    fy = (xy[:, 1] / dy).clamp(0.0, ny - 1.0e-6)
    ix, iy = torch.floor(fx).long(), torch.floor(fy).long()
    tx, ty = fx - ix.to(xy.dtype), fy - iy.to(xy.dtype)
    npx = nx + 1
    n00 = iy * npx + ix
    nodes = torch.stack(
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
    return nodes, shape


def create_parametric_cloud(
    table: CaseTable,
    constants: PhysicalConstants,
    *,
    base_x: int = 24,
    base_y: int = 14,
    annulus_radial: int = 3,
    annulus_theta: int = 48,
    p2g_x: int = 32,
    p2g_y: int = 20,
) -> ParametricCloud:
    device = table.parameters.device
    dtype = table.parameters.dtype
    point_sets: list[torch.Tensor] = []
    volume_sets: list[torch.Tensor] = []
    case_sets: list[torch.Tensor] = []
    level_sets: list[torch.Tensor] = []
    node_sets: list[torch.Tensor] = []
    shape_sets: list[torch.Tensor] = []
    nodes_per_case = (p2g_x + 1) * (p2g_y + 1)
    for case in range(table.n_cases):
        center = float(table.pipe_center_y[case].detach().cpu())
        base_xy, base_volume = _rectangular_points(
            constants, base_x, base_y, device, dtype
        )
        outside = (
            base_xy[:, 0].square()
            + (base_xy[:, 1] - center).square()
            > constants.radius**2
        )
        base_xy, base_volume = (
            base_xy[outside],
            base_volume[outside],
        )
        ann_xy, ann_volume = _annulus_points(
            center,
            constants,
            annulus_radial,
            annulus_theta,
            device,
            dtype,
        )
        for level, (xy, volume) in enumerate(
            (
                (base_xy, base_volume),
                (ann_xy, ann_volume),
            )
        ):
            nodes, shape = _connectivity(
                xy, constants, p2g_x, p2g_y
            )
            point_sets.append(xy)
            volume_sets.append(volume)
            case_sets.append(
                torch.full(
                    (xy.shape[0],),
                    case,
                    device=device,
                    dtype=torch.long,
                )
            )
            level_sets.append(
                torch.full(
                    (xy.shape[0],),
                    level,
                    device=device,
                    dtype=torch.long,
                )
            )
            node_sets.append(nodes + case * nodes_per_case)
            shape_sets.append(shape)
    return ParametricCloud(
        xy=torch.cat(point_sets),
        case_index=torch.cat(case_sets),
        volume=torch.cat(volume_sets),
        level=torch.cat(level_sets),
        node_ids=torch.cat(node_sets),
        shape=torch.cat(shape_sets),
        n_nodes=nodes_per_case * table.n_cases,
        n_cases=table.n_cases,
    )
