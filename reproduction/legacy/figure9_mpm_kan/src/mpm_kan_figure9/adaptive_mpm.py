from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import Geometry


@dataclass
class AdaptiveMaterialPointCloud:
    """
    Adaptive material-point cloud around the circular pipe.

    Level 0 covers the full trench with Cartesian material points. Levels 1
    and 2 add progressively finer polar particles in annuli around the pipe.
    An optional fourth level fills the narrow pipe-top-to-mudline gap with
    body-fitted, cosine-clustered sectors. Every particle carries an
    integration area and bilinear P2G connectivity.
    """

    xy: torch.Tensor
    volume: torch.Tensor
    level: torch.Tensor
    node_ids: torch.Tensor
    shape: torch.Tensor
    n_nodes: int
    node_xy: torch.Tensor
    counts: tuple[int, ...]
    level_names: tuple[str, ...]

    @classmethod
    def create(
        cls,
        geometry: Geometry,
        device: torch.device,
        dtype: torch.dtype,
        *,
        base_x: int = 48,
        base_y: int = 18,
        middle_radial: int = 8,
        middle_theta: int = 128,
        fine_radial: int = 7,
        fine_theta: int = 192,
        gap_normal: int = 0,
        gap_theta: int = 0,
        p2g_x: int = 96,
        p2g_y: int = 36,
    ) -> "AdaptiveMaterialPointCloud":
        if base_x < 24 or base_y < 12:
            raise ValueError("base grid must be at least 24 by 12")
        half = geometry.half_width_over_height
        radius = geometry.radius_over_height
        center_y = geometry.center_y

        base_dx = 2.0 * half / base_x
        base_dy = 1.0 / base_y
        ix = torch.arange(base_x, device=device, dtype=dtype)
        iy = torch.arange(base_y, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(
            -half + (ix + 0.5) * base_dx,
            (iy + 0.5) * base_dy,
            indexing="xy",
        )
        base_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        outside = (
            base_xy[:, 0].square()
            + (base_xy[:, 1] - center_y).square()
            > radius**2
        )
        base_xy = base_xy[outside]
        base_volume = torch.full(
            (base_xy.shape[0],), base_dx * base_dy, dtype=dtype, device=device
        )

        middle_xy, middle_volume = _polar_annulus(
            radius + 0.025,
            radius + 0.34,
            middle_radial,
            middle_theta,
            center_y,
            half,
            device,
            dtype,
        )
        fine_xy, fine_volume = _polar_annulus(
            radius + 0.0015,
            radius + 0.085,
            fine_radial,
            fine_theta,
            center_y,
            half,
            device,
            dtype,
        )
        point_sets = [base_xy, middle_xy, fine_xy]
        volume_sets = [base_volume, middle_volume, fine_volume]
        level_names = ["background", "middle_annulus", "fine_pipe_annulus"]
        if gap_normal > 0 or gap_theta > 0:
            if gap_normal < 2 or gap_theta < 16:
                raise ValueError(
                    "top-gap refinement requires gap_normal >= 2 and "
                    "gap_theta >= 16"
                )
            gap_xy, gap_volume = _top_gap_patch(
                radius,
                center_y,
                gap_normal,
                gap_theta,
                half,
                device,
                dtype,
            )
            point_sets.append(gap_xy)
            volume_sets.append(gap_volume)
            level_names.append("top_gap")

        xy = torch.cat(point_sets, dim=0)
        volume = torch.cat(volume_sets, dim=0)
        counts = tuple(points.shape[0] for points in point_sets)
        level = torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.long),
            torch.tensor(counts, device=device, dtype=torch.long),
        )

        node_ids, shape, node_xy = _p2g_connectivity(
            xy, half, p2g_x, p2g_y
        )
        return cls(
            xy=xy,
            volume=volume,
            level=level,
            node_ids=node_ids,
            shape=shape,
            n_nodes=(p2g_x + 1) * (p2g_y + 1),
            node_xy=node_xy,
            counts=counts,
            level_names=tuple(level_names),
        )

    @property
    def n_particles(self) -> int:
        return self.xy.shape[0]

    def sample_indices(
        self,
        batch_per_level: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if len(batch_per_level) != len(self.counts):
            raise ValueError(
                "batch_per_level must contain one value per particle level"
            )
        selections: list[torch.Tensor] = []
        start = 0
        for count, requested in zip(self.counts, batch_per_level):
            requested = min(requested, count)
            if requested == count:
                local = torch.arange(count, device=self.xy.device)
            else:
                local = torch.randperm(
                    count, device=self.xy.device, generator=generator
                )[:requested]
            selections.append(local + start)
            start += count
        return torch.cat(selections)

    def weighted_level_mean(
        self,
        value: torch.Tensor,
        indices: torch.Tensor,
        level_weights: tuple[float, ...],
    ) -> torch.Tensor:
        if len(level_weights) != len(self.counts):
            raise ValueError(
                "level_weights must contain one value per particle level"
            )
        levels = self.level[indices]
        volumes = self.volume[indices]
        result = torch.zeros((), dtype=value.dtype, device=value.device)
        for current_level, level_weight in enumerate(level_weights):
            mask = levels == current_level
            if not bool(mask.any()):
                continue
            local_volume = volumes[mask]
            normalized = local_volume / local_volume.sum().clamp_min(1.0e-12)
            result = result + level_weight * torch.sum(normalized * value[mask])
        return result

    def p2g_residual(
        self, value: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        volume = self.volume[indices]
        shape = self.shape[indices]
        nodes = self.node_ids[indices]
        contribution = volume[:, None] * shape * value[:, None]
        nodal_value = torch.zeros(
            self.n_nodes, dtype=value.dtype, device=value.device
        ).index_add(0, nodes.reshape(-1), contribution.reshape(-1))
        nodal_volume = torch.zeros(
            self.n_nodes, dtype=volume.dtype, device=volume.device
        ).index_add(
            0, nodes.reshape(-1), (volume[:, None] * shape).reshape(-1)
        )
        active = nodal_volume > 1.0e-10
        return nodal_value[active] / nodal_volume[active], active

    def summary(self) -> dict[str, object]:
        return {
            "particles": self.n_particles,
            "levels": dict(zip(self.level_names, self.counts)),
            "integration_area_by_level": [
                float(self.volume[self.level == level].sum().detach().cpu())
                for level in range(len(self.counts))
            ],
            "p2g_nodes": self.n_nodes,
        }


def _polar_annulus(
    inner: float,
    outer: float,
    n_radial: int,
    n_theta: int,
    center_y: float,
    half_width: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    radial_edges = torch.linspace(
        inner, outer, n_radial + 1, dtype=dtype, device=device
    )
    angular_edges = torch.linspace(
        0.0, 2.0 * math.pi, n_theta + 1, dtype=dtype, device=device
    )
    r0 = radial_edges[:-1]
    r1 = radial_edges[1:]
    # Area-centroid radius for an annular sector.
    radial_center = (2.0 / 3.0) * (
        (r1**3 - r0**3) / (r1**2 - r0**2).clamp_min(1.0e-12)
    )
    theta = 0.5 * (angular_edges[:-1] + angular_edges[1:])
    rr, tt = torch.meshgrid(radial_center, theta, indexing="ij")
    x = rr * torch.cos(tt)
    y = center_y + rr * torch.sin(tt)
    area_radial = 0.5 * (r1**2 - r0**2)
    area = (
        area_radial[:, None]
        * (2.0 * math.pi / n_theta)
        * torch.ones((n_radial, n_theta), dtype=dtype, device=device)
    )
    xy = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
    volume = area.reshape(-1)
    inside_domain = (
        (xy[:, 0].abs() < half_width)
        & (xy[:, 1] > 0.0)
        & (xy[:, 1] < 1.0)
    )
    return xy[inside_domain], volume[inside_domain]


def _top_gap_patch(
    pipe_radius: float,
    center_y: float,
    n_normal: int,
    n_theta: int,
    half_width: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Body-fitted sectors spanning the pipe-top-to-mudline gap.

    Each angular ray starts on the pipe and ends exactly at Y=1. Cosine
    spacing clusters integration points near both physical boundaries where
    the pressure-gradient and traction residuals are most sensitive.
    """

    theta_edges = torch.linspace(
        0.25 * math.pi,
        0.75 * math.pi,
        n_theta + 1,
        dtype=dtype,
        device=device,
    )
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    delta_theta = theta_edges[1:] - theta_edges[:-1]
    radial_outer = (1.0 - center_y) / torch.sin(theta).clamp_min(1.0e-12)
    gap = radial_outer - pipe_radius
    if bool((gap <= 0.0).any()):
        raise ValueError("pipe intersects the mudline; no positive top gap exists")

    fractions = torch.linspace(
        0.0, 1.0, n_normal + 1, dtype=dtype, device=device
    )
    fractions = 0.5 * (1.0 - torch.cos(math.pi * fractions))
    r0 = pipe_radius + fractions[:-1, None] * gap[None, :]
    r1 = pipe_radius + fractions[1:, None] * gap[None, :]
    radial_center = (2.0 / 3.0) * (
        (r1**3 - r0**3) / (r1**2 - r0**2).clamp_min(1.0e-15)
    )
    x = radial_center * torch.cos(theta)[None, :]
    y = center_y + radial_center * torch.sin(theta)[None, :]
    area = 0.5 * (r1**2 - r0**2) * delta_theta[None, :]
    xy = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
    volume = area.reshape(-1)
    inside_domain = (
        (xy[:, 0].abs() < half_width)
        & (xy[:, 1] > 0.0)
        & (xy[:, 1] < 1.0)
    )
    return xy[inside_domain], volume[inside_domain]


def _p2g_connectivity(
    xy: torch.Tensor, half_width: float, nx: int, ny: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = xy.dtype
    device = xy.device
    dx = 2.0 * half_width / nx
    dy = 1.0 / ny
    fx = ((xy[:, 0] + half_width) / dx).clamp(0.0, nx - 1.0e-6)
    fy = (xy[:, 1] / dy).clamp(0.0, ny - 1.0e-6)
    ix = torch.floor(fx).to(torch.long)
    iy = torch.floor(fy).to(torch.long)
    tx = fx - ix.to(dtype)
    ty = fy - iy.to(dtype)
    npx = nx + 1
    n00 = iy * npx + ix
    node_ids = torch.stack((n00, n00 + 1, n00 + npx, n00 + npx + 1), dim=1)
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
        -half_width, half_width, nx + 1, dtype=dtype, device=device
    )
    node_y = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(node_x, node_y, indexing="xy")
    node_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return node_ids, shape, node_xy
