from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import Geometry
from .model import surface_y


@dataclass
class MaterialPointCloud:
    xy: torch.Tensor
    volume: torch.Tensor
    level: torch.Tensor
    node_ids: torch.Tensor
    shape: torch.Tensor
    n_nodes: int
    counts: tuple[int, ...]
    level_names: tuple[str, ...]
    bounds: tuple[float, float, float, float]

    @property
    def n_particles(self) -> int:
        return int(self.xy.shape[0])

    def sample_indices(
        self,
        requested_per_level: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if len(requested_per_level) != len(self.counts):
            raise ValueError("one batch size is required per level")
        selections: list[torch.Tensor] = []
        start = 0
        for count, requested in zip(
            self.counts, requested_per_level
        ):
            requested = min(int(requested), count)
            if requested == count:
                local = torch.arange(
                    count, device=self.xy.device
                )
            else:
                local = torch.randperm(
                    count,
                    device=self.xy.device,
                    generator=generator,
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
            raise ValueError("one weight is required per level")
        levels = self.level[indices]
        volumes = self.volume[indices]
        result = torch.zeros(
            (), dtype=value.dtype, device=value.device
        )
        for current, weight in enumerate(level_weights):
            mask = levels == current
            if not bool(mask.any()):
                continue
            local = volumes[mask]
            normalized = local / local.sum().clamp_min(1.0e-15)
            result = result + weight * torch.sum(
                normalized * value[mask]
            )
        return result

    def p2g_residual(
        self, value: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        volume = self.volume[indices]
        shape = self.shape[indices]
        nodes = self.node_ids[indices]
        contribution = (
            volume[:, None] * shape * value[:, None]
        )
        nodal_value = torch.zeros(
            self.n_nodes,
            dtype=value.dtype,
            device=value.device,
        ).index_add(
            0, nodes.reshape(-1), contribution.reshape(-1)
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

    def summary(self) -> dict[str, object]:
        return {
            "particles": self.n_particles,
            "levels": dict(zip(self.level_names, self.counts)),
            "p2g_nodes": self.n_nodes,
            "bounds": list(self.bounds),
        }


def create_soil_cloud(
    geometry: Geometry,
    device: torch.device,
    dtype: torch.dtype,
    *,
    base_x: int = 40,
    base_y: int = 24,
    middle_radial: int = 6,
    middle_theta: int = 96,
    fine_radial: int = 6,
    fine_theta: int = 160,
    p2g_x: int = 80,
    p2g_y: int = 48,
) -> MaterialPointCloud:
    half = geometry.half_width_over_height
    base_xy, base_volume = _rectangular_points(
        -half,
        half,
        0.0,
        1.0,
        base_x,
        base_y,
        device,
        dtype,
    )
    base_mask = _soil_mask(base_xy, geometry)
    base_xy = base_xy[base_mask]
    base_volume = base_volume[base_mask]
    radius = geometry.radius_over_height
    middle_xy, middle_volume = _annulus(
        radius + 0.004,
        radius + 0.30,
        middle_radial,
        middle_theta,
        geometry,
        device,
        dtype,
    )
    middle_mask = _soil_mask(middle_xy, geometry)
    middle_xy = middle_xy[middle_mask]
    middle_volume = middle_volume[middle_mask]
    fine_xy, fine_volume = _annulus(
        radius + 0.001,
        radius + 0.09,
        fine_radial,
        fine_theta,
        geometry,
        device,
        dtype,
    )
    fine_mask = _soil_mask(fine_xy, geometry)
    fine_xy = fine_xy[fine_mask]
    fine_volume = fine_volume[fine_mask]
    return _assemble(
        (base_xy, middle_xy, fine_xy),
        (base_volume, middle_volume, fine_volume),
        ("background", "middle_pipe_annulus", "fine_pipe_annulus"),
        (-half, half, 0.0, 1.0),
        p2g_x,
        p2g_y,
    )


def create_water_cloud(
    geometry: Geometry,
    device: torch.device,
    dtype: torch.dtype,
    *,
    base_x: int = 40,
    base_y: int = 24,
    annulus_radial: int = 6,
    annulus_theta: int = 128,
    p2g_x: int = 80,
    p2g_y: int = 48,
) -> MaterialPointCloud:
    half = geometry.half_width_over_height
    y_min = min(
        geometry.pipe_center_y,
        geometry.backfill_surface_z
        / geometry.seabed_thickness
        + 1.0,
    )
    y_max = geometry.water_top_y
    base_xy, base_volume = _rectangular_points(
        -half,
        half,
        y_min,
        y_max,
        base_x,
        base_y,
        device,
        dtype,
    )
    base_mask = _water_mask(base_xy, geometry)
    base_xy = base_xy[base_mask]
    base_volume = base_volume[base_mask]
    radius = geometry.radius_over_height
    annulus_xy, annulus_volume = _annulus(
        radius + 0.001,
        radius + 0.22,
        annulus_radial,
        annulus_theta,
        geometry,
        device,
        dtype,
    )
    annulus_mask = _water_mask(
        annulus_xy, geometry
    )
    annulus_xy = annulus_xy[annulus_mask]
    annulus_volume = annulus_volume[annulus_mask]
    return _assemble(
        (base_xy, annulus_xy),
        (base_volume, annulus_volume),
        ("water_background", "exposed_pipe_annulus"),
        (-half, half, y_min, y_max),
        p2g_x,
        p2g_y,
    )


def create_rans_water_cloud(
    geometry: Geometry,
    device: torch.device,
    dtype: torch.dtype,
    *,
    base_x: int = 48,
    base_y: int = 30,
    annulus_radial: int = 8,
    annulus_theta: int = 160,
    boundary_layer_radial: int = 10,
    boundary_layer_theta: int = 224,
    p2g_x: int = 96,
    p2g_y: int = 60,
) -> MaterialPointCloud:
    """Water particles with a fourth-level pipe boundary-layer refinement."""

    half = geometry.half_width_over_height
    y_min = min(
        geometry.pipe_center_y,
        geometry.backfill_surface_z
        / geometry.seabed_thickness
        + 1.0,
    )
    y_max = geometry.water_top_y
    base_xy, base_volume = _rectangular_points(
        -half,
        half,
        y_min,
        y_max,
        base_x,
        base_y,
        device,
        dtype,
    )
    base_mask = _water_mask(base_xy, geometry)
    base_xy = base_xy[base_mask]
    base_volume = base_volume[base_mask]
    radius = geometry.radius_over_height
    annulus_xy, annulus_volume = _annulus(
        radius + 0.001,
        radius + 0.22,
        annulus_radial,
        annulus_theta,
        geometry,
        device,
        dtype,
    )
    annulus_mask = _water_mask(annulus_xy, geometry)
    annulus_xy = annulus_xy[annulus_mask]
    annulus_volume = annulus_volume[annulus_mask]
    boundary_xy, boundary_volume = _annulus(
        radius + 0.0002,
        radius + 0.05,
        boundary_layer_radial,
        boundary_layer_theta,
        geometry,
        device,
        dtype,
    )
    boundary_mask = _water_mask(boundary_xy, geometry)
    boundary_xy = boundary_xy[boundary_mask]
    boundary_volume = boundary_volume[boundary_mask]
    return _assemble(
        (base_xy, annulus_xy, boundary_xy),
        (base_volume, annulus_volume, boundary_volume),
        (
            "water_background",
            "exposed_pipe_annulus",
            "pipe_boundary_layer_level4",
        ),
        (-half, half, y_min, y_max),
        p2g_x,
        p2g_y,
    )


def _soil_mask(
    xy: torch.Tensor, geometry: Geometry
) -> torch.Tensor:
    outside_pipe = (
        xy[:, 0].square()
        + (xy[:, 1] - geometry.pipe_center_y).square()
        > geometry.radius_over_height**2
    )
    return (
        (xy[:, 1] < surface_y(xy[:, 0], geometry))
        & outside_pipe
    )


def _water_mask(
    xy: torch.Tensor, geometry: Geometry
) -> torch.Tensor:
    outside_pipe = (
        xy[:, 0].square()
        + (xy[:, 1] - geometry.pipe_center_y).square()
        > geometry.radius_over_height**2
    )
    return (
        (xy[:, 1] > surface_y(xy[:, 0], geometry))
        & outside_pipe
    )


def _rectangular_points(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    nx: int,
    ny: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    ix = torch.arange(nx, device=device, dtype=dtype)
    iy = torch.arange(ny, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(
        x0 + (ix + 0.5) * dx,
        y0 + (iy + 0.5) * dy,
        indexing="xy",
    )
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    volume = torch.full(
        (xy.shape[0],),
        dx * dy,
        device=device,
        dtype=dtype,
    )
    return xy, volume


def _annulus(
    inner: float,
    outer: float,
    n_radial: int,
    n_theta: int,
    geometry: Geometry,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = torch.linspace(
        inner, outer, n_radial + 1, device=device, dtype=dtype
    )
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        n_theta + 1,
        device=device,
        dtype=dtype,
    )
    r0 = edges[:-1]
    r1 = edges[1:]
    radius = (2.0 / 3.0) * (
        (r1**3 - r0**3)
        / (r1**2 - r0**2).clamp_min(1.0e-15)
    )
    theta = 0.5 * (angles[:-1] + angles[1:])
    rr, tt = torch.meshgrid(radius, theta, indexing="ij")
    x = rr * torch.cos(tt)
    y = geometry.pipe_center_y + rr * torch.sin(tt)
    area = (
        0.5
        * (r1.square() - r0.square())[:, None]
        * (2.0 * math.pi / n_theta)
        * torch.ones(
            (n_radial, n_theta), device=device, dtype=dtype
        )
    )
    return (
        torch.stack((x.reshape(-1), y.reshape(-1)), dim=1),
        area.reshape(-1),
    )


def _assemble(
    point_sets: tuple[torch.Tensor, ...],
    volume_sets: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    bounds: tuple[float, float, float, float],
    p2g_x: int,
    p2g_y: int,
) -> MaterialPointCloud:
    xy = torch.cat(point_sets, dim=0)
    volume = torch.cat(volume_sets, dim=0)
    counts = tuple(int(points.shape[0]) for points in point_sets)
    level = torch.repeat_interleave(
        torch.arange(
            len(counts), device=xy.device, dtype=torch.long
        ),
        torch.tensor(counts, device=xy.device, dtype=torch.long),
    )
    node_ids, shape = _connectivity(
        xy, bounds, p2g_x, p2g_y
    )
    return MaterialPointCloud(
        xy=xy,
        volume=volume,
        level=level,
        node_ids=node_ids,
        shape=shape,
        n_nodes=(p2g_x + 1) * (p2g_y + 1),
        counts=counts,
        level_names=names,
        bounds=bounds,
    )


def _connectivity(
    xy: torch.Tensor,
    bounds: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, x1, y0, y1 = bounds
    dtype = xy.dtype
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    fx = ((xy[:, 0] - x0) / dx).clamp(0.0, nx - 1.0e-6)
    fy = ((xy[:, 1] - y0) / dy).clamp(0.0, ny - 1.0e-6)
    ix = torch.floor(fx).to(torch.long)
    iy = torch.floor(fy).to(torch.long)
    tx = fx - ix.to(dtype)
    ty = fy - iy.to(dtype)
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
