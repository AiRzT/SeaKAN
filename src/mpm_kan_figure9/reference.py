from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .config import Geometry, SoilParameters, WaveParameters, dimensionless_coefficients


@dataclass
class ReferenceResult:
    theta: np.ndarray
    normalized_pressure: np.ndarray
    x: np.ndarray
    y: np.ndarray
    displacement_x: np.ndarray
    displacement_y: np.ndarray
    pressure: np.ndarray
    relative_linear_residual: float
    degrees_of_freedom: int

    def profile(self) -> dict[str, np.ndarray]:
        return {
            "theta": self.theta,
            "normalized_pressure": self.normalized_pressure,
        }


def solve_reference(
    wave: WaveParameters,
    soil: SoilParameters,
    geometry: Geometry,
    *,
    nx: int = 129,
    ny: int = 33,
    n_theta: int = 721,
) -> ReferenceResult:
    """
    Independent sparse finite-difference u-w-p check.

    The circular interface is represented by rigid displacement ghosts and
    mirrored pressure fluxes. It is intentionally independent of the KAN and
    MPM quadrature and is used only after PINN training.
    """

    if nx < 33 or ny < 17:
        raise ValueError("Reference grid must be at least 33 by 17")
    half = geometry.half_width_over_height
    x = np.linspace(-half, half, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)
    active = (
        xx**2 + (yy - geometry.center_y) ** 2
        > (0.995 * geometry.radius_over_height) ** 2
    )
    unknown = -np.ones((ny, nx), dtype=np.int64)
    unknown[active] = np.arange(int(active.sum()))
    n_nodes = int(active.sum())
    n_dof = 3 * n_nodes
    matrix = lil_matrix((n_dof, n_dof), dtype=np.complex128)
    rhs = np.zeros(n_dof, dtype=np.complex128)
    coefficients = dimensionless_coefficients(wave, soil, geometry)
    lam = soil.lambda_over_g
    mass_p = coefficients["mass_pressure"]
    mass_e = coefficients["mass_strain"]

    def valid(iy: int, ix: int) -> bool:
        return 0 <= iy < ny and 0 <= ix < nx and bool(active[iy, ix])

    def dof(field: int, iy: int, ix: int) -> int:
        return 3 * int(unknown[iy, ix]) + field

    def add(row: int, field: int, iy: int, ix: int, value: complex) -> None:
        if valid(iy, ix):
            matrix[row, dof(field, iy, ix)] += value

    def dirichlet(
        row: int, field: int, iy: int, ix: int, value: complex = 0.0
    ) -> None:
        matrix[row, dof(field, iy, ix)] = 1.0
        rhs[row] = value

    def first(
        row: int,
        field: int,
        iy: int,
        ix: int,
        axis: str,
        coefficient: complex,
    ) -> None:
        if axis == "x":
            minus, plus, h = (iy, ix - 1), (iy, ix + 1), dx
        else:
            minus, plus, h = (iy - 1, ix), (iy + 1, ix), dy
        vm, vp = valid(*minus), valid(*plus)
        if vm and vp:
            add(row, field, *plus, coefficient / (2.0 * h))
            add(row, field, *minus, -coefficient / (2.0 * h))
        elif vp:
            add(row, field, *plus, coefficient / h)
            add(row, field, iy, ix, -coefficient / h)
        elif vm:
            add(row, field, iy, ix, coefficient / h)
            add(row, field, *minus, -coefficient / h)

    def second(
        row: int,
        field: int,
        iy: int,
        ix: int,
        axis: str,
        coefficient: complex,
    ) -> None:
        if axis == "x":
            minus, plus, h = (iy, ix - 1), (iy, ix + 1), dx
        else:
            minus, plus, h = (iy - 1, ix), (iy + 1, ix), dy
        if field == 2:
            # Mirrored missing pressure neighbor: no flux at the pipe.
            if valid(*minus):
                add(row, field, *minus, coefficient / h**2)
                add(row, field, iy, ix, -coefficient / h**2)
            if valid(*plus):
                add(row, field, *plus, coefficient / h**2)
                add(row, field, iy, ix, -coefficient / h**2)
            return
        # Missing displacement neighbor is a zero rigid-pipe ghost.
        add(row, field, iy, ix, -2.0 * coefficient / h**2)
        if valid(*minus):
            add(row, field, *minus, coefficient / h**2)
        if valid(*plus):
            add(row, field, *plus, coefficient / h**2)

    def cross(
        row: int,
        field: int,
        iy: int,
        ix: int,
        coefficient: complex,
    ) -> None:
        scale = coefficient / (4.0 * dx * dy)
        for oy, ox, sign in (
            (1, 1, 1.0),
            (1, -1, -1.0),
            (-1, 1, -1.0),
            (-1, -1, 1.0),
        ):
            add(row, field, iy + oy, ix + ox, sign * scale)

    for iy in range(ny):
        for ix in range(nx):
            if not valid(iy, ix):
                continue
            base = 3 * int(unknown[iy, ix])
            is_bottom = iy == 0
            is_top = iy == ny - 1
            is_side = ix == 0 or ix == nx - 1
            if is_bottom:
                dirichlet(base, 0, iy, ix)
                dirichlet(base + 1, 1, iy, ix)
                first(base + 2, 2, iy, ix, "y", 1.0)
                continue
            if is_top:
                dirichlet(
                    base + 2,
                    2,
                    iy,
                    ix,
                    np.exp(1j * coefficients["kappa"] * x[ix]),
                )
                first(base, 0, iy, ix, "x", lam)
                first(base, 1, iy, ix, "y", 2.0 + lam)
                first(base + 1, 0, iy, ix, "y", 1.0)
                first(base + 1, 1, iy, ix, "x", 1.0)
                continue
            if is_side:
                dirichlet(base, 0, iy, ix)
                dirichlet(base + 1, 1, iy, ix)
                first(base + 2, 2, iy, ix, "x", 1.0)
                continue

            # div(sigma' - p I) = 0
            second(base, 0, iy, ix, "x", 2.0 + lam)
            second(base, 0, iy, ix, "y", 1.0)
            cross(base, 1, iy, ix, lam + 1.0)
            first(base, 2, iy, ix, "x", -1.0)

            cross(base + 1, 0, iy, ix, lam + 1.0)
            second(base + 1, 1, iy, ix, "x", 1.0)
            second(base + 1, 1, iy, ix, "y", 2.0 + lam)
            first(base + 1, 2, iy, ix, "y", -1.0)

            # lap(P) + i*a P + i*b div(U) = 0
            second(base + 2, 2, iy, ix, "x", 1.0)
            second(base + 2, 2, iy, ix, "y", 1.0)
            add(base + 2, 2, iy, ix, 1j * mass_p)
            first(base + 2, 0, iy, ix, "x", 1j * mass_e)
            first(base + 2, 1, iy, ix, "y", 1j * mass_e)

    csr = matrix.tocsr()
    solution = spsolve(csr, rhs)
    residual = float(
        np.linalg.norm(csr @ solution - rhs) / max(np.linalg.norm(rhs), 1.0e-15)
    )
    displacement_x = np.full((ny, nx), np.nan + 0j, dtype=np.complex128)
    displacement_y = np.full((ny, nx), np.nan + 0j, dtype=np.complex128)
    pressure = np.full((ny, nx), np.nan + 0j, dtype=np.complex128)
    displacement_x[active] = solution[3 * unknown[active]]
    displacement_y[active] = solution[3 * unknown[active] + 1]
    pressure[active] = solution[3 * unknown[active] + 2]
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    wall = _sample_wall(
        theta, pressure, active, x, y, geometry, max(dx, dy)
    )
    return ReferenceResult(
        theta=theta,
        normalized_pressure=wall,
        x=x,
        y=y,
        displacement_x=displacement_x,
        displacement_y=displacement_y,
        pressure=pressure,
        relative_linear_residual=residual,
        degrees_of_freedom=n_dof,
    )


def _sample_wall(
    theta: np.ndarray,
    pressure: np.ndarray,
    active: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: Geometry,
    spacing: float,
) -> np.ndarray:
    radius = geometry.radius_over_height
    values = np.empty(theta.size, dtype=float)
    active_y, active_x = np.where(active)
    coordinates = np.column_stack((x[active_x], y[active_y]))
    active_values = pressure[active]
    for index, angle in enumerate(theta):
        normal = np.array([np.cos(angle), np.sin(angle)])
        wall = np.array(
            [radius * normal[0], geometry.center_y + radius * normal[1]]
        )
        samples: list[complex] = []
        weights: list[float] = []
        for multiplier in (0.8, 1.2, 1.8, 2.6, 3.5):
            point = wall + multiplier * spacing * normal
            value = _bilinear(pressure, active, x, y, point[0], point[1])
            if value is not None:
                samples.append(value)
                weights.append(1.0 / multiplier)
        if samples:
            sample_array = np.asarray(samples)
            weight_array = np.asarray(weights)
            values[index] = abs(
                np.sum(sample_array * weight_array) / np.sum(weight_array)
            )
            continue
        target = wall + 1.5 * spacing * normal
        nearest = np.argmin(np.sum((coordinates - target) ** 2, axis=1))
        values[index] = abs(active_values[nearest])
    return values


def _bilinear(
    pressure: np.ndarray,
    active: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sx: float,
    sy: float,
) -> complex | None:
    if sx < x[0] or sx > x[-1] or sy < y[0] or sy > y[-1]:
        return None
    ix = min(int(np.searchsorted(x, sx) - 1), x.size - 2)
    iy = min(int(np.searchsorted(y, sy) - 1), y.size - 2)
    ix = max(ix, 0)
    iy = max(iy, 0)
    if not np.all(active[iy : iy + 2, ix : ix + 2]):
        return None
    tx = (sx - x[ix]) / (x[ix + 1] - x[ix])
    ty = (sy - y[iy]) / (y[iy + 1] - y[iy])
    return (
        (1.0 - tx) * (1.0 - ty) * pressure[iy, ix]
        + tx * (1.0 - ty) * pressure[iy, ix + 1]
        + (1.0 - tx) * ty * pressure[iy + 1, ix]
        + tx * ty * pressure[iy + 1, ix + 1]
    )
