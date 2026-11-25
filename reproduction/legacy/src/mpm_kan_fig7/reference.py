from __future__ import annotations

import numpy as np

from .config import Fig7Case, SoilParameters, WaveParameters


def solve_reference(
    wave: WaveParameters,
    soil: SoilParameters,
    case: Fig7Case,
    n_depth: int = 601,
) -> dict[str, np.ndarray]:
    """Independent fine-grid complex finite-difference reference solution."""
    if n_depth < 16:
        raise ValueError("n_depth must be at least 16")
    n = n_depth
    h = soil.thickness
    z = np.linspace(-h, 0.0, n)
    dz = z[1] - z[0]
    k = wave.wave_number
    omega = wave.omega
    ndof = 3 * n
    a = np.zeros((ndof, ndof), dtype=np.complex128)
    b = np.zeros(ndof, dtype=np.complex128)

    g = soil.shear_modulus
    lam = soil.lame_lambda
    d11 = 2.0 * g + lam
    d12 = lam
    d21 = lam
    d22 = 2.0 * g + lam
    d33 = g
    hydraulic = case.permeability / wave.gamma_w
    storage = soil.porosity * case.pore_water_compressibility(wave, soil)

    def iu(j: int) -> int:
        return 3 * j

    def iw(j: int) -> int:
        return 3 * j + 1

    def ip(j: int) -> int:
        return 3 * j + 2

    row = 0
    a[row, iu(0)] = 1.0
    row += 1
    a[row, iw(0)] = 1.0
    row += 1
    a[row, ip(0)] = -3.0 / (2.0 * dz)
    a[row, ip(1)] = 4.0 / (2.0 * dz)
    a[row, ip(2)] = -1.0 / (2.0 * dz)
    row += 1

    ik = 1j * k
    for j in range(1, n - 1):
        a[row, iu(j - 1)] += d33 / dz**2
        a[row, iu(j)] += -2.0 * d33 / dz**2 - d11 * k**2
        a[row, iu(j + 1)] += d33 / dz**2
        a[row, iw(j - 1)] += ik * (d12 + d33) * (-1.0 / (2.0 * dz))
        a[row, iw(j + 1)] += ik * (d12 + d33) * (1.0 / (2.0 * dz))
        a[row, ip(j)] += -ik
        row += 1

        a[row, iw(j - 1)] += d22 / dz**2
        a[row, iw(j)] += -2.0 * d22 / dz**2 - d33 * k**2
        a[row, iw(j + 1)] += d22 / dz**2
        a[row, iu(j - 1)] += ik * (d33 + d21) * (-1.0 / (2.0 * dz))
        a[row, iu(j + 1)] += ik * (d33 + d21) * (1.0 / (2.0 * dz))
        a[row, ip(j - 1)] += 1.0 / (2.0 * dz)
        a[row, ip(j + 1)] += -1.0 / (2.0 * dz)
        row += 1

        a[row, ip(j - 1)] += hydraulic / dz**2
        a[row, ip(j)] += (
            -2.0 * hydraulic / dz**2
            - hydraulic * k**2
            + 1j * omega * storage
        )
        a[row, ip(j + 1)] += hydraulic / dz**2
        a[row, iu(j)] += 1j * omega * ik
        a[row, iw(j - 1)] += 1j * omega * (-1.0 / (2.0 * dz))
        a[row, iw(j + 1)] += 1j * omega * (1.0 / (2.0 * dz))
        row += 1

    jt = n - 1
    top_derivative = (
        (jt, 3.0 / (2.0 * dz)),
        (n - 2, -4.0 / (2.0 * dz)),
        (n - 3, 1.0 / (2.0 * dz)),
    )
    a[row, ip(jt)] = 1.0
    b[row] = wave.pressure_amplitude
    row += 1
    a[row, iu(jt)] += d21 * ik
    for jj, coefficient in top_derivative:
        a[row, iw(jj)] += d22 * coefficient
    row += 1
    a[row, iw(jt)] += d33 * ik
    for jj, coefficient in top_derivative:
        a[row, iu(jj)] += d33 * coefficient

    solution = np.linalg.solve(a, b)
    u = solution[0::3]
    w = solution[1::3]
    p = solution[2::3]
    du_dz = _derivative(u, dz)
    dw_dz = _derivative(w, dz)
    eps_x = ik * u
    eps_z = dw_dz
    gamma_xz = ik * w + du_dz
    sigma_x = (2.0 * g + lam) * eps_x + lam * eps_z
    sigma_z = lam * eps_x + (2.0 * g + lam) * eps_z
    tau_xz = g * gamma_xz
    p0 = wave.pressure_amplitude
    return {
        "z_over_h": z / h,
        "pore_pressure": np.abs(p) / p0,
        "sigma_x_eff": np.abs(sigma_x) / p0,
        "sigma_z_eff": np.abs(sigma_z) / p0,
        "tau_xz": np.abs(tau_xz) / p0,
    }


def _derivative(values: np.ndarray, spacing: float) -> np.ndarray:
    derivative = np.empty_like(values)
    derivative[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (
        2.0 * spacing
    )
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * spacing)
    derivative[-1] = (
        3.0 * values[-1] - 4.0 * values[-2] + values[-3]
    ) / (2.0 * spacing)
    return derivative

