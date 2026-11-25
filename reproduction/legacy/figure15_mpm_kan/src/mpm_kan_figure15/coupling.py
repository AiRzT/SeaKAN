from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Geometry
from .model import ContactEnrichedWaterKAN, WaterPressureKAN
from .physics import squared_magnitude
from .rans import HarmonicRANSKAN


@dataclass(frozen=True)
class ContactDiagnostics:
    complex_l2: float
    left_complex_jump: float
    right_complex_jump: float
    left_amplitude_jump: float
    right_amplitude_jump: float

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "complex_l2",
                "left_complex_jump",
                "right_complex_jump",
                "left_amplitude_jump",
                "right_amplitude_jump",
            )
        }


def contact_coordinates(
    geometry: Geometry,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    radius = geometry.radius_over_height
    x = torch.as_tensor(
        (-radius, radius), dtype=dtype, device=device
    )
    y = torch.full_like(x, geometry.pipe_center_y)
    return torch.stack((x, y), dim=1)


def complex_contact_residual(
    rans_model: HarmonicRANSKAN,
    interface_model: WaterPressureKAN | ContactEnrichedWaterKAN,
    geometry: Geometry,
) -> torch.Tensor:
    parameter = next(interface_model.parameters())
    xy = contact_coordinates(
        geometry,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    return rans_model(xy).q - interface_model(xy).q


def complex_contact_loss(
    rans_model: HarmonicRANSKAN,
    interface_model: WaterPressureKAN | ContactEnrichedWaterKAN,
    geometry: Geometry,
) -> torch.Tensor:
    return squared_magnitude(
        complex_contact_residual(
            rans_model, interface_model, geometry
        )
    ).mean()


def contact_diagnostics(
    rans_model: HarmonicRANSKAN,
    interface_model: WaterPressureKAN | ContactEnrichedWaterKAN,
    geometry: Geometry,
) -> ContactDiagnostics:
    with torch.no_grad():
        parameter = next(interface_model.parameters())
        xy = contact_coordinates(
            geometry,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        rans_pressure = rans_model(xy).q
        interface_pressure = interface_model(xy).q
        residual = rans_pressure - interface_pressure
        complex_jump = residual.abs()
        amplitude_jump = (
            rans_pressure.abs() - interface_pressure.abs()
        ).abs()
        return ContactDiagnostics(
            complex_l2=float(
                torch.sqrt(
                    squared_magnitude(residual).mean()
                ).cpu()
            ),
            left_complex_jump=float(complex_jump[0].cpu()),
            right_complex_jump=float(complex_jump[1].cpu()),
            left_amplitude_jump=float(amplitude_jump[0].cpu()),
            right_amplitude_jump=float(amplitude_jump[1].cpu()),
        )
