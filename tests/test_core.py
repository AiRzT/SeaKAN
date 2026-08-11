from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from mpm_kan_generalization.cases import CaseTable
from mpm_kan_generalization.config import (
    ParameterDomain,
    PhysicalConstants,
)
from mpm_kan_generalization.design import create_design
from mpm_kan_generalization.field_inverse import (
    comsol_field_observations,
    default_field_sensors,
)
from mpm_kan_generalization.inverse import (
    bounded_inverse_parameters,
    initial_latent_from_fraction,
    select_sensor_rows,
)
from mpm_kan_generalization.model import ParametricSoilKAN
from mpm_kan_generalization.metrics import regression_metrics
from mpm_kan_generalization.physics import GeneralizationLoss
from mpm_kan_generalization.random_field import (
    lognormal_permeability_ratio,
)
from mpm_kan_generalization.scoring import score_pipe_profiles
from mpm_kan_generalization.training import _combine_case_losses


def test_design_shapes_and_fixed_boundary_blind_cases() -> None:
    domain = ParameterDomain()
    design = create_design(
        domain,
        train_cases=8,
        validation_cases=4,
        blind_cases=4,
    )
    assert design.train.shape == (8, 8)
    assert design.validation.shape == (4, 8)
    assert design.blind.shape == (4, 8)
    assert design.blind[0, 0] == domain.wave_height_m[0]
    assert design.blind[1, 0] == domain.wave_height_m[1]
    assert not np.array_equal(
        design.train[:4], design.blind
    )

    anchored = create_design(
        domain,
        train_cases=8,
        validation_cases=4,
        blind_cases=4,
        training_boundary_anchors=True,
    )
    assert anchored.train[0, 0] == domain.wave_height_m[0]
    assert anchored.train[1, 0] == domain.wave_height_m[1]
    assert not np.array_equal(
        anchored.train[:, 4:], anchored.blind[:8, 4:]
    )


def test_random_permeability_is_positive_and_varies() -> None:
    domain = ParameterDomain()
    constants = PhysicalConstants()
    parameters = torch.tensor(
        [
            [0.12, 1.5, 3.5e-5, 1.0, 1.0, -0.5, 0.3, 0.8],
            [0.12, 1.5, 3.5e-5, 1.0, -1.0, 0.5, -0.3, -0.8],
        ],
        dtype=torch.float64,
    )
    xy = torch.tensor(
        [[-0.4, 0.25], [0.4, 0.75]],
        dtype=torch.float64,
    )
    ratio = lognormal_permeability_ratio(
        xy, parameters, domain, constants
    )
    assert torch.all(ratio > 0.0)
    assert not torch.allclose(ratio[0], ratio[1])


def test_hard_interface_pressure_and_pipe_displacement() -> None:
    domain = ParameterDomain()
    constants = PhysicalConstants()
    parameters = np.asarray(
        [[0.12, 1.5, 3.5e-5, 0.75, 0.2, -0.1, 0.3, -0.4]]
    )
    table = CaseTable.from_numpy(
        parameters,
        constants,
        torch.device("cpu"),
        torch.float64,
    )
    model = ParametricSoilKAN(
        domain,
        constants,
        width=6,
        hidden_layers=1,
        n_basis=7,
        local_width=5,
        dtype=torch.float64,
    )
    x = torch.linspace(
        -constants.half_width,
        constants.half_width,
        17,
        dtype=torch.float64,
    )
    top = torch.stack((x, torch.ones_like(x)), dim=1)
    fields = model(
        top,
        table.parameters.expand(x.numel(), -1),
        table.kappa.expand(x.numel()),
        table.pipe_center_y.expand(x.numel()),
    )
    expected = torch.exp(
        torch.complex(
            torch.zeros_like(x), table.kappa[0] * x
        )
    )
    assert torch.allclose(fields.p, expected, atol=1.0e-12)

    theta = torch.linspace(
        0.0, 2.0 * torch.pi, 25, dtype=torch.float64
    )
    pipe = torch.stack(
        (
            constants.radius * torch.cos(theta),
            table.pipe_center_y[0]
            + constants.radius * torch.sin(theta),
        ),
        dim=1,
    )
    fields = model(
        pipe,
        table.parameters.expand(theta.numel(), -1),
        table.kappa.expand(theta.numel()),
        table.pipe_center_y.expand(theta.numel()),
    )
    assert fields.u.abs().max() < 1.0e-12
    assert fields.w.abs().max() < 1.0e-12

    pipe = pipe.detach().clone().requires_grad_(True)
    pressure = model(
        pipe,
        table.parameters.expand(theta.numel(), -1),
        table.kappa.expand(theta.numel()),
        table.pipe_center_y.expand(theta.numel()),
    ).p
    gradient_real = torch.autograd.grad(
        pressure.real.sum(),
        pipe,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_imag = torch.autograd.grad(
        pressure.imag.sum(),
        pipe,
        create_graph=True,
    )[0]
    gradient = torch.complex(gradient_real, gradient_imag)
    normal = torch.stack(
        (torch.cos(theta), torch.sin(theta)), dim=1
    )
    normal_flux = (gradient * normal).sum(dim=1)
    assert normal_flux.abs().max() < 1.0e-10


def test_group_dro_combination_emphasises_worst_case() -> None:
    def loss(total: float) -> GeneralizationLoss:
        value = torch.tensor(total)
        fields = {
            name: value.clone()
            for name in GeneralizationLoss.__dataclass_fields__
        }
        return GeneralizationLoss(**fields)

    average = _combine_case_losses(
        [loss(1.0), loss(9.0)],
        robust_alpha=0.0,
        temperature=0.75,
    )
    robust = _combine_case_losses(
        [loss(1.0), loss(9.0)],
        robust_alpha=0.7,
        temperature=0.75,
    )
    assert torch.isclose(average.total, torch.tensor(5.0))
    assert robust.total > average.total
    assert robust.total < torch.tensor(9.0)


def test_mlp_ablation_matches_kan_parameter_budget() -> None:
    domain = ParameterDomain()
    constants = PhysicalConstants()
    models = {
        backbone: ParametricSoilKAN(
            domain,
            constants,
            width=32,
            hidden_layers=3,
            n_basis=8,
            local_width=24,
            backbone=backbone,
        )
        for backbone in ("kan", "mlp")
    }
    counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    assert abs(counts["mlp"] / counts["kan"] - 1.0) < 0.02


def test_random_field_conditioning_ablation_masks_xi_only() -> None:
    domain = ParameterDomain()
    constants = PhysicalConstants()
    xy = torch.tensor([[0.1, 0.5], [0.1, 0.5]])
    parameters = torch.tensor(
        [
            [0.12, 1.5, 3.5e-5, 1.0, 1.0, -0.5, 0.3, 0.8],
            [0.12, 1.5, 3.5e-5, 1.0, -1.0, 0.5, -0.3, -0.8],
        ]
    )
    conditioned = ParametricSoilKAN(
        domain, constants, condition_random_field=True
    )
    masked = ParametricSoilKAN(
        domain, constants, condition_random_field=False
    )
    conditioned_features = conditioned.features(xy, parameters)
    masked_features = masked.features(xy, parameters)
    assert not torch.allclose(
        conditioned_features[0], conditioned_features[1]
    )
    assert torch.allclose(masked_features[0], masked_features[1])
    assert torch.allclose(
        conditioned_features[:, :-domain.random_modes],
        masked_features[:, :-domain.random_modes],
    )


def test_regression_metrics_known_values() -> None:
    truth = np.asarray([1.0, 2.0, 3.0])
    exact = regression_metrics(truth, truth)
    assert exact.r2 == 1.0
    assert exact.relative_l2 == 0.0
    assert exact.mse == 0.0
    assert exact.rmse == 0.0

    estimate = np.asarray([1.0, 2.0, 4.0])
    scored = regression_metrics(truth, estimate)
    assert np.isclose(scored.mse, 1.0 / 3.0)
    assert np.isclose(scored.rmse, np.sqrt(1.0 / 3.0))
    assert np.isclose(scored.relative_l2, np.sqrt(1.0 / 14.0))
    assert np.isclose(scored.r2, 0.5)


def test_inverse_parameter_transform_respects_bounds() -> None:
    domain = ParameterDomain()
    latent = initial_latent_from_fraction(
        np.asarray([0.25, 0.75]),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    permeability, burial = bounded_inverse_parameters(latent, domain)
    log_bounds = np.log10(domain.permeability_m_per_s)
    expected_log_permeability = (
        log_bounds[0] + 0.25 * (log_bounds[1] - log_bounds[0])
    )
    expected_burial = (
        domain.burial_center_depth_over_diameter[0]
        + 0.75
        * (
            domain.burial_center_depth_over_diameter[1]
            - domain.burial_center_depth_over_diameter[0]
        )
    )
    assert np.isclose(
        float(permeability.detach()), 10.0**expected_log_permeability
    )
    assert np.isclose(float(burial.detach()), expected_burial)


def test_inverse_sensor_selection_is_uniform_and_unique() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 17)
    pressure = np.cos(theta) + 2.0
    selected_theta, selected_pressure = select_sensor_rows(
        theta, pressure, 8
    )
    assert selected_theta.size == 8
    assert selected_pressure.size == 8
    assert np.unique(np.round(selected_theta, 10)).size == 8
    assert selected_theta.min() >= 0.0
    assert selected_theta.max() < 2.0 * np.pi


def test_field_inverse_sensors_remain_outside_candidate_pipe() -> None:
    constants = PhysicalConstants()
    sensors = default_field_sensors(
        device=torch.device("cpu"), dtype=torch.float64
    )
    domain = ParameterDomain()
    for burial in domain.burial_center_depth_over_diameter:
        centre = (
            1.0
            - burial
            * constants.pipe_diameter_m
            / constants.seabed_thickness_m
        )
        distance = torch.sqrt(
            sensors[:, 0].square()
            + (sensors[:, 1] - centre).square()
        )
        assert torch.all(distance > constants.radius)


def test_comsol_complex_field_interpolation() -> None:
    constants = PhysicalConstants()
    side = constants.seabed_thickness_m
    nodes = np.asarray(
        [[0.0, 0.0], [side, 0.0], [0.0, side], [side, side]]
    )
    real = 1.0 + 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1]
    imaginary = -0.5 + nodes[:, 0] - nodes[:, 1]
    values = np.column_stack(
        (nodes, nodes, real, imaginary, np.hypot(real, imaginary))
    )
    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "comsol.csv"
        np.savetxt(reference, values, delimiter=",")
        xy = torch.tensor(
            [[0.25, 0.40], [0.75, 0.60]], dtype=torch.float64
        )
        observed, quality = comsol_field_observations(
            reference, xy, constants
        )
    physical = xy.numpy() * side
    expected = (
        1.0 + 2.0 * physical[:, 0] + 3.0 * physical[:, 1]
    ) + 1j * (-0.5 + physical[:, 0] - physical[:, 1])
    assert np.allclose(observed.numpy(), expected)
    assert quality["comsol_solution_nodes"] == 4


def test_pipe_profile_scorer_exact_reference(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "prediction.csv"
    reference = tmp_path / "reference.csv"
    with prediction.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "split",
                "case_index",
                "theta_rad",
                "normalized_pressure",
                "absolute_pressure_pa",
            )
        )
        writer.writerows(
            (
                ("blind", 0, 0.0, 1.0, 1.0),
                ("blind", 0, np.pi, 2.0, 2.0),
                ("blind", 0, 2.0 * np.pi, 1.0, 1.0),
            )
        )
    with reference.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case_id",
                "theta_rad",
                "reference_normalized_pressure",
            )
        )
        writer.writerows(
            (
                ("blind_000", 0.0, 1.0),
                ("blind_000", np.pi, 2.0),
            )
        )
    result = score_pipe_profiles(prediction, reference)
    assert result["overall_micro"]["r2"] == 1.0
    assert result["overall_micro"]["relative_l2"] == 0.0
