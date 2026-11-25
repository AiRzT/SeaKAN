"""Re-evaluate benchmark weights for manuscript Figures 3, 4 and 6 on CPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "legacy"


def verify_checkpoint(number, output, device):
    if number == 7:
        project = LEGACY / "src"
        source = project
        archive = project / "outputs" / "figure7"
        entry = project / "run_fig7.py"
        checkpoint_args = ["--checkpoint-dir", str(archive)]
    else:
        project = LEGACY / f"figure{number}_mpm_kan"
        source = project / "src"
        archive = project / "artifacts"
        entry = project / f"run_figure{number}.py"
        checkpoint_args = []
    destination = output / f"benchmark_{number}"
    destination.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(source), "MPLBACKEND": "Agg"}
    with (destination / "evaluation.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(entry), "--evaluate-only", "--device", device,
                        "--output-dir", str(destination), *checkpoint_args], check=True,
                       cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
    filename = f"figure{number}_profiles.csv"
    with (archive / filename).open(encoding="utf-8-sig", newline="") as original:
        reference = list(csv.DictReader(original))
    with (destination / filename).open(encoding="utf-8-sig", newline="") as generated:
        actual = list(csv.DictReader(generated))
    def predictions(rows):
        return [row for row in rows if row["source"].startswith("mpm_kan")]
    reference, actual = predictions(reference), predictions(actual)
    if not reference or len(reference) != len(actual):
        raise ValueError(f"Benchmark {number}: prediction row count differs")
    if set(reference[0]) != set(actual[0]):
        raise ValueError(f"Benchmark {number}: header mismatch")
    maximum = 0.0
    for column in reference[0]:
        expected_values = [row[column] for row in reference]
        result_values = [row[column] for row in actual]
        try:
            expected = np.asarray(expected_values, dtype=float)
        except ValueError:
            if expected_values != result_values:
                raise ValueError(f"Benchmark {number}: labels differ in {column}")
            continue
        result = np.asarray(result_values, dtype=float)
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-5,
                                   err_msg=f"Benchmark {number}, {column}")
        maximum = max(maximum, float(np.max(np.abs(result - expected))))
    return {"benchmark": number, "prediction_rows": len(reference), "max_absolute_difference": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=int, choices=(7, 8, 13, 15), nargs="+", default=[7, 8, 13, 15],
                        help="Source-study IDs: 7 → Fig. 3; 8 and 13 → Fig. 4; 15 → Fig. 6")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmarks"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    results = [verify_checkpoint(n, output, args.device) for n in args.benchmarks]
    sys.path.insert(0, str(LEGACY / "metric_evaluation"))
    from compute_all_metrics import compute_all
    scores = compute_all()
    if scores["reconciliation"]["status"] != "passed":
        raise ValueError(f"Archived metrics differ: {scores['reconciliation']}")
    medians = {}
    for reference, printed in (("analytical", 1.22), ("numerical", 2.42)):
        errors = [r["l2_percent"] for r in scores["comparisons"]
                  if r["figure"] == "figure7" and r["reference_source"] == f"digitized_{reference}"]
        if len(errors) != 16:
            raise ValueError(f"Expected 16 soil-variable pairs for {reference}")
        medians[reference] = float(np.median(errors))
        np.testing.assert_allclose(medians[reference], printed, atol=0.005, rtol=0.)
    payload = {"checkpoint_evaluations": results, "archived_profile_comparisons": len(scores["comparisons"]),
               "archive_reconciliation": scores["reconciliation"], "flat_seabed_median_relative_l2_percent": medians}
    report = json.dumps(payload, indent=2)
    (output / "verification.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
