"""Compare newly generated COMSOL grids and complex pressures to released reference fields."""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_outputs", type=Path)
    args = parser.parse_args()
    reference = Path(__file__).with_name("outputs")
    files = sorted(reference.glob("*_domain.csv"))
    if len(files) != 73:
        raise SystemExit(f"Expected 73 archived fields; found {len(files)}")
    scores = []
    for original in files:
        old = np.loadtxt(original, comments="%", delimiter=",")
        new = np.loadtxt(args.new_outputs / original.name, comments="%", delimiter=",")
        # COMSOL can export the same mesh nodes in a different row order.
        old = old[np.lexsort((old[:, 3], old[:, 2]))]
        new = new[np.lexsort((new[:, 3], new[:, 2]))]
        np.testing.assert_allclose(new[:, 2:4], old[:, 2:4], atol=1e-10, rtol=0, err_msg=original.name)
        p_ref, p_new = old[:, 4] + 1j * old[:, 5], new[:, 4] + 1j * new[:, 5]
        relative_l2 = np.linalg.norm(p_new - p_ref) / np.linalg.norm(p_ref)
        scores.append((float(relative_l2), original.name))
    maximum, name = max(scores)
    print(f"Compared {len(scores)} meshes and complex pressure fields. "
          f"Worst relative L2 difference = {maximum:.9g} ({name}).")
    if not np.all(np.isfinite([r[0] for r in scores])) or maximum > 1e-5:
        raise SystemExit("Field difference exceeds 1e-5 or contains a nonfinite value.")


if __name__ == "__main__":
    main()
