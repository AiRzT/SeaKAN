"""Compile and run the COMSOL 6.4 sealed-bottom suite in a separate directory."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, help="Directory containing comsolcompile and comsolbatch.")
    parser.add_argument("--cores", type=int, help="Optional COMSOL CPU core count.")
    args = parser.parse_args()

    def command(action: str) -> list[str]:
        directory = str(args.bin_dir.resolve()) if args.bin_dir else None
        separate = shutil.which("comsol" + action, path=directory)
        if separate:
            return [separate]
        combined = shutil.which("comsol", path=directory)
        if combined:
            return [combined, action]
        raise SystemExit(f"Cannot find COMSOL {action}; specify --bin-dir for COMSOL 6.4.")

    compiler, batch = command("compile"), command("batch")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    outputs = work / "outputs"
    outputs.mkdir(exist_ok=False)
    original = Path(__file__).with_name("CleanSealedBiotSuite.java")
    source = work / original.name
    output_value = outputs.as_posix().replace('"', r'\"') + "/"
    source.write_text(original.read_text(encoding="utf-8").replace('"./outputs/"', f'"{output_value}"'), encoding="utf-8")
    subprocess.run([*compiler, str(source)], cwd=work, check=True)
    args_batch = [*batch, "-inputfile", str(source.with_suffix(".class")), "-outputfile",
                  str(work / "final_validation.mph"), "-batchlog", str(work / "batch.log")]
    if args.cores is not None:
        args_batch.extend(["-np", str(args.cores)])
    stdout_file = work / "console.log"
    with stdout_file.open("wb") as log:
        subprocess.run(args_batch, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    timings = {}
    for path in (stdout_file, work / "batch.log"):
        if path.exists():
            for line in path.read_bytes().decode("utf-8", errors="replace").splitlines():
                if "TIMING_ROW," in line:
                    tag, tier, het, elems, verts, mesh, solve, total = line.split("TIMING_ROW,", 1)[1].strip().split(",")
                    # Each of the three P2 unknowns has one DOF per vertex and per edge.
                    # In a triangular mesh with one interior hole, E = V + F.
                    dofs = 3 * (2 * int(verts) + int(elems))
                    timings[tag] = (tag, tier, het, elems, verts, mesh, solve, total, dofs)
    if len(timings) != 74 or len(list(outputs.glob("*_complex_domain.csv"))) != 52:
        raise RuntimeError(f"Incomplete run: {len(timings)} timing rows; see {stdout_file} and batch.log.")
    with (work / "comsol_timings_and_mesh.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tag", "tier", "het", "elems", "verts", "mesh_ms", "solve_ms", "total_ms", "dofs"))
        writer.writerows(timings.values())
    print(f"Generated {len(timings)} runs and 52 fine reference fields in {work}")


if __name__ == "__main__":
    main()
