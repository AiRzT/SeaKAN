"""Entry point for the archived Table 6 four-model baseline training implementation."""
from pathlib import Path
import sys

LEGACY = Path(__file__).resolve().parent / "legacy"
sys.path.insert(0, str(LEGACY / "figure9_mpm_kan" / "src"))
sys.path.insert(0, str(LEGACY / "figure9_model_comparison" / "src"))
from figure9_comparison.run import main

if __name__ == "__main__":
    main()
