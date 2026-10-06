"""Repository-root entry point for the isolated offline CPU diagnostic."""

from pathlib import Path
import sys

# -I omits the script directory. Add only the repository root, so the package's
# types.py cannot shadow the standard library's top-level types module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_growth.l0_qwen_cpu_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
