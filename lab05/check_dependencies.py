"""Check the Python environment and dependencies used by download.py."""

from __future__ import annotations

import importlib.util
import os
import sys


DEPENDENCIES = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "imageio": "imageio",
    "scipy": "scipy",
}


def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    print("Python executable:", sys.executable)
    print("Python version:   ", sys.version.split()[0])
    print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV", "not active"))
    print("CONDA_PREFIX:     ", os.environ.get("CONDA_PREFIX", "not active"))
    print("\nDependency check:")

    status = {}
    for display_name, module_name in DEPENDENCIES.items():
        status[display_name] = is_installed(module_name)
        marker = "OK" if status[display_name] else "MISSING"
        print(f"  {display_name:<12} {marker}")

    # download.py requires NumPy and Matplotlib. It first uses ImageIO for
    # saving images and only attempts SciPy as a legacy fallback.
    usable = status["numpy"] and status["matplotlib"] and (
        status["imageio"] or status["scipy"]
    )

    print()
    if usable:
        print("Result: download.py has the import dependencies it needs.")
        return 0

    missing_required = [
        name for name in ("numpy", "matplotlib") if not status[name]
    ]
    if missing_required:
        print("Required packages missing:", ", ".join(missing_required))
    if not status["imageio"] and not status["scipy"]:
        print("Image saving dependency missing: install imageio.")
    print("Result: install the missing packages in the active Conda environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
