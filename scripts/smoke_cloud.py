"""
scripts/smoke_cloud.py — Driver status table for all DRIVER_SPECS entries.

Prints whether each engine's packages are installed and their version
(via importlib.metadata where available).  Makes zero network calls —
this is the "what's activated on this machine" CLI panel.

Usage:
    python scripts/smoke_cloud.py
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.driver_manager import DRIVER_SPECS, is_driver_installed  # noqa: E402


def _get_version(package_name: str) -> str:
    """Return installed version of *package_name* or '?' if not found."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def main() -> None:
    print("\nHealthPipe AI v2 — Driver Status")
    print("=" * 62)
    print(f"  {'Engine':<18} {'Label':<28} {'Installed':<10} {'Version'}")
    print("-" * 62)

    for engine_id, spec in DRIVER_SPECS.items():
        label = spec.get("label", engine_id)
        installed = is_driver_installed(engine_id)
        status = "✓ yes" if installed else "✗ no"

        # Get version of the first pip package as representative
        first_pip = spec.get("pip", [])
        if first_pip and installed:
            # pip package name = everything before '=='
            pkg_name = first_pip[0].split("==")[0]
            version = _get_version(pkg_name)
        elif not first_pip and installed:
            version = "(stdlib)"
        else:
            version = "-"

        print(f"  {engine_id:<18} {label:<28} {status:<10} {version}")

    print("-" * 62)
    installed_count = sum(1 for eid in DRIVER_SPECS if is_driver_installed(eid))
    total = len(DRIVER_SPECS)
    print(f"\n  {installed_count}/{total} drivers installed.")
    print(
        "\n  To install a missing driver:\n"
        "    from core.driver_manager import install_driver\n"
        "    install_driver('<engine_id>')\n"
    )


if __name__ == "__main__":
    main()
