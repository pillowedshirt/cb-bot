import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CPP_SOURCE = BASE_DIR / "cpp" / "fast_calibration_core.cpp"
SETUP_FILE = BASE_DIR / "setup_fast_calibration.py"
STATUS_FILE = BASE_DIR / "CSVs" / "07_runtime_state" / "fast_calibration_build_status.json"


def _write_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status["ts"] = time.time()
    with STATUS_FILE.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2, sort_keys=True)


def _compiled_module_exists() -> bool:
    patterns = [
        "fast_calibration_core*.pyd",
        "fast_calibration_core*.so",
        "fast_calibration_core*.dll",
    ]
    for pattern in patterns:
        if list(BASE_DIR.glob(pattern)):
            return True
    return False


def main() -> int:
    started = time.time()

    if _compiled_module_exists():
        _write_status({
            "ok": True,
            "action": "already_built",
            "reason": "compiled fast_calibration_core module already exists",
            "duration_sec": time.time() - started,
        })
        return 0

    if not CPP_SOURCE.exists():
        _write_status({"ok": False, "action": "missing_cpp_source", "reason": f"missing {CPP_SOURCE}", "duration_sec": time.time() - started})
        return 2

    if not SETUP_FILE.exists():
        _write_status({"ok": False, "action": "missing_setup_file", "reason": f"missing {SETUP_FILE}", "duration_sec": time.time() - started})
        return 3

    try:
        cmd = [sys.executable, str(SETUP_FILE), "build_ext", "--inplace"]
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=180)
        built = _compiled_module_exists()
        _write_status({
            "ok": bool(proc.returncode == 0 and built),
            "action": "build_ext_inplace",
            "cmd": cmd,
            "returncode": proc.returncode,
            "compiled_module_exists": built,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "duration_sec": time.time() - started,
            "repair_hint": (
                "Install Microsoft C++ Build Tools with the Desktop development with C++ workload, then rerun run.bat."
                if not built
                else ""
            ),
        })
        return 0 if proc.returncode == 0 and built else 1
    except Exception as exc:
        _write_status({
            "ok": False,
            "action": "build_exception",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "duration_sec": time.time() - started,
            "repair_hint": "Install Microsoft C++ Build Tools and confirm pybind11 is installed in the active venv.",
        })
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
