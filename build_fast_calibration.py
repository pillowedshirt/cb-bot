import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CALIBRATION_SETUP_FILE = BASE_DIR / "setup_fast_calibration.py"
INSTITUTIONAL_SETUP_FILE = BASE_DIR / "setup_fast_institutional.py"

CALIBRATION_CPP_SOURCE = BASE_DIR / "cpp" / "fast_calibration_core.cpp"
INSTITUTIONAL_CPP_SOURCE = BASE_DIR / "fast_institutional_core.cpp"

STATUS_FILE = BASE_DIR / "CSVs" / "07_runtime_state" / "fast_calibration_build_status.json"
FULL_LOG_FILE = BASE_DIR / "CSVs" / "07_runtime_state" / "fast_calibration_build_full.log"


def _write_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status["ts"] = time.time()
    with STATUS_FILE.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2, sort_keys=True)


def _module_file_exists(module_prefix: str) -> bool:
    patterns = [
        f"{module_prefix}*.pyd",
        f"{module_prefix}*.so",
        f"{module_prefix}*.dll",
    ]
    for pattern in patterns:
        if list(BASE_DIR.glob(pattern)):
            return True
    return False


def _module_import_ok(module_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except Exception:
        return False


def _pybind11_available() -> bool:
    try:
        import pybind11  # noqa: F401
        return True
    except Exception:
        return False


def _run_build(setup_file: Path, module_name: str) -> dict:
    started = time.time()

    if not setup_file.exists():
        return {
            "module": module_name,
            "ok": False,
            "action": "missing_setup_file",
            "reason": f"missing {setup_file}",
            "duration_sec": time.time() - started,
        }

    cmd = [sys.executable, str(setup_file), "build_ext", "--inplace"]

    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=240,
    )

    exists = _module_file_exists(module_name)
    import_ok = _module_import_ok(module_name)

    return {
        "module": module_name,
        "ok": bool(proc.returncode == 0 and exists),
        "action": "build_ext_inplace",
        "cmd": cmd,
        "returncode": proc.returncode,
        "compiled_module_exists": bool(exists),
        "import_spec_found": bool(import_ok),
        "stdout_tail": (proc.stdout or "")[-8000:],
        "stderr_tail": (proc.stderr or "")[-8000:],
        "duration_sec": time.time() - started,
    }


def main() -> int:
    started = time.time()

    status = {
        "ok": False,
        "action": "split_fast_cpp_build",
        "pybind11_available": _pybind11_available(),
        "calibration": {},
        "institutional": {},
        "duration_sec": 0.0,
        "repair_hint": "",
    }

    if not status["pybind11_available"]:
        status["repair_hint"] = (
            "pybind11 is not importable in the active Python environment. "
            "Run: python -m pip install pybind11 setuptools wheel"
        )
        status["duration_sec"] = time.time() - started
        _write_status(status)
        return 1

    if not CALIBRATION_CPP_SOURCE.exists():
        status["repair_hint"] = f"Missing calibration C++ source: {CALIBRATION_CPP_SOURCE}"
        status["duration_sec"] = time.time() - started
        _write_status(status)
        return 2

    if not INSTITUTIONAL_CPP_SOURCE.exists():
        status["institutional"] = {
            "module": "fast_institutional_core",
            "ok": False,
            "action": "missing_cpp_source",
            "reason": f"missing {INSTITUTIONAL_CPP_SOURCE}",
        }

    try:
        calibration_already_ok = _module_file_exists("fast_calibration_core") and _module_import_ok("fast_calibration_core")
        institutional_already_ok = _module_file_exists("fast_institutional_core") and _module_import_ok("fast_institutional_core")

        if calibration_already_ok:
            status["calibration"] = {"module": "fast_calibration_core", "ok": True, "action": "already_built", "compiled_module_exists": True, "import_spec_found": True}
        else:
            status["calibration"] = _run_build(CALIBRATION_SETUP_FILE, "fast_calibration_core")

        if INSTITUTIONAL_CPP_SOURCE.exists():
            if institutional_already_ok:
                status["institutional"] = {"module": "fast_institutional_core", "ok": True, "action": "already_built", "compiled_module_exists": True, "import_spec_found": True}
            else:
                status["institutional"] = _run_build(INSTITUTIONAL_SETUP_FILE, "fast_institutional_core")

        calibration_ok = bool(status.get("calibration", {}).get("ok"))
        institutional_ok = bool(status.get("institutional", {}).get("ok"))

        status["ok"] = bool(calibration_ok)
        status["fast_calibration_core_available"] = bool(calibration_ok)
        status["fast_institutional_core_available"] = bool(institutional_ok)

        if not calibration_ok:
            status["repair_hint"] = ("fast_calibration_core failed. Install Microsoft C++ Build Tools with Desktop development with C++, confirm pybind11 is installed, then rerun run.bat.")
        elif not institutional_ok:
            status["repair_hint"] = ("fast_calibration_core is working. fast_institutional_core failed or is unavailable, so institutional scoring will use Python fallback.")
        else:
            status["repair_hint"] = ""

        status["duration_sec"] = time.time() - started

        FULL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with FULL_LOG_FILE.open("w", encoding="utf-8", errors="replace") as log:
            log.write(json.dumps(status, indent=2, sort_keys=True))

        _write_status(status)

        return 0 if calibration_ok else 1

    except Exception as exc:
        status["ok"] = False
        status["action"] = "build_exception"
        status["error"] = str(exc)
        status["traceback"] = traceback.format_exc()
        status["duration_sec"] = time.time() - started
        status["repair_hint"] = "Install Microsoft C++ Build Tools and confirm pybind11 is installed in the active venv."
        _write_status(status)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
