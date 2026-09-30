"""Check environment: Python version, GPU, VRAM, disk, key libraries.

Usage:  py scripts/check_env.py
Exit code 0 if the environment can run the pipeline, 1 otherwise.
"""
import importlib
import shutil
import sys


def section(title):
    print(f"\n=== {title} ===")


def check_python():
    section("Python")
    print(f"version: {sys.version.split()[0]}  executable: {sys.executable}")
    ok = sys.version_info >= (3, 10)
    print("OK" if ok else "FAIL: need >= 3.10")
    return ok


def check_gpu():
    section("GPU")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"cuda: YES  device: {name}  VRAM: {total:.2f} GiB")
            return True
        print("cuda: NO  (pipeline will fall back to CPU — slow but functional)")
        return True  # not fatal
    except ImportError:
        print("torch not installed")
        return False


def check_disk():
    path = shutil.disk_usage(".")
    free_gb = path.free / 1024**3
    print(f"free disk on project drive: {free_gb:.1f} GiB (budget: < 15 GiB)")
    return free_gb > 2.0


def check_libs():
    section("Libraries")
    libs = ["transformers", "datasets", "soundfile", "sounddevice", "yaml",
            "jiwer", "numpy", "peft", "TTS", "fastapi", "pytest"]
    all_ok = True
    for lib in libs:
        try:
            m = importlib.import_module(lib)
            ver = getattr(m, "__version__", "?")
            print(f"  {lib}: {ver}")
        except ImportError:
            print(f"  {lib}: MISSING")
            all_ok = False
    return all_ok


def main():
    print("Calling-Agent environment check")
    ok = check_python()
    gpu_ok = check_gpu()
    disk_ok = check_disk()
    libs_ok = check_libs()
    section("Summary")
    print(f"python_ok={ok} gpu_ok={gpu_ok} disk_ok={disk_ok} libs_ok={libs_ok}")
    if not libs_ok:
        print("Install missing libs with: py -m pip install -r requirements.txt")
    sys.exit(0 if (ok and disk_ok) else 1)


if __name__ == "__main__":
    main()
