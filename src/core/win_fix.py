import os
import sys

# Windows PyTorch DLL & OpenMP duplicate fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Reconfigure stdout/stderr to UTF-8 for Windows console emoji/Vietnamese text support
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    conda_bin = os.path.join(sys.prefix, "Library", "bin")

    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

    if os.path.exists(conda_bin):
        try:
            os.add_dll_directory(conda_bin)
        except Exception:
            pass
        os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

# Pre-load torch DLLs safely into process memory
try:
    import torch
except Exception as e:
    pass
