import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Set

@dataclass
class HardwareProfile:
    device: str               # "cuda" or "cpu"
    compute_type: str         # "float16", "int8_float16", "int8", "float32"
    gpu_name: Optional[str]   # e.g. "NVIDIA GeForce RTX 3090"
    vram_mb: Optional[int]    # e.g. 24576
    is_gpu: bool
    cuda_count: int
    details: str

def configure_system_cuda_paths() -> List[str]:
    """
    Discovers and registers CUDA and cuDNN DLL directories on Windows across:
    1. System environment variables (CUDA_PATH, CUDA_PATH_V12_X, etc.)
    2. Standard installation paths (C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*)
    3. Python virtualenv packages (site-packages/nvidia/*)
    """
    registered_paths: List[str] = []
    if sys.platform != "win32":
        return registered_paths

    candidate_dirs: List[Path] = []

    # 1. Check environment variables
    for env_var in ["CUDA_PATH", "CUDA_PATH_V12_0", "CUDA_PATH_V12_1", "CUDA_PATH_V12_2", "CUDA_PATH_V12_3", "CUDA_PATH_V12_4", "CUDA_PATH_V12_5", "CUDA_PATH_V11_8"]:
        val = os.environ.get(env_var)
        if val:
            p = Path(val) / "bin"
            if p.is_dir():
                candidate_dirs.append(p)

    # 2. Check standard Program Files installation root
    prog_files_toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if prog_files_toolkit.is_dir():
        for ver_folder in prog_files_toolkit.iterdir():
            bin_dir = ver_folder / "bin"
            if bin_dir.is_dir():
                candidate_dirs.append(bin_dir)

    # 3. Check Python venv site-packages nvidia wheels
    site_packages_nvidia = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia"
    if site_packages_nvidia.is_dir():
        for subfolder in ["cublas", "cudnn", "cuda_nvrtc"]:
            bin_dir = site_packages_nvidia / subfolder / "bin"
            if bin_dir.is_dir():
                candidate_dirs.append(bin_dir)

    # Register uniquely
    seen: Set[str] = set()
    for c_dir in candidate_dirs:
        str_dir = str(c_dir.resolve())
        if str_dir.lower() not in seen:
            seen.add(str_dir.lower())
            os.environ["PATH"] = str_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str_dir)
                    registered_paths.append(str_dir)
                except Exception:
                    pass

    return registered_paths

# Run DLL configuration on import
configure_system_cuda_paths()

def detect_hardware(requested_device: str = "auto", requested_compute: str = "auto") -> HardwareProfile:
    """
    Adaptively probes the current environment and selects the best available device
    and precision without crashing on non-GPU or non-CUDA systems.
    """
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_count = 0

    gpu_name = None
    vram_mb = None

    # Probe GPU metadata if available
    if cuda_count > 0:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    vram_mb = int(float(parts[1]))
        except Exception:
            gpu_name = "NVIDIA CUDA GPU"

    # Evaluate device choice
    if requested_device == "cuda":
        if cuda_count < 1:
            raise RuntimeError(
                "Explicit '--device cuda' requested, but no CUDA-capable GPU is available on this system. "
                "Ensure your NVIDIA drivers and CUDA libraries are installed, or run with '--device auto' or '--device cpu'."
            )
        target_device = "cuda"
    elif requested_device == "cpu":
        target_device = "cpu"
    else:
        # Auto selection
        target_device = "cuda" if cuda_count > 0 else "cpu"

    # Evaluate compute precision
    if target_device == "cuda":
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        if requested_compute != "auto":
            target_compute = requested_compute
        else:
            if "float16" in supported:
                target_compute = "float16"
            elif "int8_float16" in supported:
                target_compute = "int8_float16"
            else:
                target_compute = "float32"

        details = f"GPU: {gpu_name or 'NVIDIA'} ({vram_mb or 'N/A'} MB VRAM) | Engine: CUDA {target_compute}"
        return HardwareProfile(
            device="cuda",
            compute_type=target_compute,
            gpu_name=gpu_name,
            vram_mb=vram_mb,
            is_gpu=True,
            cuda_count=cuda_count,
            details=details
        )
    else:
        # CPU Profile
        if requested_compute != "auto":
            target_compute = requested_compute
        else:
            target_compute = "int8"  # 8-bit quantized on CPU is 3-4x faster than float32

        details = f"Device: CPU | Engine: Quantized {target_compute} (portable mode)"
        return HardwareProfile(
            device="cpu",
            compute_type=target_compute,
            gpu_name=None,
            vram_mb=None,
            is_gpu=False,
            cuda_count=0,
            details=details
        )

def resolve_model_cache_dir(custom_path: Optional[Path] = None) -> Optional[Path]:
    """
    Locates an existing Whisper / HuggingFace model cache to prevent redundant 3GB downloads.
    Hierarchically checks:
    1. Explicit custom path (--model-dir)
    2. WHISPER_CACHE_DIR environment variable
    3. HF_HOME environment variable
    4. User's global ~/.cache/huggingface/hub if already populated
    """
    if custom_path and custom_path.exists():
        return custom_path.resolve()

    for env_var in ["WHISPER_MODELS_DIR", "WHISPER_CACHE_DIR", "HF_HOME"]:
        val = os.environ.get(env_var)
        if val and Path(val).is_dir():
            return Path(val).resolve()

    # Check global user HuggingFace hub cache
    hf_global = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_global.is_dir():
        return hf_global

    return None
