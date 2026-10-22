#!/usr/bin/env python3
"""selfcheck.py - will the player start? Answered before a model is loaded.

    python selfcheck.py            human-readable report + a final JSON line
    (the launcher runs this in the background and reads the final JSON line)

## Why this exists

When something is missing, the symptom is always **waiting tens of seconds
for the model to load, then a crash**. The wait is wasted, and the message is
a traceback most people cannot read.

## Why it runs in a separate process instead of being imported by the launcher

`import torch` creates a CUDA context that holds hundreds of MB of GPU memory
**until the process ends**. The launcher stays open (the user can play again
and again), so that memory would stay taken and be missing from the player.
In a subprocess, everything is returned once the check is done - the same
reason the player itself runs as a subprocess (see `launcher.py`).

## Why it must actually rasterize once

The earlier checks (modules present, GPU present, driver working) only answer
"is it there". Only **sending Gaussians through the rasterizer** answers
"does it work" - `no kernel image is available for execution on the device`
(the GPU's architecture was not compiled in) appears only when a kernel is
actually launched.

Measured cost: `import torch` 2.08 s + minimal rasterization 0.11 s = **2.11 s**,
in exchange for not waiting tens of seconds after pressing play only to fail.
"""
from __future__ import annotations

import json
import sys
import time

# All four are required. ch3 is the rasterizer used by every model (any tier).
NEEDED = ["torch", "cv2", "plyfile", "diff_gaussian_rasterization_ch3"]

# Architectures compiled into ch3 by default (setup.ps1 -CudaArch).
# This list is **not used for decisions**; it only tells the user what is
#    supported when the check fails. The real test is always "run it once and
#    see" - a hard-coded list goes stale.
BUILT_ARCHS = "7.5 / 8.0 / 8.6 / 8.9 / 9.0 / 12.0"


def driver_version():
    """Driver version, or None if it cannot be read - for the user only, not used for decisions."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        v = out.stdout.strip().splitlines()
        return v[0].strip() if v else None
    except Exception:                                        # noqa: BLE001
        return None


def tiny_render():
    """Minimal rasterization: 4 Gaussians, 64x64. Exceptions propagate to the caller, which records them."""
    import torch
    from diff_gaussian_rasterization_ch3 import (GaussianRasterizationSettings,
                                                 GaussianRasterizer)
    dev, n = "cuda", 4
    xyz = torch.zeros(n, 3, device=dev)
    xyz[:, 2] = 2.0
    rot = torch.zeros(n, 4, device=dev)
    rot[:, 0] = 1.0
    eye = torch.eye(4, device=dev)
    rs = GaussianRasterizationSettings(
        image_height=64, image_width=64, tanfovx=0.5, tanfovy=0.5,
        bg=torch.zeros(9, device=dev), scale_modifier=1.0, viewmatrix=eye,
        projmatrix=eye, sh_degree=0, campos=torch.zeros(3, device=dev),
        prefiltered=False)
    sp = torch.zeros_like(xyz, requires_grad=True) + 0
    out = GaussianRasterizer(raster_settings=rs)(
        means3D=xyz, means2D=sp, shs=None,
        colors_precomp=torch.full((n, 9), 0.5, device=dev),
        opacities=torch.full((n, 1), 0.5, device=dev),
        scales=torch.full((n, 3), 0.05, device=dev),
        rotations=rot, cov3D_precomp=None)
    torch.cuda.synchronize()
    return tuple(out[0].shape)


def run():
    """Returns a dict. **Every failing step has its own `stage` and a plain-language message.**"""
    from version import PLAYER_VERSION
    r = {"ok": False, "version": PLAYER_VERSION, "stage": "", "msg": "",
         "detail": "", "secs": 0.0}
    t0 = time.perf_counter()

    # -- 1 modules present (fast, no import)
    import importlib.util as iu
    miss = [m for m in NEEDED if iu.find_spec(m) is None]
    if miss:
        r["stage"] = "modules"
        r["msg"] = "This Python environment is missing packages: " + ", ".join(miss)
        r["detail"] = f"Interpreter in use:\n{sys.executable}"
        return r

    # -- 2 a usable NVIDIA GPU
    try:
        import torch
    except Exception as e:                                   # noqa: BLE001
        r["stage"] = "import"
        r["msg"] = "Loading torch failed; this usually means a system component is missing"
        r["detail"] = f"{type(e).__name__}: {e}"
        return r

    r["torch"] = torch.__version__
    r["cuda_runtime"] = torch.version.cuda
    r["driver"] = driver_version()

    if not torch.cuda.is_available():
        r["stage"] = "cuda"
        r["msg"] = ("No usable NVIDIA GPU found.\n"
                    "This computer may have no NVIDIA GPU, or **the GPU driver is too old**.")
        r["detail"] = (f"torch {torch.__version__} (bundled CUDA "
                       f"{torch.version.cuda} runtime)\n"
                       f"Detected driver version: {r['driver'] or 'unknown'}\n"
                       "-> Update the GPU driver from NVIDIA and try again.")
        return r

    p = torch.cuda.get_device_properties(0)
    r["gpu"] = p.name
    r["cc"] = f"{p.major}.{p.minor}"
    r["vram_mb"] = p.total_memory // 1024 // 1024

    # -- 3 actually rasterize once - only this proves that it works
    try:
        r["shape"] = tiny_render()
    except Exception as e:                                   # noqa: BLE001
        txt = f"{type(e).__name__}: {e}"
        r["stage"] = "raster"
        if "no kernel image" in txt:
            r["msg"] = (f"This GPU is a {p.name} (compute capability {r['cc']}), "
                        "which the compiled rasterizer does not support.")
            r["detail"] = (f"Compiled for: {BUILT_ARCHS}\n"
                           "-> Recompile with setup.ps1 -RebuildExtensions -CudaArch <your value>; a driver update does not help.\n\n" + txt)
        else:
            r["msg"] = "The GPU is usable, but the rendering kernel does not run."
            r["detail"] = txt
        return r

    # -- 4 the three VR packages (**reported only, never blocking**)
    #
    # Why check them: the native DLLs of `glfw` / `openvr` are loaded at run
    #    time through ctypes, and `OpenGL` imports its platform backend under a
    #    **computed module name**. Missing pieces are easy to overlook and
    #    **would only show up with the headset on**.
    #
    # Flat mode does not need them, so a failure here does not affect `ok`.
    try:
        import glfw            # noqa: F401
        import openvr          # noqa: F401
        from OpenGL.GL import glGetString  # noqa: F401
        r["vr_modules"] = "ok"
    except Exception as e:                                   # noqa: BLE001
        r["vr_modules"] = f"{type(e).__name__}: {e}"

    r["ok"] = True
    # Rounded, not integer division: a 12287 MB card would show as "11 GB"
    #    and look misdetected (it is a 12 GB card).
    r["msg"] = f"{r['gpu']}   {round(r['vram_mb'] / 1024)} GB"
    r["secs"] = round(time.perf_counter() - t0, 2)
    return r


def main():
    r = run()
    # the human-readable part goes to stderr and the JSON to stdout - they never mix
    print(("[OK] " if r["ok"] else "[FAIL] ") + r["msg"], file=sys.stderr)
    if r["detail"]:
        print(r["detail"], file=sys.stderr)
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
