# Rebuild guide (for an AI or a developer starting cold)

This file lets someone with **no prior context** install, verify, understand
and modify the player. Each section ends with a way to check it.

Related repositories: `stg-recorder` (records the input on a Raspberry Pi) and
`stg-studio` (calibrates, preprocesses and trains the models this plays).

---

## 1. What the player shows

A four-camera rig (about 15 x 9 cm) recorded a scene; `stg-studio` trained a
SpacetimeGaussians (STG) "lite" model from it. The player renders that model
as **a window**: the rig frame is the window frame, the viewer stands behind
it and may move a little, and only views inside the convex hull of the
cameras are expected to look right. Everything outside the window is covered
by a fogged wall; the key help is drawn **on** that wall (as a 3D texture, so
it has correct stereo disparity in VR).

## 2. Installation, step by step

The repository's `.venv` is independent (no `--system-site-packages`), so
packages installed elsewhere on the machine cannot break it.

| # | Action | Done when |
|---|---|---|
| 1 | Install NVIDIA driver, CUDA Toolkit 13.0, Visual Studio with C++ workload, Python 3.12, git (SteamVR for VR) | `nvcc --version` shows 13.0 |
| 2 | `py -3.12 -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130` | `torch.cuda.is_available()` is `True`, numpy is 1.26.4 |
| 3 | `git -c core.longpaths=true -c core.autocrlf=false clone https://github.com/oppo-us-research/SpacetimeGaussians.git third_party\SpacetimeGaussians`, `git checkout 427abfc`, `git -c core.autocrlf=false apply --whitespace=nowarn upstream\stg_427abfc.patch` | `git status --short` shows 12 `M` lines |
| 4 | After `vcvars64.bat`, with `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX"`, `DISTUTILS_USE_SDK=1`: `pip install --no-build-isolation third_party\SpacetimeGaussians\thirdparty\gaussian_splatting\submodules\gaussian_rasterization_ch3` | `python -c "import torch, diff_gaussian_rasterization_ch3"` works (**torch first**) |
| 5 | `.venv\Scripts\python.exe player\selfcheck.py` | first stderr line `[OK] <GPU> <GB>`, exit code 0 |
| 6 | `.venv\Scripts\pythonw.exe player\launcher.py` | the menu opens, the GPU line is grey (not red) |

Why the trainer source is needed at all: `render_at.py` imports
`utils.graphics_utils`, `utils.sh_utils` and `scene/colmap_loader.py` from
`third_party/SpacetimeGaussians/thirdparty/gaussian_splatting`, and the
rasterizer is compiled from its `submodules/`. The patch matters for the
compile flags (`-allow-unsupported-compiler -std=c++20` in `setup.py`).
`STG_HOME` can point the player at another copy.

## 3. Input data

A **shooting root** as produced by `stg-studio`:

```
<root>/
    stg<N>/                       dataset
        colmap_0/sparse/0/        images.bin, cameras.bin: camera poses and intrinsics (all time steps share them)
        window.json               metadata (below)
        audio/audio.wav           optional
    <name>_out/                   model
        point_cloud/iteration_<k>/point_cloud.ply
        cfg_args                  written by the trainer; source_path names the dataset
        window.json               only when a time range was trained (overrides the dataset's)
    player.json                   optional: {"default": {"model": "I_out", "iter": 100000}}
```

`window.json` fields read by the player:

| Field | Use |
|---|---|
| `duration_s` | seconds per loop (time in the model is normalized 0..1) |
| `background` | background colour the model was trained with; must match |
| `unit_mm` | mm per scene unit (1000 = metres); `metric: false` disables audio localization and the wall HUD, and VR refuses to play |
| `win`, `win_order` | window frame corners (9 decimals). Missing -> a nominal 150x95 mm black frame, labelled NOMINAL |
| `audio` | `path`, `t0_offset_s`, `gain_db`, `mic_spacing_mm` |
| `segments`, `segment_ramp` | combined-training models: row ranges of the PLY per time segment, cross-faded at boundaries |

PLY fields (STG lite): `x y z`, `opacity`, `scale_0..2`, `rot_0..3`,
`f_dc_0..2` (RGB), `trbf_center`, `trbf_scale`, `motion_0..8`, `omega_0..3`.

## 4. Code map and design decisions

| File | Role and key decisions |
|---|---|
| `launcher.py` | tkinter menu. Runs `selfcheck.py` and the player as **subprocesses**: a CUDA context in the menu would hold GPU memory while it stays open, and a separate process frees everything when it exits. Play is enabled only when a model is chosen **and** the self-check passed. Errors are shown in a copyable text window. |
| `player.py` | argument parsing; checks VR readiness **before** loading the model; imports `vr_viewer` (glfw/openvr/OpenGL) only for VR. |
| `player_core.py` | `Scene`: loads poses from `colmap_0`, computes the window frame from the camera layout (not from one camera's orientation), reads `window.json`, loads the PLY, sets up audio. `vr_status()` queries OpenVR without initializing it. |
| `viewer.py` | flat player: OpenCV window, mouse look, N/M focal length steps, panel on the right. Uses `cv2.pollKey()` (`waitKey(1)` actually waits 15.6 ms on Windows). `render_once()` renders one frame without window or audio, for regression tests. |
| `vr_viewer.py` | VR: the rasterizer only supports symmetric frustums, so each eye renders an enclosing symmetric frustum and crops the off-axis part. Coordinate conversion OpenVR <-> COLMAP via `F = diag(1,-1,-1)`; the first head pose is the origin (OpenVR's Y=0 is the floor). The desktop window shows a status panel and receives keys. |
| `render_at.py` | the renderer. STG time model: `op = opacity * exp(-(dt / exp(trbf_scale))^2)`, polynomial motion, rotation + omega x dt. Time-window culling drops Gaussians with `op <= 0.005` (86-89% per frame, about 5x faster). |
| `hudwall.py` | wall fog, rounded window frame and text, all built on the wall plane and projected (correct perspective and disparity). ASCII only (cv2 fonts). |
| `audio4d.py` | **the audio clock is the master clock**; the picture asks `now_t()` each frame. Two virtual speakers at the microphone positions on the window plane, ITD + ILD per ear (world-locked stereo). Never raises; without audio the picture follows the wall clock. |
| `captureset.py` | model/dataset discovery from folder names and `cfg_args`; `player.json` only stores the default choice. |
| `selfcheck.py` | imports, CUDA, and **one real rasterization** (catches "no kernel image" for an uncompiled GPU architecture); prints a JSON line for the launcher. |

### Pitfalls already met (do not reintroduce)

| Symptom | Cause | Where handled |
|---|---|---|
| `CUDA illegal memory access` at some angles | quaternions with norm exactly 0 -> NaN when normalized by hand | `F.normalize` in `render_at.load_ply` and `render` |
| crash on the second render in a process | `means2D` must be a separate tensor with gradient | `render_at.render` |
| crash when moving into the scene | `znear` from the nearest Gaussian; huge screen radii overflow tile indices | znear tied to the median depth; Gaussians with radius > image diagonal dropped |
| crash in VR when lowering the cull threshold | int32 overflow in the rasterizer at VR resolution | lower limit 0.005 in `cull_step` |
| audio sounds stepped | audio chased the frame time | audio is the master clock |
| tilted window frame | screen axes taken from one camera | `window_frame()` from the camera layout |
| text glued to the screen in VR | text drawn in screen space | textures pasted onto the wall plane |

## 5. Verification

| # | Check | Pass |
|---|---|---|
| 1 | `.venv\Scripts\python.exe player\selfcheck.py` | exit code 0 |
| 2 | `.venv\Scripts\python.exe player\player.py --check-vr` | prints the VR state (not ready is fine without a headset) |
| 3 | regression: with a shooting root, `Viewer(ply, ds).render_once(t, dpos_mm, yaw)` before and after a change | arrays identical (`numpy.array_equal`) unless the change is meant to alter the picture |
| 4 | play a model in the launcher | **visual inspection**: scene inside the frame, wall with the key help, moving and looking around work |
| 5 | VR | inspect with the headset; depth errors are only visible in stereo |

Numeric metrics such as PSNR do not reveal depth errors; acceptance is by eye.
