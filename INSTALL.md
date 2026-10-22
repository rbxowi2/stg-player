# Installing STG player

Everything except the system prerequisites is installed **inside this
repository folder** by `setup.ps1`. Moving or deleting the folder removes the
installation; nothing else on the machine is changed apart from one desktop
shortcut. The system Python is only used to create `.venv` (an independent
environment without access to system packages); nothing is installed into it.

The player is installed independently of `stg-studio`: each repository has
its own `.venv` and its own copy of the trainer source.

## 1. System prerequisites (by hand)

| Software | Version used | Notes |
|---|---|---|
| Windows | 11 (10 also works) | |
| NVIDIA GPU + driver | RTX 20xx or newer | `nvidia-smi` must work |
| CUDA Toolkit | **13.0** | needed to compile the rasterizer; default location `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` |
| Visual Studio | 2022 or newer, workload **"Desktop development with C++"** | Community edition is enough |
| Python | **3.12** | from python.org |
| git | any recent | |
| SteamVR | only for VR | a headset connected through SteamVR (Quest 2 with Link was used) |

**Check:**

```powershell
nvidia-smi
nvcc --version          # "release 13.0"
py -3.12 --version      # Python 3.12.x
git --version
```

## 2. Run setup.ps1

From this folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

It takes 15-30 minutes (mostly downloading PyTorch and compiling). Each step
prints `OK` or `ERROR`; on an error, fix the cause and run the script again -
finished steps are skipped.

| Step | Creates | Check that it worked |
|---|---|---|
| 1 prerequisites | nothing | five `OK` lines |
| 2 Python environment | `.venv\` | prints torch / `True` / numpy 1.26.4 / OpenCV 4.11 |
| 3 trainer source | `third_party\SpacetimeGaussians\` at commit 427abfc with `upstream\stg_427abfc.patch` applied | "patch applied" |
| 4 CUDA rasterizer | `diff_gaussian_rasterization_ch3` inside `.venv` | "built and importable" |
| 5 self-check | nothing | `[OK] <GPU name>  <memory>` |
| 6 shortcut | `STG player.lnk` on the desktop | the file exists |

Options: `-CudaArch 8.6` compiles only for one GPU generation (8.6 = RTX 30xx,
8.9 = RTX 40xx, 12.0 = RTX 50xx), `-Python`, `-CudaHome`,
`-RebuildExtensions`, `-NoShortcut`. `Get-Help .\setup.ps1 -Detailed` lists them.

## 3. Start

Double-click **STG player** on the desktop, or:

```powershell
.venv\Scripts\pythonw.exe player\launcher.py
```

1. **Root**: choose the shooting root - the folder that **contains** the model
   folders (`fast_out\`, ...) and the dataset (`stg301\`), not a model folder itself.
2. **Model**: every model and saved iteration is listed, newest first.
3. **Mode**: Flat or VR (VR is greyed out with the reason when SteamVR or the
   headset is not ready).
4. **Play**.

**Check:** the GPU line shows the GPU name and memory (the self-check passed),
the model list is filled, and Play opens the player window with the scene
inside a framed window and the key help on the wall to its left.

From the command line:

```powershell
.venv\Scripts\python.exe player\player.py <shooting root>              # flat
.venv\Scripts\python.exe player\player.py <shooting root> --vr         # VR
.venv\Scripts\python.exe player\player.py --check-vr                   # only report the VR state
```

## Keys (flat)

| Key | Action |
|---|---|
| W S A D Q E | move forward / back / left / right / down / up |
| mouse drag | look around |
| N M | focal length wide / tele |
| space | pause |
| - = 0 | slower / faster / normal speed |
| R | reset view |
| F | window frame and wall on/off |
| B | black/white background (shows transparent regions) |
| T, 4 5 6 | time-window culling on/off, threshold default / smaller / larger |
| J K ; ' | mute, world-locked/plain stereo, volume down/up |
| H G ESC | hide panel, fullscreen, quit |

The same keys are drawn on the wall next to the window. In VR the head moves
the view; the keyboard focus must be on the small desktop window.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| GPU line red: "no kernel image" | the rasterizer was not compiled for this GPU; `setup.ps1 -RebuildExtensions -CudaArch <value>` |
| GPU line red: no usable NVIDIA GPU | driver missing or too old for CUDA 13.0 |
| "This is a model folder, not the root" | choose the parent folder |
| VR greyed out | SteamVR not installed, or the headset not connected/on; flat mode is unaffected |
| `SpacetimeGaussians ... not found` | `third_party\SpacetimeGaussians` missing; run setup again, or set `STG_HOME` |
| playback has no sound | the dataset has no audio, or no audio device; the panel shows `audio off` |
| black window frame, `NOMINAL` in the panel | the dataset's `window.json` has no window corners (preprocessed without calibration); the frame size is a placeholder |
