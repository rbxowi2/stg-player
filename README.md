# stg-player

Plays 4D Gaussian Splatting models trained with `stg-studio`, on a flat
screen or in VR (SteamVR / OpenVR).

The model is shown as **a window**: the scene is seen through the frame of
the four-camera rig that recorded it, and the viewpoint can move slightly in
front of it. Outside the window a wall covers what the cameras never saw; the
key help is drawn on that wall. Audio recorded by the rig is played as
world-locked stereo (the sound stays at the window when the head turns).

## Install

Windows 10/11 with an NVIDIA GPU. See [INSTALL.md](INSTALL.md); in short:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

then start **STG player** from the desktop shortcut, choose a shooting root
(the folder containing the `*_out` model folders and the dataset), and press Play.

## Layout

```
player/
    launcher.py      start menu (tkinter): choose root, model, flat/VR, scale
    player.py        command-line entry point: python player.py <root> [--vr]
    viewer.py        flat-screen player (OpenCV window)
    vr_viewer.py     VR player (OpenVR + OpenGL)
    player_core.py   shared scene loading, window geometry, viewpoint limits
    render_at.py     Gaussian rasterization at any pose (uses the trainer's CUDA rasterizer)
    hudwall.py       wall, window frame and text drawn on the wall plane
    audio4d.py       audio with the audio clock as master clock
    captureset.py    finds models and datasets in a shooting root
    selfcheck.py     GPU / rasterizer check before loading a model
    version.py       PLAYER_VERSION
upstream/            stg_427abfc.patch: changes made to the upstream trainer source
THIRD_PARTY_LICENSES/
```

[REBUILD_FOR_AI.md](REBUILD_FOR_AI.md) describes the whole program in enough
detail to rebuild or modify it without prior context.

## Upstream

* SpacetimeGaussians, https://github.com/oppo-us-research/SpacetimeGaussians (commit 427abfc), MIT
* gaussian-splatting (Inria / MPII), contained in SpacetimeGaussians, under its own license

Their license texts are in [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/).
