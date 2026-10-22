"""Shared base of the player - scene loading, window geometry and viewpoint limits for flat and VR.

## Why this file exists

The flat viewer (`viewer.py`) and the VR viewer (`vr_viewer.py`) used to have
their own loading code and geometry functions. The same logic in two places
means **changing one and forgetting the other** - that is how "the wall
stutters in VR" happened (the flat wall blending was optimized, the VR copy
was not), and how the error messages of `find_meta` had drifted apart.

## Three differences are deliberate; do not unify them

They are passed in by the caller; this file **hard-codes none of them and has
no `if vr` branches** (branches would make the two modes affect each other):

    start_back_mm    flat 750 / VR 120     on screen the whole window must be visible; in VR the face is at the window
    wall density     flat 0.87 / VR 0.82   in VR the wall should read like a real wall
    frame colour     BGR / RGB             cv2 and OpenGL use different channel orders

A fourth difference is `require_metric`: without a metric scale, flat mode
only warns and disables audio and the wall HUD, while **VR refuses to play** -
head movement is in real metres, and a wrong scale is uncomfortable.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import captureset as CS
import render_at as R

from version import PLAYER_VERSION      # noqa: F401  (kept for the old import path)

try:
    import audio4d as A4
except Exception as _e:                                    # noqa: BLE001
    print(f"  ERROR audio4d failed to load; playing without sound: {_e}")

    class A4:                                              # type: ignore[no-redef]
        class Audio:
            ok = False
            info = "no audio module"

            def __init__(self, *a, **k):
                pass

            def __getattr__(self, _n):
                return lambda *a, **k: None

            def hud(self):
                return "off"


# Fallback used only when window.json has no unit_mm.
#    The value is "the measured longest rig diagonal in mm", which **changes
#    with every recalibration**; hard-coding it in the player is a design
#    flaw - the right source is unit_mm in window.json, computed by preprocessing.
RIG_MAX_MM_FALLBACK = 174.6

# Without `win` in `window.json`, a **nominal window frame** of this size is drawn.
#    Not to "guess the real window" - but so that **the wall still exists**,
#    because the key help is drawn on the wall (`hudwall.FLAT_KEYS`); without
#    a wall the user cannot even see the keys.
#    It is a **hard-coded nominal value** and is labelled NOMINAL on screen.
#    The real window always comes from calibration.
FALLBACK_WIN_MM = (150.0, 95.0)

MOVE_MM = 35.0              # distance per key press (mm)
WALL_MARGIN_MM = 20.0       # minimum distance between viewpoint and wall
DEFAULT_SECS = 10.0         # clip length without metadata; labelled ASSUMED on screen

# Default threshold for time-window culling. This is also the lower limit -
#    below it an int32 overflow crashes the renderer (measured in VR).
CULL_DEFAULT = R.CULL_DEFAULT


def vr_status():
    """Is the VR device ready? Returns `(ok, explanation)`.

    **This does not initialize OpenVR**, so calling it on a machine without a
    headset does not crash - `isRuntimeInstalled()` and `isHmdPresent()` are
    queries that leave no state behind.

    Four states are reported separately, because what is missing and what the
    user must do differ:

        module missing      this Python has no openvr package
        runtime missing     no SteamVR on this computer
        headset absent      SteamVR installed, headset not connected or off   <- most common
        ready               both true

    **Do not import glfw or OpenGL here.** Detection must complete on a
       machine with no VR at all, and either package missing would make the
       import fail - "you need VR to find out whether you can use VR".
    """
    try:
        import openvr
    except Exception as e:                                 # noqa: BLE001
        return False, f"this Python has no openvr package ({e})"
    try:
        if not openvr.isRuntimeInstalled():
            return False, "SteamVR not detected (OpenVR runtime not installed)"
        if not openvr.isHmdPresent():
            return False, "SteamVR is installed, but the headset is not connected or not on"
    except Exception as e:                                 # noqa: BLE001
        # a failing query also counts as not ready - detection must never bring the player down
        return False, f"querying the VR state failed: {e}"
    return True, "VR ready"


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def find_meta(ply, root):
    """Find window.json. Returns (dict, source path) or (None, None)."""
    d = os.path.dirname(os.path.abspath(ply))
    cands = [d,                                        # next to the PLY
             os.path.dirname(os.path.dirname(d)),      # two levels above point_cloud/iteration_N/
             os.path.abspath(root)]
    for c in cands:
        f = os.path.join(c, "window.json")
        if os.path.isfile(f):
            try:
                return json.load(open(f, encoding="utf-8")), f
            except Exception as e:                     # noqa: BLE001
                print(f"  ! reading {f} failed: {e}")
    return None, None


def window_frame(centers, Rref):
    """**The window's own frame** from the rig geometry; returns the world -> view rotation.

    **Do not use the reference camera's orientation as the screen horizontal.**

    Using cam_A's pose as the basis ties the screen horizontal to cam_A. But
    the cameras are **mounted on the rig** with 1-2 deg mechanical tolerance
    each - cam_A's right axis was measured 1.59 deg off the rig's long side -
    so **the window frame appears tilted on screen**: the top edge spanning
    498 px is off by 14 px, visible by eye.

    The window is the viewer's origin and should always be level, on screen
    and in VR. So the basis comes from **the placement of the four cameras**,
    not the orientation of one:

        X (right)    long side direction       <- direction of largest variance of the camera centres
        Y (down)     short side direction
        Z (forward)  rig plane normal          = X x Y

    Signs are resolved with the reference camera's axes (the cameras are
    mounted roughly upright, within 1-2 deg) to remove the 180 deg ambiguity.

    This also works for **older datasets** - it only uses the relative
    camera centres, regardless of the preprocessing version.
    """
    P = np.asarray(centers, float)
    ctr = P.mean(0)
    _, _, Vt = np.linalg.svd(P - ctr)
    long_dir, short_dir, normal = Vt[0], Vt[1], Vt[2]

    x = long_dir * (1.0 if np.dot(long_dir, Rref[0]) >= 0 else -1.0)
    y = short_dir * (1.0 if np.dot(short_dir, Rref[1]) >= 0 else -1.0)
    z = np.cross(x, y)
    if np.dot(z, Rref[2]) < 0:
        y, z = -y, -z
    x = x / np.linalg.norm(x)
    y = y - x * np.dot(x, y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z])


class Scene:
    """A loaded 4DGS scene - window geometry, metric scale, model, audio, playback state.

    Both the flat and VR viewers inherit from it. **It touches no display API**
    (no cv2, no OpenGL, no OpenVR), so it loads on a machine where no VR
    device can be detected.
    """

    def __init__(self, ply, root, secs=None, start_back_mm=750.0,
                 require_metric=False):
        m0 = os.path.join(root, "colmap_0", "sparse", "0")
        E = R.cl.read_extrinsics_binary(os.path.join(m0, "images.bin"))
        C = {v.name: dict(Rc=R.cl.qvec2rotmat(v.qvec),
                          C=-R.cl.qvec2rotmat(v.qvec).T @ np.array(v.tvec))
             for v in E.values()}
        self.m0 = m0
        self.names = sorted(C)
        centers = np.stack([C[n]["C"] for n in self.names])
        self.centers = centers
        self.ctr = centers.mean(0)
        self.Rc0 = window_frame(centers, C[self.names[0]]["Rc"])
        self.fwd0 = self.Rc0[2]                           # wall normal (toward the scene)
        self.span = max(np.linalg.norm(a - b) for a in centers for b in centers)

        meta, mpath = find_meta(ply, root)
        self.meta, self.mpath = meta, mpath
        # Order: `_load_scale` first, because the fallback frame without win
        #    needs `self.u` to convert mm into scene units.
        self._load_scale(meta, mpath, require_metric)
        self._load_window(meta)

        print("  loading...")
        self.g = R.load_ply(ply)
        self.n_gauss = int(self.g["xyz"].shape[0])
        print(f"  {self.n_gauss:,} Gaussians")
        print(f"  scale 1 unit = {self.u:.3f} mm  <- {self.usrc}")

        # Segment table of combined training (None = an ordinary single-segment model, unchanged behaviour).
        # Segment boundaries are row ranges in the PLY; rendering shows only the current segment (see render_at.render).
        self.segments = (meta or {}).get("segments") or None
        self.seg_ramp = float((meta or {}).get("segment_ramp") or 0.0)
        if self.segments:
            print(f"  combined training model: {len(self.segments)} segments"
                  f" (cross-fade {self.seg_ramp:.4f})")

        self._load_background(meta)
        self._load_duration(meta, mpath, secs)

        # which shoot and which model - several experiments may share a dataset
        self.shot, self.dataset, self.model = CS.provenance(ply, root)
        print(f"  shooting root {self.shot}   dataset {self.dataset}   model {self.model}")

        self.pos = self.ctr - self.fwd0 * (start_back_mm / self.u)
        self.start_back_mm = start_back_mm
        self.t, self.speed, self.playing = 0.0, 1.0, True

        # World-locked virtual stereo: two virtual speakers fixed on the window
        #    plane, at the real positions of the two microphones on the rig.
        #    Turning the head leaves the sound at the window - a direct
        #    consequence of the "window" definition: the sound field outside
        #    does not turn with the viewer. Without a scale no audio is set up:
        #    the speed-of-sound conversion would be wrong (see audio4d.py).
        self.audio = A4.Audio(meta if self.metric else None,
                              root, mpath, self.u, self.secs,
                              win_ctr=self.ctr,
                              win_right=self.Rc0[0], win_fwd=self.fwd0)
        self.audio.start()

        # Time-window culling. On by default - measured 4.7x faster with 70%
        #    less VRAM, and no difference visible by eye. It stays a toggle
        #    for A/B comparison at any time, especially **flicker in motion**,
        #    which static screenshots cannot verify.
        self.cull = CULL_DEFAULT
        self.n_sent = 0

    # ------------------------------------------------------------ the four loading steps
    def _load_window(self, meta):
        """Window frame corners: **read `win` from `window.json`, not computed here**.

        Why not computed here

        Recomputing from the colmap extrinsics duplicated what preprocessing
        already writes into `window.json` - **two implementations of one value**.

        And that algorithm (angle sort around the centroid) **is only right
        when every camera lies on the convex hull**. With a camera inside (for
        example a planned fifth, central camera) the frame caves in, 25% less
        area with synthetic coordinates. Fix it once, not in two places.

        `window.json` is already the interface between stages; the player
        reads `unit_mm` / `duration_s` / `background` from it, so reading
        `win` adds no dependency.

        Without `win`, a **nominal frame** is drawn rather than none

        Drawing nothing would avoid drawing a wrong window, **but the key help
        would disappear with it** - the help is drawn **on the wall**
        (`hudwall.FLAT_KEYS`), and without a wall the user would not know the
        keys.

        So a **nominal frame hard-coded in the program** (`FALLBACK_WIN_MM`)
        is drawn: wall and help remain, and **the screen labels it NOMINAL**.

        The key point is that it is **not computed from data**: it is an
        obvious nominal value and cannot be mistaken for the real window of
        this recording. The alternative, "the player computes the convex hull
        itself", is what must be avoided - it would put the hull bug on **the
        least tested path**, and the result would look real.

        This state is not theoretical: `win` is **written only by stage 2**.
        Running stage 1 only (`--stage1`, or a shooting root without `rig/`)
        leaves it out, and `write_stage1()` rewrites the whole `window.json`.
        """
        w = (meta or {}).get("win")
        if w:
            self.wincorners = [np.asarray(p, float) for p in w]
            self.win_nominal = False
            self.win_src = f"window.json ({len(self.wincorners)} corners)"
            if meta.get("win_order"):
                self.win_src += ": " + ", ".join(meta["win_order"])
        else:
            # nominal frame: a rectangle on the rig plane along the window frame's right/down axes.
            hw, hh = (FALLBACK_WIN_MM[0] / 2 / self.u,
                      FALLBACK_WIN_MM[1] / 2 / self.u)
            right, down = self.Rc0[0], self.Rc0[1]
            self.wincorners = [self.ctr - right * hw - down * hh,
                               self.ctr + right * hw - down * hh,
                               self.ctr + right * hw + down * hh,
                               self.ctr - right * hw + down * hh]
            self.win_nominal = True
            self.win_src = (
                f"**nominal {FALLBACK_WIN_MM[0]:g}x{FALLBACK_WIN_MM[1]:g} mm"
                f" (hard-coded, not the real window of this recording)** - "
                f"window.json has no win; running stage 2 (--rig) adds it")
        print(f"  window frame {self.win_src}")

    def _load_scale(self, meta, mpath, require_metric):
        """1 SfM unit = how many mm. Metadata first; the hard-coded rig diagonal only as fallback."""
        self.metric = not (meta and meta.get("metric") is False)
        if not self.metric and require_metric:
            # VR must refuse. Head movement is in real metres; with wrong world
            #    units one step flies far away, and it is uncomfortable. The
            #    flat player can warn; VR cannot.
            raise SystemExit(
                "  ERROR this dataset has **no metric scale**; VR cannot play it.   "
                "Head movement is in real metres, and wrong world units distort the whole space.   "
                "-> measure the six edges, run calibrate.py with --edge, then run prep_4d.py again")
        if meta and "unit_mm" in meta:
            self.u = float(meta["unit_mm"])
            self.usrc = f"window.json (rig diagonal {self.u * self.span:.1f} mm)"
        elif not self.metric:
            # The calibration has no metric scale. The fallback ("measured rig
            #    diagonal in mm") applied to unscaled data only gives a
            #    plausible-looking wrong number, so it is not used; metric
            #    features are disabled instead.
            self.u = 1.0
            self.usrc = ("**no metric scale** (calibrated without --edge)"
                         " - audio localization and wall HUD disabled")
        else:
            self.u = RIG_MAX_MM_FALLBACK / self.span
            self.usrc = (f"fallback RIG_MAX_MM_FALLBACK={RIG_MAX_MM_FALLBACK}"
                         f" - metric readings not guaranteed")

    def _load_background(self, meta):
        """The background colour **must match training**.

        Final pixel = sum(Gaussian contributions) + background x prod(1 - alpha).
        With a black training background, a region that "should be black" can
        be represented two ways: black opaque Gaussians, or **nothing at all**
        (the black background provides black). The loss cannot tell them
        apart, so "leaving it empty" costs nothing - the optimizer has no
        reason to fill dark areas.

        Measured on a dark scene (mean brightness 60/255):
            model H: 34.0% of pixels alpha < 0.99, 9.3% alpha < 0.7
            model I: 24.5%                      , 6.3%
        Played on white, those holes turn into white fog. Black vs white
        background differ by only 14.6 dB PSNR.

        So the background is **a property of the model**, given by
        `background` in window.json; the player does not guess. Without it,
        0.0 is used (the trainer's default).
        """
        self.bg = float(meta["background"]) if meta and "background" in meta else 0.0
        self.bg_known = bool(meta and "background" in meta)
        print(f"  background {self.bg:.2f}"
              f"{'' if self.bg_known else '  ! not in window.json; using the trainer default 0.0'}"
              f"   (key B toggles it, to check where the model is transparent)")

    def _load_duration(self, meta, mpath, secs):
        """Seconds per loop: command line > window.json > default."""
        if secs is not None:
            self.secs, self.secs_known, src = float(secs), True, "command line"
        elif meta and "duration_s" in meta:
            self.secs, self.secs_known = float(meta["duration_s"]), True
            src = f"window.json ({mpath})"
        else:
            self.secs, self.secs_known, src = DEFAULT_SECS, False, \
                "default - no window.json found; playback speed does not match the recording"
        print(f"  clip length {self.secs:.4f} s  <- {src}")

    # ------------------------------------------------------------ viewpoint
    def clamp(self, p):
        """The viewpoint cannot pass through the wall (rig plane) - only this side of it."""
        d = float(np.dot(p - self.ctr, self.fwd0))        # >0 = past the wall
        lim = -WALL_MARGIN_MM / self.u
        if d > lim:
            p = p - self.fwd0 * (d - lim)
        return p

    def move(self, Rn, dx, dy, dz):
        """Move one step along the current orientation `Rn`, staying on this side of the wall."""
        step = MOVE_MM / self.u
        self.pos = self.clamp(
            self.pos + (Rn[0] * dx + Rn[1] * dy + Rn[2] * dz) * step)

    def home(self):
        """Back to the start position (start_back_mm behind the window)."""
        self.pos = self.ctr - self.fwd0 * (self.start_back_mm / self.u)

    def say_cull(self):
        print(f"  time culling {'off' if self.cull is None else '%g' % self.cull}"
              f"   last frame sent {self.n_sent:,} / {self.n_gauss:,}"
              f" ({self.n_sent / max(self.n_gauss, 1) * 100:.0f}%)", flush=True)
