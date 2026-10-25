#!/usr/bin/env python3
"""viewer.py - interactive 4DGS player, flat screen

    python player.py <shooting root> [--model I] [--iter 100000]     <- the entry point
    python player.py <STG PLY> <colmap_N root> [--secs N]           <- older form, still supported

Everything produced by one shoot lives in **one root folder** (calibration,
datasets, models), so playback only needs that folder, and the whole folder
can be moved. Which model/iteration: `player.json` > command line > automatic
(most recently modified + largest iteration). See [captureset.py](captureset.py).

## Metadata: `window.json`

STG normalizes time to [0,1]; **neither the model nor the COLMAP dataset
carries absolute time**, so "seconds per loop" must come from outside. The
player looks, in order:

    1. the command line (temporary override)
    2. window.json next to the PLY
    3. two levels above the PLY (the model root above point_cloud/iteration_N/)
    4. the colmap_N root

Only if none is found is a default used, labelled **ASSUMED** in the panel -
then playback speed does not match the recording.

`window.json` is written by `prep_4d.py` (stg-studio). The player reads
`duration_s` / `background` / `unit_mm` / **`win`**.

**`win` (window corners) is read directly, not computed here**: computing it
twice (player and preprocessing) used an algorithm (angle sort around the
centroid) that **is only right when every camera lies on the convex hull** - a
fifth camera in the centre would cave the frame in. Fix it in one place.
Without `win` a **nominal frame** (`player_core.FALLBACK_WIN_MM`) is used and
the panel says NOMINAL. `ctr` / `fwd` are still derived from colmap (they have
no ordering issue).

**Run it with the repository's virtual environment** (the CUDA rasterizer is
installed there). The working directory does not matter - `render_at.py`
finds SpacetimeGaussians by itself (`STG_HOME` overrides it).

## Controls

    W / S      forward / back
    A / D      left / right
    Q / E      down / up
    mouse drag look around (left button held)
    N / M      equivalent focal length: wide / tele (15-80 mm full-frame, including the camera's own)
    space      pause / continue
    R          back to the window centre; focal length back to native
    - / =      slower / faster playback (also , . and [ ])
    0          playback speed back to x1.0
    B          toggle background (black/white) - differences show transparent regions
    J          mute
    K          toggle world-locked / plain stereo - compare while moving left and right
    ; / '      volume
    F          toggle the window frame
    H          show / hide the information panel
    G          toggle fullscreen
    ESC        quit

## Coordinate principle

Window = rig = origin. **The viewpoint cannot pass through the wall** (the
rig plane); it can only move on this side of it - the physical constraint of
"looking through a window".
"""
from __future__ import annotations

import math
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.argv_bak = sys.argv
sys.argv = ["x"]
import render_at as R                                      # noqa: E402
sys.argv = sys.argv_bak

# Audio is **optional**. The module itself never raises (see audio4d.py),
#    but even its import can fail, so it is guarded too.
import hudwall as HW                                     # noqa: E402
# Scene loading, window geometry and viewpoint limits come from the shared
#    base (VR uses the same one). The audio and captureset imports and
#    RIG_MAX_MM_FALLBACK moved there - that fallback used to exist once in
#    flat and once in VR, and it changes with every recalibration.
from player_core import Scene, rot                        # noqa: E402

# Colours of the information panel (RGB; converted to BGR when drawn).
# It sits on the fogged wall, so thin dark text; titles in yellow.
PANEL_TITLE = (200, 160, 0)
PANEL_LABEL = (70, 70, 70)
PANEL_VALUE = (20, 20, 20)

LOOK_SENS = 0.22            # mouse sensitivity (degrees per pixel)
# Start 750 mm behind the window.
#    150 mm is too close (the frame nearly fills the picture, no wall
#    visible); 1 m is too far (the rig is only 153x92 mm, at 1 m the window is
#    8% of the picture width).
START_BACK_MM = 750.0
# The wall must be opaque enough. If translucent, **extrapolation artifacts
#    outside the convex hull show through** - at 1 m, radial coloured streaks
#    over the whole picture: not the scene, but angles the model was never
#    trained on. The wall's job is to cover the region without guarantees.
FOG = 0.87                  # fog strength outside the window
WALL_GRAY = 208             # wall colour (the frame colour is hudwall.FRAME_EDGE)
FRAME_BGR = (200, 120, 40)  # frame colour
FRAME_W = 12
FF_W = 36.0                       # full-frame sensor width, for focal length <-> horizontal FoV
# Equivalent focal length steps (mm, full-frame). Common prime focal lengths,
# not equal steps - the visual change of focal length is logarithmic: 15->18
# differs far more than 70->80. The camera's **native focal length** is
# inserted into this list at start-up and is the default.
LENS_STEPS = [15.0, 18.0, 21.0, 24.0, 28.0, 35.0, 50.0, 80.0]


# pollKey (non-blocking event polling) exists since OpenCV 4.5.4
_POLL = hasattr(cv2, "pollKey")


def letterbox(fr, win):
    """Pad with black to the window's aspect ratio, **never stretch**.

    `WINDOW_NORMAL` scales the image to the window size, so whenever the
    aspect ratios differ (fullscreen, or the window dragged to another shape)
    the picture is distorted. Padding the image to the window's ratio first
    makes the scaling uniform. Returned unchanged when the ratios match.
    """
    try:
        _, _, ww, wh = cv2.getWindowImageRect(win)
    except cv2.error:
        return fr
    H, W = fr.shape[:2]
    if ww <= 0 or wh <= 0 or abs(ww * H - wh * W) <= max(ww, wh):
        return fr
    if ww * H > wh * W:                 # window wider -> pad left and right
        pad = round(H * ww / wh) - W
        return cv2.copyMakeBorder(fr, 0, 0, pad // 2, pad - pad // 2,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pad = round(W * wh / ww) - H        # window taller -> pad top and bottom
    return cv2.copyMakeBorder(fr, pad // 2, pad - pad // 2, 0, 0,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


class Viewer(Scene):
    def __init__(self, ply, root, secs=None, scale=1.0):
        self.nsteps = len([d for d in os.listdir(root) if d.startswith("colmap_")])
        # Scene geometry, metric scale, model, audio and playback state live in the shared base (VR uses it too).
        super().__init__(ply, root, secs=secs, start_back_mm=START_BACK_MM)
        print(f"  {self.nsteps} time steps")
        # `scale` is **the render resolution factor**, 1.0 = training
        #    resolution (best quality). `camera_from_colmap` takes a divisor,
        #    hence 1/scale. Use --scale 0.5 for frame rate.
        self.proto = R.camera_from_colmap(self.m0, self.names[0],
                                          res=1.0 / float(scale))

        # Equivalent focal length (full-frame). The rig camera's **native focal
        # length** is inserted into the steps and is the default, so stepping
        # with N/M always passes "the camera's real field of view".
        self.flen_native = round((FF_W / 2) / math.tan(self.proto["FoVx"] / 2), 1)
        self.lens = sorted(set(LENS_STEPS) | {self.flen_native})
        self.li = self.lens.index(self.flen_native)
        print(f"  focal length steps {'/'.join(f'{v:g}' for v in self.lens)} mm"
              f" (N wide / M tele, default {self.flen_native:g} mm = camera native)")

        self.yaw = self.pitch = 0.0
        self.show_frame = True
        self.show_panel = True
        self.fps = 0.0
        self.drag, self.last = False, (0, 0)

    # ------------------------------------------------------------ camera
    def R_now(self):
        return rot("x", self.pitch) @ rot("y", self.yaw) @ self.Rc0

    def move(self, dx, dy, dz):
        super().move(self.R_now(), dx, dy, dz)

    # ------------------------------------------------------------ drawing
    def window_quad(self, Rn, cam):
        """Window corners projected into the image. None if any corner is behind the camera.

        `wincorners` is **never None** - without `win` in `window.json` the
        nominal frame is used (see `player_core.Scene._load_window`), because
        the wall carries the key help and must always be drawn.
        """
        W, H = cam["W"], cam["H"]
        fx = (W / 2) / math.tan(cam["FoVx"] / 2)
        fy = (H / 2) / math.tan(cam["FoVy"] / 2)
        q = []
        for p in self.wincorners:
            v = Rn @ (p - self.pos)
            if v[2] <= 1e-3:
                return None
            q.append((W / 2 + fx * v[0] / v[2], H / 2 + fy * v[1] / v[2]))
        return q

    def draw(self):
        Rn = self.R_now()
        cam = dict(self.proto)
        cam["R"] = np.transpose(Rn)
        cam["T"] = -Rn @ self.pos
        W, H = cam["W"], cam["H"]

        # N/M change the focal length: converting to a field of view **changes
        #    only FoVx; FoVy follows the picture's aspect ratio**. Converting
        #    both axes from 36 mm squashes the picture - the rasterizer takes
        #    tanfovx/tanfovy, whose ratio is the pixel aspect ratio.
        cam["FoVx"] = 2.0 * math.atan((FF_W / 2) / self.lens[self.li])
        cam["FoVy"] = 2.0 * math.atan(math.tan(cam["FoVx"] / 2) * H / W)

        q = self.window_quad(Rn, cam)

        # Fading and "do not render when turned away" were removed. Both were
        # protections from the time of CUDA crashes, whose real cause was NaN
        # from the hand-written quaternion normalization in `render_at.py`,
        # which is fixed. Every angle is rendered directly now.
        with torch.no_grad():
            im = R.render(self.g, cam, t=self.t, bg=self.bg,
                          cull_opacity=self.cull,
                          segments=self.segments, seg_ramp=self.seg_ramp)
        self.n_sent = R.LAST_SENT
        # The transfer goes through render_at.to_uint8 (shared with VR).
        #    `.cpu().numpy() * 255 .astype(uint8)` here transferred float32
        #    and did element-wise work in numpy: 14.5 ms, more than half a
        #    frame. Converting on the GPU first: 1.0 ms, bit-identical.
        fr = R.to_uint8(im, bgr=True)

        # With any frame corner behind the camera q is None and there is no
        # polygon - then only **the frame is not drawn**; the scene still is.
        if self.show_frame and q is not None:
            fx = (W / 2) / math.tan(cam["FoVx"] / 2)
            fy = (H / 2) / math.tan(cam["FoVy"] / 2)
            # The frame and the wall mask use **the same rounded path**, so the edges match.
            poly, tw = HW.window_polygon(self.wincorners, self.pos, Rn,
                                         W, H, fx, fy, self.u)
            if poly is not None:
                # The wall is composited by the shared hudwall.draw_wall (VR uses it too).
                fr = HW.draw_wall(fr, poly, FOG, WALL_GRAY)
                HW.draw_wall_text(fr, HW.FLAT_KEYS, self.wincorners, self.pos, Rn,
                                  W, H, fx, fy, self.u, bgr=True,
                                  wall_basis=(self.Rc0[0], self.Rc0[1]))
                # No "4DGS WINDOW" title on a nominal frame - the title
                #    identifies "this is the window", and a nominal frame is
                #    not. The key help is still drawn (it is why the wall exists).
                if not self.win_nominal:
                    HW.draw_wall_title(fr, self.wincorners, self.pos, Rn,
                                       W, H, fx, fy, self.u, bgr=True,
                                       wall_basis=(self.Rc0[0], self.Rc0[1]))
                # a nominal frame is black, visibly not computed from this recording
                HW.draw_frame(fr, poly, tw, bgr=True,
                              col=HW.FRAME_NOMINAL if self.win_nominal else None)

        return self._hud(fr, Rn)

    # Panel text **must be ASCII** - cv2.putText cannot draw CJK characters.
    # The key list is drawn **on the wall** (hudwall.FLAT_KEYS), not in the
    #    panel; two copies would drift apart and a long panel covers the picture.

    def _hud(self, fr, Rn=None):
        if Rn is not None:
            self._arrow(fr, Rn)
        if not self.show_panel:
            return fr

        H, W = fr.shape[:2]
        # The font size follows the picture and the width follows **the
        #    content**. A fixed "196 px wide, size 0.4" at 1746x982 gave tiny
        #    text and cut the longest row at the right edge. The 620 px
        #    reference is the old window height, so small pictures look as before.
        scale = max(1.0, H / 620.0)

        d = float(np.dot(self.pos - self.ctr, self.fwd0)) * self.u
        pct = self.n_sent / max(self.n_gauss, 1) * 100
        rows = [("SOURCE", None),
                ("shot", self.shot),
                ("data", self.dataset),
                ("model", self.model),
                (None, None),
                ("STATUS", None),
                ("time", f"{self.t * self.secs:5.2f}s / {self.secs:.2f}s"
                         f"{'' if self.secs_known else '  ASSUMED'}"),
                ("state", "PLAY" if self.playing else "PAUSE"),
                ("speed", f"x{self.speed:.1f}"),
                ("lens", f"{self.lens[self.li]:g} mm"),
                ("bg", f"{self.bg:.2f}" + ("" if self.bg_known else "  ASSUMED")),
                ("audio", self.audio.hud()),
                ("wall", f"{-d:.0f} mm"),
                ("yaw", f"{self.yaw:+.0f}"),
                ("pitch", f"{self.pitch:+.0f}"),
                ("fps", f"{self.fps:.1f}"),
                # Split into two rows. As one row it was the longest and
                #    alone decided the width of the whole panel.
                ("cull", "OFF" if self.cull is None else f"{self.cull:g}  {pct:.0f}%"),
                ("sent", f"{self.n_sent:,} / {self.n_gauss:,}"),
                ]
        # A nominal frame must be stated on screen, or it looks real.
        #    One line only - the full explanation (size, how to fix) is
        #    printed on the console at start-up.
        if self.win_nominal:
            rows += [("window", "NOMINAL (no window.json)")]
        # The panel shows only "the current state" - what is not on the wall
        #    and keeps changing. Its layout is the shared hudwall.draw_panel,
        #    the same as the VR desktop panel.
        pad = int(18 * scale)
        pw, _ = HW.panel_size(rows, scale)
        # In flat mode the panel sits on **the fogged wall** (light grey), so
        #    thin dark text without outline is enough. The VR desktop panel is
        #    on near-black and keeps bright text with an outline.
        #    With the wall off (`F`) the text sits on the raw picture and is hard to read in dark scenes.
        HW.draw_panel(fr, rows, x0=max(0, W - pw - pad), y0=int(22 * scale),
                      scale=scale, bgr=True, outline=False,
                      title_col=PANEL_TITLE, label_col=PANEL_LABEL,
                      value_col=PANEL_VALUE)
        return fr

    def _arrow(self, fr, Rn, cx=52, cy=64, L=30):
        """Red pointer in the top left, pointing to the window centre - to find it again after looking away."""
        v = Rn @ (self.ctr - self.pos)
        if v[2] > 1e-3:
            dx, dy = float(v[0]), float(v[1])
        else:                                    # window behind: point sideways
            dx, dy = (1.0 if v[0] >= 0 else -1.0), 0.0
        n = math.hypot(dx, dy)
        if n < 1e-6:
            cv2.circle(fr, (cx, cy), 6, (0, 0, 220), 2, cv2.LINE_AA)   # centred
            return
        dx, dy = dx / n, dy / n
        cv2.circle(fr, (cx, cy), L + 8, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.arrowedLine(fr, (cx - int(dx * L * 0.6), cy - int(dy * L * 0.6)),
                        (cx + int(dx * L), cy + int(dy * L)),
                        (0, 0, 220), 3, cv2.LINE_AA, tipLength=0.45)

    # ------------------------------------------------------------ main loop
    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag, self.last = True, (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drag = False
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.yaw -= (x - self.last[0]) * LOOK_SENS
            self.pitch = max(-60, min(60, self.pitch + (y - self.last[1]) * LOOK_SENS))
            self.last = (x, y)
        # Do not use cv2.EVENT_MOUSEWHEEL: `cv2.getMouseWheelDelta` **does not
        #    exist in the Python binding** (C++ only); calling it raises
        #    AttributeError, and OpenCV **silently swallows exceptions in mouse
        #    callbacks** - the wheel simply does nothing, with no error message.
        #    Focal length uses the N / M keys instead.

    # `_say_cull` moved into player_core.Scene.say_cull (VR uses the same)
    _say_cull = Scene.say_cull

    def render_once(self, t, dpos_mm=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0):
        """Render **one frame** and return it, without a window, audio or advancing time.

        For regression tests: renders before and after a change must be
        bit-identical, which cannot be tested with window screenshots.

        **Read-only**: `t / pos / yaw / pitch` are saved on entry and restored
        afterwards - a test must not leave the player in a different state.
        Whoever sets a transient state is responsible for restoring it.

        `dpos_mm` is the offset from the default viewpoint (mm), `t` the time 0..1.
        """
        keep = (self.t, tuple(self.pos), self.yaw, self.pitch)
        try:
            self.t = float(t) % 1.0
            self.yaw, self.pitch = float(yaw), float(pitch)
            base = self.ctr - self.fwd0 * (self.start_back_mm / self.u)
            self.pos = self.clamp(base + np.array(dpos_mm, float) / self.u)
            return self.draw()
        finally:
            self.t, p, self.yaw, self.pitch = keep
            self.pos = np.array(p, float)

    def run(self):
        win = (f"4DGS  {self.shot}  |  {self.model}  |  "
               f"WASD/QE move, drag look, N/M lens, SPACE pause, R reset, G fullscreen, ESC quit")
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Open the window at the **data's aspect ratio**. A fixed 1280x720
        #    (16:9) made `WINDOW_NORMAL` stretch 4:3 data 33% wider.
        cv2.resizeWindow(win, 1280, round(1280 * self.proto["H"] / self.proto["W"]))
        full = False
        cv2.setMouseCallback(win, self.on_mouse)
        prev, fps = time.time(), 0.0
        while True:
            # **Closing the window with its X must also quit**, and this check
            #    must come **before** `imshow` - `imshow` recreates a closed
            #    window, so checking afterwards always sees it "still there".
            #    Without the window `getWindowProperty` returns <1 or raises; both count as closed.
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            fr = self.draw()
            now = time.time()
            dt = now - prev
            prev = now
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-3)
            cv2.imshow(win, letterbox(fr, win))

            # **Audio is the master clock; the picture asks it where playback
            #    is** (as VLC / mpv do). The other way round, audio would chase
            #    every 10.7 ms something that only advances every 26-100 ms,
            #    and the playhead rate would jitter - heard as a "stepped,
            #    electronic" graininess.
            self.audio.update(self.speed, self.playing,
                              pos=self.pos, R=self.R_now())
            if self.playing:
                if self.audio.ok:
                    self.t = self.audio.now_t()
                else:
                    # Without audio (or after it failed) the wall clock is used.
                    # Continue **from the current self.t**, not from 0 - or
                    #    time would jump back to the start at the switch.
                    # The denominator is the real clip length; speed x1.0 = recording speed.
                    self.t = (self.t + dt * self.speed / self.secs) % 1.0

            # This used to be `cv2.waitKey(1)`, which **actually waits
            #    15.6 ms**, not 1 ms - the default Windows timer tick, to which
            #    HighGUI's wait is aligned. However fast drawing was, a frame
            #    took at least 15.6 ms, locking the player near 30 fps.
            #    `pollKey()` does not wait (measured 0.00 ms) and still handles events.
            #    Older OpenCV without pollKey falls back to waitKey(1), only slower.
            k = (cv2.pollKey() if _POLL else cv2.waitKey(1)) & 0xFF
            if k == 27:
                break
            elif k in (ord("w"), ord("W")):
                self.move(0, 0, 1)
            elif k in (ord("s"), ord("S")):
                self.move(0, 0, -1)
            elif k in (ord("a"), ord("A")):
                self.move(-1, 0, 0)
            elif k in (ord("d"), ord("D")):
                self.move(1, 0, 0)
            elif k in (ord("q"), ord("Q")):
                self.move(0, 1, 0)
            elif k in (ord("e"), ord("E")):
                self.move(0, -1, 0)
            elif k == ord(" "):
                self.playing = not self.playing
            elif k in (ord("r"), ord("R")):
                self.home()
                self.yaw = self.pitch = 0.0
                self.li = self.lens.index(self.flen_native)
            # Playback speed is bound to several keys. [ ] alone are awkward
            #    and some input methods replace them. - = is the most
            #    intuitive pair, , . suits video-editing habits, [ ] are kept.
            #    0 returns to normal speed.
            elif k in (ord("-"), ord("_"), ord("["), ord(",")):
                self.speed = max(0.1, self.speed / 1.3)
            elif k in (ord("="), ord("+"), ord("]"), ord(".")):
                self.speed = min(8.0, self.speed * 1.3)
            elif k == ord("0"):
                self.speed = 1.0
            elif k in (ord("n"), ord("N")):          # wide (shorter focal length)
                self.li = max(0, self.li - 1)
            elif k in (ord("m"), ord("M")):          # tele (longer focal length)
                self.li = min(len(self.lens) - 1, self.li + 1)
            elif k in (ord("f"), ord("F")):
                self.show_frame = not self.show_frame
            elif k in (ord("h"), ord("H")):
                self.show_panel = not self.show_panel
            elif k in (ord("t"), ord("T")):
                # Diagnostic: toggle time-window culling. Both states **should
                #    look the same**; only speed changes. If a difference is
                #    visible the threshold is too high (lowering it does not help).
                self.cull = None if self.cull is not None else R.CULL_DEFAULT
                self._say_cull()
            elif k in (ord("5"), ord("6"), ord("4")):
                # Threshold: 5 smaller (slower, more Gaussians) / 6 larger (faster) / 4 default.
                # Left to right on the keyboard: left smaller, right larger.
                # The lower limit is the default 0.005 (`cull_step`'s lower
                #    bound); below it only gets slower with no visible quality
                #    gain. VR follows the same rule.
                # The logic is render_at.cull_step, shared with VR.
                before = self.cull
                self.cull = (R.CULL_DEFAULT if k == ord("4")
                             else R.cull_step(self.cull, k == ord("6")))
                if k == ord("5") and self.cull == before:
                    print(f"  time culling is already at the lower limit {R.CULL_DEFAULT:g}"
                          f" (lower is only slower, not better; press T to compare both ends)",
                          flush=True)
                else:
                    self._say_cull()
            elif k in (ord("b"), ord("B")):
                # Diagnostic: toggle black/white background. The largest differences are regions left transparent in training
                self.bg = 1.0 if self.bg < 0.5 else 0.0
            elif k in (ord("j"), ord("J")):
                self.audio.toggle_mute()
            elif k in (ord("k"), ord("K")):
                # Diagnostic: toggle "world-locked" and "plain stereo".
                #    Compare while moving left and right in front of the
                #    window; the difference is what spatialization does.
                self.audio.toggle_spatial()
            elif k in (ord(";"), ord(":")):
                self.audio.vol_step(1 / 1.3)
            elif k == ord("'"):
                self.audio.vol_step(1.3)
            elif k in (ord("g"), ord("G")):
                full = not full
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full
                                      else cv2.WINDOW_NORMAL)
        cv2.destroyAllWindows()


# The command line is in `player.py` - this file only provides the Viewer class (flat mode).
if __name__ == "__main__":
    raise SystemExit(
        "  ERROR the players are combined into one program: use player.py   "
        "python player.py <shooting root> (add --vr for VR)")
