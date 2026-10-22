#!/usr/bin/env python3
"""vr_viewer.py - 4DGS VR player (SteamVR / OpenVR)

    python player.py <shooting root> --vr [--model H] [--iter 30000] [--scale 1.0]   <- the entry point
    python player.py <STG PLY> <colmap_N root> --vr [--scale 0.5]                   <- older form

Everything produced by one shoot lives in **one root folder**; playback only
needs that folder. Which model/iteration: `player.json` > command line >
automatic. See captureset.py.

**Run it with the repository's virtual environment** (the CUDA rasterizer is
installed there). The working directory does not matter - `render_at.py`
finds SpacetimeGaussians by itself (`STG_HOME` overrides it).
Hardware used: Quest 2 + Link -> SteamVR. Start SteamVR before the player.

## Why this exists: **checking geometry**

Every number from the flat player is a **photometric error** (PSNR).
Gaussians can sit at the wrong depth and still project correctly into the
training cameras, with the same high PSNR.

**VR is the first thing that honestly reflects geometry** - binocular
disparity follows directly from depth; wrong depth is uncomfortable, and
noticeable immediately.

It is also **the first strict test of the metric scale**: with `unit_mm` off
by 3%, the whole world is 3% larger, which feels "odd" in a way that is hard
to name.

## Controls (as in the flat player)

    head        rotation and small movements tracked by the HMD (movement is limited to the window)
    W/S A/D Q/E move the window origin (large moves by keyboard, to avoid walking into real walls)
    space       pause / continue
    - / = / 0   playback speed
    R           back to the window centre
    C           recentre at the current head position (after standing up / sitting down)
    F           frame and wall: frame + wall -> frame only -> off
    T           toggle time-window culling - **look for flicker with both eyes**
    4 / 5 / 6   culling threshold: default / smaller / larger   the lower limit is 0.005
    J           mute
    K           toggle world-locked / plain stereo - **compare while turning the head**
    ; / '       volume
    ESC         quit

Keyboard events come from **the mirror window on the desktop**, so that window
must keep the focus.

## Known technical limitations

### 1. Asymmetric frustum -> render larger, then crop

Each VR eye has an **off-axis** frustum (left eye shifted left, right eye
right). `diff_gaussian_rasterization_ch3` only takes `tanfovx`/`tanfovy` and
**can only do symmetric frustums** (the same limitation as "the principal
point must be centred").

Workaround: compute **a symmetric frustum enclosing the asymmetric one**,
render it, and crop the correct sub-region. The cost is 20-40% more pixels.

> **gsplat takes a full intrinsics matrix `Ks[3,3]`; switching to it would avoid this.**

### 2. Low frame rate

Quest 2 needs 72-90 Hz, while one eye at 912x512 runs at about 20-30 fps.
Two eyes at VR resolution -> **possibly only 5-15 fps**.

Time-window culling (`T`, on by default) was measured 4.7x faster in flat
mode. The gain in VR should be larger (two eyes x 20-40% extra pixels from
the asymmetric frustum) but has not been measured.

**This is for "standing still and checking whether the geometry is right",
not a comfort test.** Use `--scale` to trade resolution for frame rate.
SteamVR's reprojection fills in frames, but from a low source rate it smears
- an artifact of reprojection, not of the model; **do not judge quality by it**.

### 3. Head movement is limited to the window

Window = rig = origin. Leaning forward is stopped `WALL_MARGIN_MM` in front
of the wall. **A hard stop is uncomfortable in VR**; how to handle the window
boundary in VR is an open question.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

# All imports are either "modules in this folder" or "installed packages";
#    no sys.path manipulation is needed.
sys.argv_bak = sys.argv
sys.argv = ["x"]
import render_at as R                                      # noqa: E402
sys.argv = sys.argv_bak

import hudwall as HW                                     # noqa: E402
# Scene loading, window geometry and viewpoint limits come from the shared
#    base (the flat player uses it too). The audio and captureset imports,
#    `find_meta` / `window_frame` / `clamp` and RIG_MAX_MM_FALLBACK are
#    there - they used to exist once in each player.
from player_core import Scene                             # noqa: E402

import glfw                                                # noqa: E402
import openvr                                              # noqa: E402
import torch                                               # noqa: E402
from OpenGL.GL import *                                    # noqa: E402,F403

# Start 120 mm behind the window.
#
#    **VR and the flat player use different values on purpose.**
#       Flat keeps 750 mm: on a screen the whole window must stay in the
#       picture. In VR the face is at the window looking out - from further
#       back the window is a small part of the view and the "window" feeling
#       is lost. Being close is the point of the product.
#
#    The forward limit is enforced by clamp() and WALL_MARGIN_MM (20 mm in
#    front of the wall); to step back use S, or R to return to this distance.
START_BACK_MM = 120.0


def mat34(m):
    """OpenVR HmdMatrix34_t -> 4x4 numpy."""
    a = np.eye(4)
    for i in range(3):
        for j in range(4):
            a[i, j] = m[i][j]
    return a


class VRViewer(Scene):
    def __init__(self, ply, root, scale=1.0, secs=None):
        # Scene geometry, metric scale, model, audio and playback state live in the shared base.
        # `require_metric=True` is VR-only: head movement is in real metres,
        #    and with wrong world units one step flies far away and is
        #    uncomfortable. The flat player only warns and disables metric
        #    features; VR refuses to play.
        super().__init__(ply, root, secs=secs, start_back_mm=START_BACK_MM,
                         require_metric=True)
        self.frame_mode = 2          # 0 off / 1 frame only / 2 frame + wall
        # Time-window culling (T toggles). VR gains the most - each eye is
        #    rendered separately, and the asymmetric-frustum workaround adds
        #    20-40%, about 2.5x the flat cost. On by default: flat mode showed
        #    no visible difference, but **flicker in stereo must be checked
        #    separately**; one eye cannot show it.
        #    The threshold's **lower limit is the default**; below it an int32
        #    overflow crashes the rasterizer (measured in VR).
        # one persistent upload buffer per eye (RGB, see the end of render_eye)
        self._upload = {}
        self.scale = scale
        # OpenVR defaults to the standing space, **Y=0 at the floor** - the
        #    head is at about (0, 1.7, 0). Adding that to the scene directly:
        #    1.7 m x (1000/unit_mm) ~ 93 scene units, while the whole rig spans
        #    only 10 -> the viewpoint lands 9 rig-lengths above the window.
        #    So the first frame's head position becomes the origin and only
        #    **relative movement** is used.
        self.head0 = None

        # -- OpenVR
        self.vr = openvr.init(openvr.VRApplication_Scene)
        w, h = self.vr.getRecommendedRenderTargetSize()
        self.W, self.H = int(w * scale), int(h * scale)
        print(f"  HMD recommended resolution {w}x{h} -> rendering {self.W}x{self.H}"
              f" (--scale {scale})")

        # per eye: the four tangents of the off-axis frustum + eye offset from the head
        self.eye = {}
        for e, name in ((openvr.Eye_Left, "L"), (openvr.Eye_Right, "R")):
            l, r, t, b = self.vr.getProjectionRaw(e)
            e2h = mat34(self.vr.getEyeToHeadTransform(e))
            # OpenVR does not document the sign convention of top/bottom
            #    (implementations differ), so **nothing is assumed** - min/max
            #    covers both conventions, and the actual values are printed for checking.
            #    Y is negated: OpenVR +Y is up, COLMAP camera +Y is down.
            xlo, xhi = min(l, r), max(l, r)
            ylo, yhi = min(-t, -b), max(-t, -b)          # already in COLMAP's +Y down
            tx, ty = max(abs(xlo), abs(xhi)), max(abs(ylo), abs(yhi))
            self.eye[e] = dict(raw=(l, r, t, b), off=e2h[:3, 3], name=name,
                               frust=(xlo, xhi, ylo, yhi, tx, ty))
            over = (2 * tx) * (2 * ty) / ((xhi - xlo) * (yhi - ylo))
            print(f"    {name} eye  raw(l,r,t,b)=({l:+.3f},{r:+.3f},{t:+.3f},{b:+.3f})")
            print(f"          -> COLMAP frustum x[{xlo:+.3f},{xhi:+.3f}] "
                  f"y[{ylo:+.3f},{yhi:+.3f}]   symmetric envelope renders {(over-1)*100:.0f}% more")
            print(f"          eye offset {e2h[0,3]*1000:+.1f} mm")
        ipd = abs(self.eye[openvr.Eye_Left]["off"][0]
                  - self.eye[openvr.Eye_Right]["off"][0]) * 1000
        print(f"    IPD {ipd:.1f} mm  -> scene units {ipd/self.u:.4f}")

        self.comp = openvr.VRCompositor()
        self.F = np.diag([1.0, -1.0, -1.0])
        # There used to be a "self-check" here: F I F Rc0 == Rc0 for Hr = I.
        #    That check **proves nothing** - F^2 = I holds for any involutory
        #    matrix, so the assert passes whatever F is. Only looking through
        #    the headset can verify it:
        #      upside down -> texture coordinates (fixed: flipped before upload)
        #      left/right or depth reversed -> then F or A is wrong
        self._gl_init()

    # -------------------------------------------------- OpenGL
    def _gl_init(self):
        if not glfw.init():
            raise SystemExit("glfw initialization failed")
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.win = glfw.create_window(720, 405, "4DGS VR - keep keyboard focus here", None, None)
        glfw.make_context_current(self.win)
        # Turn off vsync for the desktop window.
        #    With it on, `swap_buffers` **blocks until the next vertical
        #    retrace**, quantizing frame time to multiples of 16.7 ms (a 62 ms
        #    frame becomes 66.7). The VR pace is set by `comp.waitGetPoses`
        #    (the headset compositor); the desktop window only shows
        #    information and receives keys and does not need to sync.
        #    The same family of problem as `cv2.waitKey(1)` waiting 15.6 ms in
        #       the flat player: "looks like polling, actually waits".
        glfw.swap_interval(0)
        glfw.set_key_callback(self.win, self._key)
        self.tex = {}
        for e in self.eye:
            t = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, t)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.W, self.H, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, None)
            self.tex[e] = t

    # Redrawing the information panel takes 3.5 ms (a dozen putText calls +
    # a flip), and it is for people standing nearby - two or three updates a
    # second are enough. Only **the texture** is throttled; the quad is drawn every frame.
    PANEL_EVERY = 6

    def _draw_mirror(self, fps):
        """The desktop window shows **an information panel**, not the VR picture.

        It was originally only there to receive keyboard events and was
           black. Mirroring the VR picture is pointless - the person in the
           headset already sees it, and people nearby need **the state** (fps,
           time, culling...). The layout is shared with the flat player's
           panel via `hudwall.draw_panel`.
        """
        try:
            import cv2
        except ImportError:
            return
        mw, mh = glfw.get_framebuffer_size(self.win)
        if mw <= 0 or mh <= 0:
            return
        if getattr(self, "_hud_tex", None) is None:
            self._hud_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._hud_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 512, 512, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, None)

        self._panel_n = getattr(self, "_panel_n", 0) + 1
        if self._panel_n % self.PANEL_EVERY == 1:
            self._panel_upload(fps)
        self._panel_blit(mw, mh)

    def _panel_upload(self, fps):
        import cv2  # noqa: F401  (needed by draw_panel)
        n = int(self.g["xyz"].shape[0])
        d = float(np.dot(self.pos - self.ctr, self.fwd0)) * self.u
        rows = [("SOURCE", None),
                ("shot", self.shot),
                ("model", self.model),
                (None, None),
                ("STATUS", None),
                ("time", f"{self.t * self.secs:5.2f}s / {self.secs:.2f}s"),
                ("state", "PLAY" if self.playing else "PAUSE"),
                ("speed", f"x{self.speed:.1f}"),
                ("wall", f"{-d:.0f} mm"),
                ("scale", f"{self.scale:.2f}  {self.W}x{self.H}"),
                ("frame", ("off", "frame", "frame+wall")[self.frame_mode]),
                ("audio", self.audio.hud() if hasattr(self.audio, "hud") else "-"),
                ("cull", "OFF" if self.cull is None else
                         f"{self.cull:g}  {self.n_sent:,} / {n:,}"
                         f"  ({self.n_sent * 100 // max(n, 1)}%)"),
                ("fps", f"{fps:.1f}"),
                ]
        # A nominal frame must be marked. In VR the frame is **the depth
        #    reference** (known distance and size let the eyes judge how far
        #    things outside are); with a hard-coded nominal size that judgement
        #    does not hold, and the user must know. One line only; the full
        #    explanation is printed on the console at start-up.
        if self.win_nominal:
            rows += [("window", "NOMINAL (no window.json)")]
        img = np.zeros((512, 512, 4), np.uint8)
        img[:, :, 3] = 255
        HW.draw_panel(img, rows, x0=16, y0=34, scale=1.35)
        # OpenGL textures start bottom-left, so flip before upload (as on the headset path)
        img = np.ascontiguousarray(img[::-1])

        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512,
                        GL_RGBA, GL_UNSIGNED_BYTE, img)

    def _panel_blit(self, mw, mh):
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glViewport(0, 0, mw, mh)
        glClearColor(0.07, 0.07, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        for tu, tv, vx, vy in ((0, 0, -1, -1), (1, 0, 1, -1),
                               (1, 1, 1, 1), (0, 1, -1, 1)):
            glTexCoord2f(tu, tv)
            glVertex2f(vx, vy)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    # `_say_cull` moved into player_core.Scene.say_cull (the flat player uses the same)
    _say_cull = Scene.say_cull

    def _key(self, win, key, sc, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        k = {glfw.KEY_W: (0, 0, 1), glfw.KEY_S: (0, 0, -1),
             glfw.KEY_A: (-1, 0, 0), glfw.KEY_D: (1, 0, 0),
             glfw.KEY_Q: (0, 1, 0), glfw.KEY_E: (0, -1, 0)}.get(key)
        if k:
            # VR uses `Rc0` (the window frame), not the head orientation - WASD
            #    directions do not turn with the head, or looking down and
            #    pressing W would dive into the floor.
            self.move(self.Rc0, *k)
        elif key == glfw.KEY_SPACE:
            self.playing = not self.playing
        elif key in (glfw.KEY_MINUS, glfw.KEY_COMMA, glfw.KEY_LEFT_BRACKET):
            self.speed = max(0.1, self.speed / 1.3)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_PERIOD, glfw.KEY_RIGHT_BRACKET):
            self.speed = min(8.0, self.speed * 1.3)
        elif key == glfw.KEY_0:
            self.speed = 1.0
        elif key == glfw.KEY_R:
            self.home()
        elif key in (glfw.KEY_C, glfw.KEY_H):     # recentre at the current head position
            self.head0 = None
        elif key == glfw.KEY_T:
            # Diagnostic: toggle time-window culling. **Look with both eyes** -
            #    flicker invisible to one eye is sometimes amplified in stereo.
            #    If it is visible, turn it off and report it.
            self.cull = None if self.cull is not None else R.CULL_DEFAULT
            self._say_cull()
        elif key in (glfw.KEY_5, glfw.KEY_6, glfw.KEY_4):
            # Threshold: 5 smaller / 6 larger / 4 default (left smaller, right larger).
            # **The lower limit is the default 0.005** (`cull_step`'s lower
            #    bound, the same in both players). In VR it is a safety issue:
            #    lowering it doubles the Gaussians sent, and at VR resolution
            #    (both eyes x symmetric envelope) that does not slow down but
            #    overflows the rasterizer's int32 and crashes.
            before = self.cull
            self.cull = (R.CULL_DEFAULT if key == glfw.KEY_4
                         else R.cull_step(self.cull, key == glfw.KEY_6))
            if key == glfw.KEY_5 and self.cull == before:
                print(f"  time culling is already at the lower limit {R.CULL_DEFAULT:g}"
                      f" (VR cannot go lower; it would overflow the rasterizer)", flush=True)
                return
            self._say_cull()
        elif key == glfw.KEY_F:
            self.frame_mode = (self.frame_mode + 1) % 3
        elif key == glfw.KEY_J:
            self.audio.toggle_mute()
        elif key == glfw.KEY_K:
            self.audio.toggle_spatial()
        elif key == glfw.KEY_SEMICOLON:
            self.audio.vol_step(1 / 1.3)
        elif key == glfw.KEY_APOSTROPHE:
            self.audio.vol_step(1.3)
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    # -- window frame and wall ------------------------------------------
    #
    # Both are far more useful in VR than in the flat player.
    #
    #    In the flat player the frame is a hint. In VR it is **the depth
    #    reference**: it sits on the real rig plane (known distance, known
    #    size of 153x92 mm), so both eyes can judge directly whether things
    #    outside are at a plausible distance. Without it, a skewed
    #    reconstruction only feels "odd", without saying why.
    #
    #    The wall (fog outside the window) marks **the viewable range**:
    #    window = rig = convex hull; outside it is extrapolation, with no quality guarantee.
    #
    # Drawn with the symmetric frustum's intrinsics, with the crop origin
    #    passed along, so no separate off-axis offset needs computing.
    FRAME_RGB = (40, 120, 200)      # VR images are RGB, not the flat player's BGR
    FRAME_W_PX = 10
    # Wall opacity. 1.0 = fully blocked; nothing outside the window visible.
    # Much higher in VR than in the flat player: on a screen the wall is a
    #    hint, in VR it must **read like a real wall** so the window boundary
    #    feels solid. A translucent wall becomes "fog floating in the air" in
    #    stereo and disturbs depth judgement.
    WALL_FOG = 0.82
    WALL_RGB = 205

    def _corners_px(self, Rn, eye_pos, BW, BH, tx, ty):
        """Window corners projected into the large image. None if any corner is behind the camera.

        `wincorners` is **never None** (a nominal frame without `win`); see
        `player_core.Scene._load_window`.
        """
        fx, fy = (BW / 2) / tx, (BH / 2) / ty
        out = []
        for p in self.wincorners:
            v = Rn @ (p - eye_pos)
            if v[2] <= 1e-3:
                return None
            out.append((BW / 2 + fx * v[0] / v[2], BH / 2 + fy * v[1] / v[2]))
        return np.array(out, np.int32).reshape(-1, 1, 2)

    def _overlay(self, a, Rn, eye_pos, BW, BH, tx, ty, ox, oy):
        """Overlay the frame and wall.

        **Drawn on the cropped sub-region**, not on the large symmetric
           envelope image. Overlaying the large image and cropping afterwards
           wasted the 18-40% that is thrown away. After cropping the principal
           point is no longer centred, so the crop origin (ox, oy) is passed to
           `window_polygon` / `draw_wall_text` - a pure geometric offset, no
           resampling; pixels inside the picture are identical.
        """
        if self.frame_mode == 0:
            return a
        try:
            import cv2
        except ImportError:
            return a
        fx, fy = (BW / 2) / tx, (BH / 2) / ty
        # The frame and wall mask use **the same rounded path**, so the edges match.
        poly, tw = HW.window_polygon(self.wincorners, eye_pos, Rn,
                                     BW, BH, fx, fy, self.u, ox=ox, oy=oy)
        if poly is None:
            return a
        if self.frame_mode == 2:
            # The wall is composited by the shared hudwall.draw_wall (the flat
            #    player uses it too). Separate copies once diverged, and VR
            #    **runs it twice per frame** (two eyes), doubling the cost of a missed change.
            a = HW.draw_wall(a, poly, self.WALL_FOG, self.WALL_RGB)
            HW.draw_wall_text(a, HW.VR_KEYS, self.wincorners, eye_pos, Rn,
                              BW, BH, fx, fy, self.u,
                              wall_basis=(self.Rc0[0], self.Rc0[1]),
                              ox=ox, oy=oy)
            # no "4DGS WINDOW" title on a nominal frame (see viewer.py)
            if not self.win_nominal:
                HW.draw_wall_title(a, self.wincorners, eye_pos, Rn,
                                   BW, BH, fx, fy, self.u,
                                   wall_basis=(self.Rc0[0], self.Rc0[1]),
                                   ox=ox, oy=oy)
        HW.draw_frame(a, poly, tw,
                      col=HW.FRAME_NOMINAL if self.win_nominal else None)
        return a

    def render_eye(self, e, head):
        """Render one eye.

        The rasterizer only does **symmetric** frustums, while VR's are
           **off-axis**. Method: take the largest tangents to build an
           enclosing symmetric frustum, render it larger, and crop the real
           sub-region.
        """
        xlo, xhi, ylo, yhi, tx, ty = self.eye[e]["frust"]
        # size of the enclosing symmetric frustum such that the crop is exactly W x H
        BW = int(round(self.W * (2 * tx) / (xhi - xlo)))
        BH = int(round(self.H * (2 * ty) / (yhi - ylo)))
        x0 = int(round((xlo + tx) / (2 * tx) * BW))
        y0 = int(round((ylo + ty) / (2 * ty) * BH))

        # Coordinate conversion - derived from both sides' **documented
        #   conventions**, not by trial and error
        #
        #   OpenVR: +X right, +Y up, -Z forward, right-handed, metres
        #   COLMAP camera: +X right, +Y down, +Z forward
        #   -> COLMAP-local <-> VR-local differ by F = diag(1,-1,-1) (F is its own inverse)
        #
        #   `getWorld2View2(R,t)` uses `R.transpose()` as world->camera, so
        #   `cam["R"]` holds **camera->world** and `cam["T"]` holds COLMAP's t = -R_w2c C.
        #
        #   The VR tracking space and the scene are **two different world
        #   frames** that must be aligned first:
        #       VR right (+X) -> scene Rc0[0] (the reference camera's right)
        #       VR up    (+Y) -> scene -Rc0[1] (camera y points down, so negated)
        #       VR back  (+Z) -> scene -Rc0[2] (camera z points forward, so negated)
        #   As a matrix: A = Rc0^T F (takes VR-world vectors into the scene world)
        #
        #   So camera->world = A Hr F, and world->camera R_w2c = F Hr^T F Rc0
        #   Check: with Hr = I (head still, facing forward) R_w2c = Rc0
        Hr = head[:3, :3]
        F = self.F
        Rn = F @ Hr.T @ F @ self.Rc0                    # world -> camera
        A = self.Rc0.T @ F                              # VR world -> scene world
        # head movement and eye offset, metres -> scene units
        k = 1000.0 / self.u
        hoff = head[:3, 3] - (self.head0 if self.head0 is not None else head[:3, 3])
        eye_pos = self.clamp(self.pos
                             + A @ hoff * k
                             + A @ (Hr @ self.eye[e]["off"]) * k)
        cam = dict(W=BW, H=BH,
                   FoVx=2 * math.atan(tx), FoVy=2 * math.atan(ty),
                   R=np.transpose(Rn), T=-Rn @ eye_pos)
        with torch.no_grad():
            im = R.render(self.g, cam, t=self.t, bg=self.bg,
                          cull_opacity=self.cull,
                          segments=self.segments, seg_ramp=self.seg_ramp)
        self.n_sent = R.LAST_SENT
        # **Crop the sub-region on the GPU first**, so every later transfer
        #    and overlay handles only that part. The 18-40% extra from the
        #    symmetric envelope is dropped here instead of travelling to the CPU.
        im = im[:, y0:y0 + self.H, x0:x0 + self.W]
        # The transfer goes through the shared render_at.to_uint8 (the flat
        #    player uses it too); one contiguous uint8 transfer. The former
        #    float32 readback + np.dstack into a new image + tobytes copy took
        #    47-56 ms per eye, and VR runs it twice per frame.
        a = R.to_uint8(im)
        if a.shape[0] != self.H or a.shape[1] != self.W:
            pad = np.zeros((self.H, self.W, 3), np.uint8)
            pad[:a.shape[0], :a.shape[1]] = a[:self.H, :self.W]
            a = pad
        # after cropping the top-left, the principal point is off-centre; the crop origin goes to the overlay.
        a = self._overlay(a, Rn, eye_pos, BW, BH, tx, ty, x0, y0)

        # Flip vertically into **a persistent three-channel buffer**.
        #
        #    OpenGL textures start bottom-left, numpy images top-left, hence
        #    the flip (a texture coordinate convention, not wrong camera geometry).
        #
        #    `np.dstack([sub[::-1], 255])` + `ascontiguousarray` allocated two
        #    new images and copied twice per eye per frame: 11.8 ms.
        #
        #    **Do not add alpha on the CPU.** Writing into a persistent RGBA
        #       buffer still took 7.8 ms because `buf[:, :, :3] = ...` is a
        #       strided destination. Uploading with `GL_RGB` lets OpenGL add
        #       alpha (guaranteed 1.0 by the specification), leaving one
        #       contiguous row copy: **7.8 -> 0.13 ms**.
        #
        #    The flip deliberately stays **after the overlay**. Moving it to
        #       the GPU (`to_uint8(flip_v=True)`) would mean overlaying an
        #       upside-down image, with `fy` negated and the text corners
        #       mirrored - saving 2 ms at the cost of another easy mistake in
        #       the geometry. The expensive parts are already gone above.
        buf = self._upload.get(e)
        if buf is None or buf.shape != a.shape:
            buf = np.empty(a.shape, np.uint8)
            self._upload[e] = buf
        buf[:] = a[::-1]
        return buf

    # -------------------------------------------------- main loop
    def listener(self, head):
        """Listener position and orientation (head centre, no eye offset). Same derivation as render_eye.

        Returns (scene position, world -> head rotation). `R[0]` is right and
        `R[2]` forward, exactly what audio4d._pan expects.
        """
        Hr = head[:3, :3]
        F = self.F
        Rn = F @ Hr.T @ F @ self.Rc0                    # world -> head
        A = self.Rc0.T @ F                              # VR world -> scene world
        k = 1000.0 / self.u
        hoff = head[:3, 3] - (self.head0 if self.head0 is not None else head[:3, 3])
        return self.clamp(self.pos + A @ hoff * k), Rn

    def run(self):
        poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        prev, fps, n = time.time(), 0.0, 0
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.comp.waitGetPoses(poses, None)
            hp = poses[openvr.k_unTrackedDeviceIndex_Hmd]
            head = mat34(hp.mDeviceToAbsoluteTracking) if hp.bPoseIsValid else np.eye(4)
            if self.head0 is None and hp.bPoseIsValid:
                self.head0 = head[:3, 3].copy()
                print(f"  current head position is the origin {np.round(self.head0, 3)} (C recentres)")
            for e in self.eye:
                img = self.render_eye(e, head)
                glBindTexture(GL_TEXTURE_2D, self.tex[e])
                # The numpy array is passed directly. `img.tobytes()` copied
                #    the whole picture again per eye per frame (9-16 MB at VR
                #    resolution), and PyOpenGL accepts contiguous arrays anyway.
                # Upload format GL_RGB into an RGBA8 texture - OpenGL fills
                #    alpha with 1.0 (guaranteed). Adding alpha on the CPU took 7.8 ms.
                #    Alignment 1: an RGB row is width x 3 bytes, not necessarily a multiple of 4.
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H,
                                GL_RGB, GL_UNSIGNED_BYTE, img)
                tex = openvr.Texture_t(handle=int(self.tex[e]),
                                       eType=openvr.TextureType_OpenGL,
                                       eColorSpace=openvr.ColorSpace_Gamma)
                self.comp.submit(e, tex)

            now = time.time()
            dt, prev = now - prev, now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
            # **Audio is the master clock; the picture asks it** (same as flat mode).
            # listener = head centre (no eye offset; ear spacing is handled in audio4d).
            lp, lR = self.listener(head)
            self.audio.update(self.speed, self.playing, pos=lp, R=lR)
            if self.playing:
                if self.audio.ok:
                    self.t = self.audio.now_t()
                else:
                    # without audio, the wall clock continues from the current self.t
                    self.t = (self.t + dt * self.speed / self.secs) % 1.0
            n += 1
            # render_at no longer flushes the allocator every frame (a hard
            # sync, 1.29x slower at VR resolution); it is flushed here every
            # 120 frames to avoid growth during long playback.
            if n % 120 == 0:
                torch.cuda.empty_cache()
            if n % 30 == 0:
                d = float(np.dot(self.pos - self.ctr, self.fwd0)) * self.u
                glfw.set_window_title(
                    self.win,
                    f"4DGS VR  {self.shot} | {self.model}   {fps:4.1f} fps   "
                    f"t={self.t*self.secs:5.2f}s/{self.secs:.2f}s   "
                    f"wall {-d:.0f}mm   x{self.speed:.1f}")
            self._draw_mirror(fps)
            glfw.swap_buffers(self.win)
        self.audio.stop()
        openvr.shutdown()
        glfw.terminate()


# The command line is in `player.py` - this file only provides the VRViewer class.
if __name__ == "__main__":
    raise SystemExit(
        "  ERROR the players are combined into one program: use player.py   "
        "python player.py <shooting root> --vr")
