#!/usr/bin/env python3
"""hudwall.py - draws the key help **on the wall plane** (shared by the flat and VR players)

## Why on the wall and not over the picture

Text overlaid on the picture (a HUD overlay) sits at **the same pixel in
both eyes**, with zero disparity - stereo vision places it **at infinity**.
But the wall is within a metre, so the eyes are asked to focus near and
converge at infinity at the same time. That vergence-accommodation conflict
is uncomfortable over time, and it disturbs judging "how far away is that
thing outside the window" - the very reason the VR player exists.

So the text is **a 3D object on the wall plane**: each line is anchored at a
world position on the rig plane and projected per eye. It has the correct
disparity and reads like a sheet of paper on the wall.

## Two columns; no alignment with spaces

Joining "key + description" into one string and right-aligning it gives
**ragged lines**: cv2's HERSHEY font is **proportional**, so spaces in
`"W/S A/D Q/E  move"` cannot form a fixed column, and each description
starts at a different x.

The right way is **two independently positioned columns**: the left edges of
the key column and the description column are fixed world positions on the
wall, and they line up after projection.

## ASCII only

`cv2.putText` cannot draw CJK characters; they come out as rows of question
marks. (Other scripts would need PIL plus a font file - another dependency
and a font to distribute.)
"""

from __future__ import annotations

import numpy as np

LINE_MM = 26.0          # line height on the wall (physical mm)
KEY_COL_MM = 0.0        # left edge of the key column (relative to the block's left edge)
DESC_COL_MM = 160.0     # left edge of the description column (wide enough for the longest key label)
BLOCK_W_MM = 400.0      # width of the whole block
GAP_MM = 45.0           # gap between the block and the window frame
TITLE_MM = 12.0         # extra space between the title and the first line

TEXT_RGB = (55, 55, 55)
TITLE_RGB = (20, 85, 160)


TEX_PPMM = 2.6          # texture resolution: pixels per mm on the wall
_TEX_CACHE = {}


def _build_texture(rows, title):
    """Draw the help into a texture (RGB + mask). Coordinates are wall mm x TEX_PPMM.

    This does not depend on the viewer at all - it is "what is written on
    the wall", drawn once.
    """
    import cv2
    key = (id(rows), title)
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]

    n_lines = len(rows) + 2
    Wt = int(BLOCK_W_MM * TEX_PPMM)
    Ht = int(LINE_MM * n_lines * TEX_PPMM)
    tex = np.zeros((Ht, Wt, 3), np.uint8)
    msk = np.zeros((Ht, Wt), np.uint8)

    scale = LINE_MM * TEX_PPMM / 30.0        # HERSHEY scale=1 is about 30 px line height
    th = max(1, int(round(scale * 1.7)))

    def put(x_mm, line_i, s, col):
        org = (int(x_mm * TEX_PPMM), int((line_i + 1) * LINE_MM * TEX_PPMM))
        cv2.putText(tex, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th,
                    cv2.LINE_AA)
        cv2.putText(msk, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, th,
                    cv2.LINE_AA)

    put(0, 0, title, TITLE_RGB)
    for i, (k, d) in enumerate(rows, start=2):
        put(KEY_COL_MM, i, k, TEXT_RGB)
        put(DESC_COL_MM, i, d, TEXT_RGB)

    _TEX_CACHE[key] = (tex, msk, BLOCK_W_MM, LINE_MM * n_lines)
    return _TEX_CACHE[key]


def draw_wall_text(img, rows, corners, pos, Rn, W, H, fx, fy, unit_mm,
                   bgr=False, title="CONTROLS", wall_basis=None,
                   ox=0, oy=0):
    """**Paste the help onto the wall** - with a perspective warp, not putText on the picture.

    ## Why putText alone does not work

    `cv2.putText` draws **screen-space** text: always horizontal, always
    upright, always the same shape. Projecting only each line's **anchor** to
    the wall leaves the glyphs unaffected by the wall - turning, looking
    sideways or up, the text neither tilts nor foreshortens, and appears
    **stuck to the screen, following the viewer**.

    The right way: draw the whole help block into a **texture** (independent
    of the viewer - "what is written on the wall"), project its four corners
    onto the wall plane, and paste it with `warpPerspective`.

    Only then is it really part of the wall:
      closer -> larger   sideways -> foreshortened   turning -> stays and tilts with the wall

    ## The wall basis must not come from the viewer's rotation either

    `wall_basis` should be `viewer.Rc0[0], Rc0[1]` (the window frame's right
    and down). Resolving the SVD's 180 deg ambiguity with
    `np.dot(right, Rn[0]) < 0`, where `Rn` is the viewer's orientation, made
    the whole block jump 1044 mm to the other side of the window at 150 deg.
    The wall is fixed, and so are its right and down.
    """
    tex, msk, blk_w_mm, blk_h_mm = _build_texture(rows, title)
    ctr, right, down = wall_axes(corners, wall_basis)
    half_w = max(abs(np.dot(p - ctr, right)) for p in np.asarray(corners, float))
    x0 = -(half_w * unit_mm) - GAP_MM - blk_w_mm
    y0 = -blk_h_mm / 2.0
    return paste_wall_block(img, tex, msk, (x0, y0, blk_w_mm, blk_h_mm),
                            ctr, right, down, pos, Rn, W, H, fx, fy, unit_mm,
                            bgr=bgr, ox=ox, oy=oy)


def wall_axes(corners, wall_basis=None):
    """Centre and two axes (right, down) of the wall plane.

    The basis must be **the window's own** axes (`wall_basis`), not the
       viewer's rotation. Resolving the SVD's 180 deg ambiguity with the
       viewer's orientation made the text jump to the other side of the
       window at 150 deg. The wall is fixed, and so are its right and down.
    """
    P = np.asarray(corners, float)
    ctr = P.mean(0)
    if wall_basis is not None:
        return ctr, np.asarray(wall_basis[0]), np.asarray(wall_basis[1])
    _, _, Vt = np.linalg.svd(P - ctr)
    return ctr, Vt[0], Vt[1]


def paste_wall_block(img, tex, msk, rect_mm, ctr, right, down, pos, Rn,
                     W, H, fx, fy, unit_mm, bgr=False, ox=0, oy=0):
    """Paste a texture into a rectangle on the wall plane (units: mm relative to the window centre).

    **Everything on the wall goes through this** (key help, title, ...).
       This is the easiest part of the file to get wrong or slow; a single
       copy avoids "changed one, forgot the other".
    """
    import cv2

    # `W, H` is **the image size used for projection** (principal point at
    #    its centre); `ox, oy` is **the crop origin**: subtracted after
    #    projection, to draw onto an `img` smaller than W x H.
    #
    #    VR crops a sub-region first and then composites (compositing on the
    #    large symmetric envelope image threw away 18-40% of the work with the
    #    cropped edges), so W,H describe the large image, ox,oy the crop
    #    position, and `img` is the cropped part.
    #
    #    **The offset must be applied after projection**; do not change the
    #       principal point to `W/2 - ox` instead. That changes the order of
    #       the floating-point operations, and `floor()` flips at exact integer
    #       boundaries, shifting the whole texture by one pixel - measured as
    #       a full row of 96 pixels off by up to 5/255. Subtracting an integer
    #       afterwards is exact (no rounding).
    x0, y0, blk_w_mm, blk_h_mm = rect_mm
    cx, cy = W / 2, H / 2
    ih, iw = img.shape[:2]

    dst = []
    for xm, ym in ((x0, y0), (x0 + blk_w_mm, y0),
                   (x0 + blk_w_mm, y0 + blk_h_mm), (x0, y0 + blk_h_mm)):
        p3 = ctr + right * (xm / unit_mm) + down * (ym / unit_mm)
        v = Rn @ (p3 - pos)
        if v[2] <= 1e-4:
            return img          # a corner behind the camera -> no valid homography
        dst.append([cx + fx * v[0] / v[2], cy + fy * v[1] / v[2]])
    # The offset is subtracted **after converting to float32**. Subtracting
    #    first lands the two paths on different float32 grid points (measured
    #    550.2526 vs 550.2527), the warp samples slightly differently, and a
    #    full row of text edge differs by 5/255. Subtracting an integer within
    #    the same exponent range is exact, so doing it afterwards adds no error.
    dst = np.array(dst, np.float32) - np.array([ox, oy], np.float32)

    # nothing to paste if the block is entirely off screen (only saves time; hides nothing)
    if (dst[:, 0].max() < 0 or dst[:, 0].min() > iw
            or dst[:, 1].max() < 0 or dst[:, 1].min() > ih):
        return img

    # Warp and blend **only inside the rectangle the text actually covers**.
    #    Warping to the full picture and blending the full picture in floating
    #    point wasted 85% of the work (text covers about 15%); it was the main
    #    part of the measured 15.2 ms.
    x0i = max(0, int(np.floor(dst[:, 0].min())))
    y0i = max(0, int(np.floor(dst[:, 1].min())))
    x1i = min(iw, int(np.ceil(dst[:, 0].max())) + 1)
    y1i = min(ih, int(np.ceil(dst[:, 1].max())) + 1)
    bw, bh = x1i - x0i, y1i - y0i
    if bw <= 0 or bh <= 0:
        return img

    ht, wt = tex.shape[:2]
    src = np.array([[0, 0], [wt, 0], [wt, ht], [0, ht]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst - np.array([x0i, y0i], np.float32))
    warped = cv2.warpPerspective(tex, M, (bw, bh), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpPerspective(msk, M, (bw, bh), flags=cv2.INTER_LINEAR)
    if bgr:
        warped = warped[:, :, ::-1]
    # Blending uses cv2's uint8 path, **not numpy floating point**.
    #
    #    `roi * (1 - a) + warped * a` in numpy is element-wise floating point,
    #    and the text block covers 18% of the picture (521x577) - measured
    #    **9.70 ms**, the largest part of the 14.5 ms spent on wall and frame.
    #    cv2's SIMD path takes **0.43 ms (22x)**.
    #
    #    This one is **not bit-identical**: cv2's `multiply` rounds, while the
    #       numpy path truncated with `.astype(uint8)`, so about 3% of the
    #       text's anti-aliased edge pixels differ by **1/255**. The scene
    #       itself is untouched.
    m3 = cv2.merge([wmask, wmask, wmask])
    roi = img[y0i:y1i, x0i:x1i]
    cv2.add(cv2.multiply(np.ascontiguousarray(warped), m3, scale=1 / 255.0),
            cv2.multiply(roi, cv2.bitwise_not(m3), scale=1 / 255.0), dst=roi)
    return img


def panel_size(rows, scale=1.0):
    """How wide and tall the panel really is (pixels).

    **Do not hard-code the width.** A fixed 196 px with font size 0.4 had two
       problems at 1746x982: the text was too small, and the longest row (the
       `cull` row, "0.005  232,687 / 2,366,565 (10%)") ran off the right edge.
       The width must come from **the content**, and the font size must follow the picture.
    """
    import cv2
    fs, th = 0.4 * scale, max(1, int(round(scale)))
    vx = int(66 * scale)
    w = 0
    h = 0
    for a, b in rows:
        if a is None:
            h += int(8 * scale)
            continue
        if b is None:
            (tw, _), _ = cv2.getTextSize(a, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.42 * scale, th)
            w = max(w, tw)
            h += int(20 * scale)
            continue
        (tw, _), _ = cv2.getTextSize(str(b), cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        w = max(w, vx + tw)
        h += int(17 * scale)
    return w, h


def draw_panel(img, rows, x0=12, y0=22, scale=1.0, bgr=False,
               outline=True, title_col=None, label_col=None,
               value_col=None):
    """Draw a list of (label, value) as an information panel. **Both players share this layout.**

    `rows`:
        ("TITLE", None)   a title
        (None, None)      a blank separator
        ("label", "value") an ordinary row

    Two colour schemes, chosen by the caller:

        `outline=True` (VR desktop panel)
            over a near-black background: bright text with a dark outline.
        `outline=False` (flat player)
            over the fogged wall (light grey): thin dark text is enough; an outline looks dirty.

    **The outline is not arbitrary**: text drawn directly over the picture
       meets both bright areas (a white wall) and dark ones (shadows), and
       single-colour text disappears on one of them. Dropping the outline
       requires **a controlled background** - in flat mode the wall fog keeps
       it light grey. With the wall turned off (`F`) the text sits on the raw
       picture and is hard to read in dark scenes.
    """
    import cv2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4 * scale
    th = max(1, int(round(scale))) if outline else 1     # thin
    tcol = title_col if title_col is not None else (120, 200, 255)
    lcol = label_col if label_col is not None else (190, 190, 190)
    vcol = value_col if value_col is not None else (255, 255, 255)
    if bgr:
        tcol, lcol, vcol = tcol[::-1], lcol[::-1], vcol[::-1]

    def put(txt, x, y, sc, col):
        if outline:
            cv2.putText(img, txt, (x, y), FONT, sc, (0, 0, 0), th + 2,
                        cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), FONT, sc, col, th, cv2.LINE_AA)

    y = int(y0)
    for a, b in rows:
        if a is None:
            y += int(8 * scale)
            continue
        if b is None:
            put(a, x0, y, 0.42 * scale, tcol)
            y += int(20 * scale)
            continue
        put(a, x0, y, fs, lcol)
        put(str(b), x0 + int(66 * scale), y, fs, vcol)
        y += int(17 * scale)
    return img


FLAT_KEYS = [
    ("WASD QE", "move"),
    ("DRAG", "look around"),
    ("N   M", "lens wide/tele"),
    ("SPACE", "time pause"),
    ("-  =  0", "time speed"),
    ("R", "reset view"),
    ("F", "frame + wall"),
    ("B", "background"),
    ("T", "time cull"),
    ("4 5 6", "cull def/-/+"),
    ("H", "hide panel"),
    ("G", "fullscreen"),
    ("J   K", "mute / audio"),
    ("(quote)", "volume"),
    ("ESC", "quit"),
]

VR_KEYS = [
    ("HEAD", "look / lean"),
    ("WASD QE", "move origin"),
    ("C", "recenter"),
    ("F", "frame + wall"),
    ("SPACE", "time pause"),
    ("-  =  0", "time speed"),
    ("R", "reset view"),
    ("J   K", "mute / audio"),
    ("(quote)", "volume"),
    ("ESC", "quit"),
    ("T", "time cull"),
    ("4 5 6", "cull def/-/+"),
    ("", "(5 stops at 0.005)"),
]


# ========================================================== window frame
#
# The frame width is in **physical mm**, not fixed pixels.
#
#    A fixed 12 px (flat) / 10 px (VR) frame was thicker than the window when
#    standing far away - at 1 m the window is only 71 px wide, with a 12 px
#    frame it looks like a sticker. Defined in mm on the wall, it thins with
#    distance after projection and always keeps the proportions of a frame.
FRAME_MM = 2.5
CORNER_MM = 13.0        # corner radius (physical mm on the wall)
CORNER_SEG = 7          # segments per rounded corner
# A single thin line. A "dark line + white main line" for a sense of depth
#    made the white line vanish on the light wall and the dark one look
#    thick, like a sticker. The frame's job is to **mark the boundary**, not decorate.
FRAME_EDGE = (86, 156, 214)
# The nominal frame (no win in window.json) is **black**, distinguishable at
#    a glance from a real one. Flat and VR share this value; do not write a copy.
FRAME_NOMINAL = (0, 0, 0)


def window_polygon(corners, pos, Rn, W, H, fx, fy, unit_mm,
                   radius_mm=CORNER_MM, ox=0, oy=0):
    """The **rounded** window frame polygon projected into the image. Returns (polygon, line width px) or (None, 0).

    Rounded corners must be generated **on the wall plane** and then
       projected, not drawn on the image. Drawn on the image, the corners
       stay perfect circles when the window is seen at an angle, while real
       rounded corners are squashed into ellipses by perspective - the
       inconsistency is especially visible in VR, where each eye squashes
       by a different amount.

    Method: convert the corners to 2D coordinates on the wall plane, build
    the rounded path there, then convert each point back to 3D and project it.
    """
    # `W, H` is the projection image size (principal point at the centre), `ox, oy` the crop origin.
    # The offset is subtracted **after** projection; see the explanation in paste_wall_block.
    cx, cy = W / 2, H / 2

    P = np.asarray(corners, float)
    ctr = P.mean(0)
    _, _, Vt = np.linalg.svd(P - ctr)
    right, down = Vt[0], Vt[1]

    # 2D coordinates on the wall plane (scene units)
    uv = np.array([[np.dot(p - ctr, right), np.dot(p - ctr, down)] for p in P])
    r = radius_mm / unit_mm

    pts2 = []
    n = len(uv)
    for i in range(n):
        a, b, c = uv[i - 1], uv[i], uv[(i + 1) % n]
        e1, e2 = a - b, c - b
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if l1 < 1e-9 or l2 < 1e-9:
            pts2.append(b)
            continue
        rr = min(r, l1 / 2.5, l2 / 2.5)
        e1, e2 = e1 / l1, e2 / l2
        p1, p2 = b + e1 * rr, b + e2 * rr
        for k in range(CORNER_SEG + 1):        # quadratic Bezier: round enough and cheap
            t = k / CORNER_SEG
            pts2.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * b + t ** 2 * p2)

    out, depth = [], []
    for u2, v2 in pts2:
        p3 = ctr + right * u2 + down * v2
        v = Rn @ (p3 - pos)
        if v[2] <= 1e-4:
            return None, 0
        out.append((cx + fx * v[0] / v[2], cy + fy * v[1] / v[2]))
        depth.append(v[2])
    tw = max(1, int(round(fx / float(np.mean(depth)) * (FRAME_MM / unit_mm))))
    # The offset is subtracted **after** conversion to int. int32 conversion
    #    truncates toward zero, so subtracting first would truncate negative
    #    coordinates into a different cell. Same reason as in paste_wall_block.
    return (np.array(out, np.int32).reshape(-1, 1, 2)
            - np.array([[[ox, oy]]], np.int32)), tw


# ========================================================== title above the frame
#
# Like the key help, **pasted onto the wall plane**, not overlaid on the picture.
#    Reason in the module docstring: text on the picture is at the same pixel
#    in both eyes and is perceived at infinity, while the wall is within a
#    metre - an uncomfortable conflict.
TITLE_TEXT = "4DGS WINDOW"
TITLE_H_MM = 46.0        # height of the block (text + arrow)
TITLE_GAP_MM = 16.0      # distance above the frame's top edge
_TITLE_CACHE = {}


def _build_title(text):
    """Title texture: one line of text with an arrow below it pointing at the window.

    The arrow is drawn (`arrowedLine`), not a Unicode down arrow -
       `cv2.putText` only draws ASCII; U+2B07 would become a question-mark box.
    """
    import cv2
    if text in _TITLE_CACHE:
        return _TITLE_CACHE[text]

    Wt = int(BLOCK_W_MM * TEX_PPMM)
    Ht = int(TITLE_H_MM * TEX_PPMM)
    tex = np.zeros((Ht, Wt, 3), np.uint8)
    msk = np.zeros((Ht, Wt), np.uint8)

    scale = LINE_MM * TEX_PPMM / 30.0
    th = max(1, int(round(scale * 1.7)))
    (tw, thh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x = (Wt - tw) // 2
    y = int(thh + 4 * TEX_PPMM)
    for dstim, col in ((tex, TITLE_RGB), (msk, 255)):
        cv2.putText(dstim, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col,
                    th, cv2.LINE_AA)
        cv2.arrowedLine(dstim, (Wt // 2, y + int(6 * TEX_PPMM)),
                        (Wt // 2, Ht - int(2 * TEX_PPMM)), col,
                        max(2, th), cv2.LINE_AA, tipLength=0.45)

    _TITLE_CACHE[text] = (tex, msk, BLOCK_W_MM, TITLE_H_MM)
    return _TITLE_CACHE[text]


def draw_wall_title(img, corners, pos, Rn, W, H, fx, fy, unit_mm,
                    bgr=False, text=TITLE_TEXT, wall_basis=None, ox=0, oy=0):
    """Paste a "4DGS WINDOW" block with a down arrow **above** the frame, pointing at the window.

    Its position follows the window's own geometry (centred horizontally,
    `TITLE_GAP_MM` above the top edge), so it grows when approached and
    foreshortens when seen from the side, moving with the frame.
    """
    tex, msk, blk_w_mm, blk_h_mm = _build_title(text)
    ctr, right, down = wall_axes(corners, wall_basis)
    P = np.asarray(corners, float)
    half_h = max(abs(np.dot(p - ctr, down)) for p in P)
    x0 = -blk_w_mm / 2.0
    y0 = -(half_h * unit_mm) - TITLE_GAP_MM - blk_h_mm
    return paste_wall_block(img, tex, msk, (x0, y0, blk_w_mm, blk_h_mm),
                            ctr, right, down, pos, Rn, W, H, fx, fy, unit_mm,
                            bgr=bgr, ox=ox, oy=oy)


_WALL_CACHE = {}


def draw_wall(img, poly, fog, gray):
    """Fog everything outside the window into a wall; **flat and VR share this**.

    This used to exist in both `viewer.py` and `vr_viewer.py`; only the
       flat copy got optimized, while the VR copy kept the old slow version
       and even ran twice per frame - noticed as "the wall stutters".
       **One copy cannot be forgotten.**

    Two points:

    * **uint8 SIMD path, no floating point.** Allocating a wall image plus a
      float alpha per frame and blending the whole image in floating point
      took 14.5 ms.
    * **Restore the inside of the window only in its small region.** The
      window usually covers 1-2% of the picture; masking the whole image is wasted.

    Returns the composited image (a new array, not modified in place).
    """
    import cv2
    wall = _WALL_CACHE.get(img.shape)
    if wall is None:
        wall = np.full(img.shape, gray, np.uint8)
        _WALL_CACHE[img.shape] = wall
    fogged = cv2.addWeighted(img, 1.0 - fog, wall, fog, 0.0)

    # mask and restore only inside the window's bounding rectangle
    x, y, w, h = cv2.boundingRect(poly)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return fogged
    ins = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(ins, [poly - np.array([[[x0, y0]]], poly.dtype)], 255)
    cv2.copyTo(img[y0:y1, x0:x1], ins, fogged[y0:y1, x0:x1])
    return fogged


def draw_frame(img, poly, tw, bgr=False, col=None):
    """Draw the window frame: one thin line.

    Without `col`, `FRAME_EDGE` (sky blue) is used.
    **A nominal frame passes `FRAME_NOMINAL` (black)** - when the size was not
    computed from this recording, the colour must show it at a glance, not
    only a line of text in the panel.
    Colours passed in are always RGB; they are flipped here when `bgr=True`.
    """
    import cv2
    if poly is None:
        return img
    c = FRAME_EDGE if col is None else col
    cv2.polylines(img, [poly], True, c[::-1] if bgr else c, tw, cv2.LINE_AA)
    return img
