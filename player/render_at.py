"""render_at.py - render any 3DGS/STG PLY from **any pose**, optionally compared with ground truth

## Why this is written here

Each trainer has its own hold-out convention (LichtFeld samples by index
with `--test-every N`, STG always holds out the first camera). To compare
numbers across trainers they must be measured with **the same ruler**, not
with each trainer's own evaluation.

A test camera may also be one that **never took part in training** (for
example a camera placed at the centre of the window), so the only way to
compare is to render its pose afterwards.

It is also the core of the player: "load a PLY -> give a pose -> get an
image" is the player's main loop.

## Usage

    python render_at.py --ply <PLY> --model <COLMAP sparse folder> \
        --image <image name such as cam_B/p2.jpg> --gt <ground truth image> --out <output png>

Two kinds of PLY are supported:
  * **3DGS**: has `f_rest_*` (spherical harmonics), evaluated along the view direction
  * **STG lite**: only `f_dc_*` (direct RGB), plus motion/omega/trbf; time given with `--t`
"""
import argparse
import importlib.util
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# `utils.*` and `colmap_loader` live in the SpacetimeGaussians thirdparty
#    tree. This file is not inside that tree, so it cannot find them relative
#    to itself. It first looks next to itself (a copy inside the trainer),
#    then STG_HOME, then the repository's third_party folder.
_HERE = os.path.abspath(os.path.dirname(__file__))
_TP = "thirdparty/gaussian_splatting"
STG_HOME = _HERE if os.path.isdir(os.path.join(_HERE, _TP)) else \
    os.environ.get("STG_HOME", os.path.join(os.path.dirname(_HERE), "third_party", "SpacetimeGaussians"))
if not os.path.isdir(os.path.join(STG_HOME, _TP)):
    raise SystemExit(
        f"  ERROR SpacetimeGaussians {_TP} not found\n"
        f"    looked in: {_HERE}, {STG_HOME}\n"
        f"    run setup.ps1, or set the environment variable STG_HOME to the SpacetimeGaussians folder")
sys.path.append(os.path.join(STG_HOME, _TP))

import numpy as np
import torch
from plyfile import PlyData

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.sh_utils import eval_sh

_s = importlib.util.spec_from_file_location(
    "cl", os.path.join(STG_HOME, _TP, "scene/colmap_loader.py"))
cl = importlib.util.module_from_spec(_s); _s.loader.exec_module(cl)


def load_ply(path):
    v = PlyData.read(path)["vertex"]
    n = [p.name for p in v.properties]
    g = {}
    g["xyz"] = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32).cuda()
    g["opacity"] = torch.sigmoid(torch.tensor(np.asarray(v["opacity"]),
                                              dtype=torch.float32).cuda()).unsqueeze(1)
    g["scale"] = torch.exp(torch.tensor(
        np.stack([v[f"scale_{i}"] for i in range(3)], 1), dtype=torch.float32).cuda())
    r = torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1),
                     dtype=torch.float32).cuda()
    # Same as in render(): trained models contain **quaternions with norm
    #    exactly 0**; a hand-written division would bake NaN into g["rot"]
    #    **at load time**. F.normalize is used (as upstream does).
    g["rot"] = torch.nn.functional.normalize(r, dim=1)
    dc = torch.tensor(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1),
                      dtype=torch.float32).cuda()
    rest = sorted([x for x in n if x.startswith("f_rest_")],
                  key=lambda s: int(s.split("_")[-1]))
    if rest:                                        # 3DGS: spherical harmonics
        g["kind"] = "3dgs"
        e = torch.tensor(np.stack([v[x] for x in rest], 1), dtype=torch.float32).cuda()
        # stored in the PLY as (3, (deg+1)^2-1), flattened
        g["sh"] = torch.cat([dc.unsqueeze(1),
                             e.reshape(e.shape[0], 3, -1).transpose(1, 2)], 1)
        g["deg"] = int(math.sqrt(g["sh"].shape[1]) - 1)
    else:                                           # STG lite: direct RGB
        g["kind"] = "stg"
        g["rgb"] = dc
        for k in ("trbf_center", "trbf_scale"):
            if k in n:
                g[k] = torch.tensor(np.asarray(v[k]), dtype=torch.float32).cuda().unsqueeze(1)
        if "motion_0" in n:
            g["motion"] = torch.tensor(np.stack([v[f"motion_{i}"] for i in range(9)], 1),
                                       dtype=torch.float32).cuda()
        if "omega_0" in n:
            g["omega"] = torch.tensor(np.stack([v[f"omega_{i}"] for i in range(4)], 1),
                                      dtype=torch.float32).cuda()
    return g


def camera_from_colmap(model_dir, image_name, res=1):
    """res: resolution divisor. **Must match training** - STG's `cfg_lite.json`
    says `"resolution": 2` (half-resolution training and evaluation); LichtFeld
    has `--max-width`. PSNR at different resolutions **cannot be compared**:
    the same model scores several dB higher at half resolution."""
    E = cl.read_extrinsics_binary(os.path.join(model_dir, "images.bin"))
    I = cl.read_intrinsics_binary(os.path.join(model_dir, "cameras.bin"))
    ex = next((e for e in E.values() if e.name == image_name), None)
    if ex is None:
        raise SystemExit(f"image {image_name} not found; available: {[e.name for e in E.values()]}")
    it = I[ex.camera_id]
    # `res` may be fractional (the player passes 1/scale), so it is cast to
    #    int explicitly; `//` on a float returns a float, which crashes the rasterizer.
    fx, fy = it.params[0] / res, it.params[1] / res
    W, H = int(it.width // res), int(it.height // res)
    return dict(R=np.transpose(cl.qvec2rotmat(ex.qvec)), T=np.array(ex.tvec),
                FoVx=2 * math.atan(W / (2 * fx)), FoVy=2 * math.atan(H / (2 * fy)),
                W=W, H=H)


# Number of Gaussians actually sent to the rasterizer by the last render(); shown in the player HUD.
LAST_SENT = 0


CULL_DEFAULT = 0.005
CULL_MAX = 0.2


def cull_step(cur, up, lo=CULL_DEFAULT):
    """Adjust the time-window culling threshold. `up=True` **raises the threshold** (culls more, faster).

    **Both players share this function**; do not write a copy.

    Keys are `5` smaller and `6` larger - left to right on the keyboard. What
    changes is **the threshold number**, not "culling strength", but the two
    point the same way: larger threshold = more dropped = faster.

    **The lower limit is the default 0.005 for both players; pressing below it does nothing.**

       In VR it is a safety issue: halving the threshold doubles the number
       of Gaussians sent, and at VR resolution (both eyes x symmetric
       envelope) that overflows the rasterizer's buffers - measured at
       6.3 MPix with 2.33 million Gaussians: `Storage size calculation
       overflowed` (int32 overflow). Not slower - a crash.

       On a flat screen it is **pointless**: 0.005 -> 0.00125 raised the sent
       share from 9% to 24% and dropped 25 fps to 20 with no visible quality
       gain (see below). Speed is paid, nothing is gained.

    Steps multiply/divide by 2 rather than add a constant: the threshold
    spans several orders of magnitude, and linear steps would crawl at the
    low end and overshoot at the high end.

    The default 0.005 was **chosen by measurement**: 0.005 and 0.002 give
       almost the same quality (synthetic-view PSNR 58.2 vs 58.3) at half the
       speed, meaning the error comes from the accumulation of the large mass
       of even fainter Gaussians. **Lowering the threshold cannot fix quality,
       only slow things down.**

       To check this, use `T` (culling off) to compare the two ends; there is
       nothing worth tuning in between.

    Pressing either key while culling is off (`None`) restarts at the default.
    """
    if cur is None:
        return CULL_DEFAULT
    v = cur * 2.0 if up else cur / 2.0
    return min(CULL_MAX, max(lo, v))


def to_uint8(img, bgr=False, alpha=False, flip_v=False):
    """Move render() output (3,H,W) into a displayable uint8 numpy array (H,W,C).

    **Both players use this; do not write a copy.** When the flat and VR
       players each had their own copy, one was changed and the other forgotten.

    The previous form `(im.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)`
    stacked three costs:

        float32 is transferred          uint8 is 1/4 of the data for the same picture
        non-contiguous after permute    .cpu() does a strided copy, not one memcpy
        x255 and astype run in numpy    single-threaded, over millions of elements

    Converting on the GPU and transferring one contiguous uint8 block:
    measured 1746x982 **14.5 -> 1.0 ms**, 1888x2076 (one VR eye) **34.8 -> 3.0 ms**.

    This is **bit-identical**, not an approximation. `render()` already
       clamps to [0,1]; what remains is `x255` and truncation, and float32
       multiplication and truncation behave the same on CPU and GPU.

    Parameters (each matches an existing need of the caller; do not repeat it there):
        bgr     OpenCV wants BGR (flat player)
        alpha   append an opaque A channel (OpenGL textures want RGBA)
        flip_v  flip vertically. OpenGL textures start bottom-left, numpy images top-left (VR)
    """
    x = (img.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8)
    if bgr:
        x = x.flip(2)
    if alpha:
        x = torch.cat(
            [x, torch.full(x.shape[:2] + (1,), 255, dtype=torch.uint8,
                           device=x.device)], dim=2)
    if flip_v:
        x = x.flip(0)
    return np.ascontiguousarray(x.contiguous().cpu().numpy())


def seg_weight(t, seg, ramp, first=False, last=False):
    """Segment weight for combined training: 1.0 inside, a 0-1 ramp at the boundary, 0 outside.

    Neighbouring segments share their boundary step (decided by the merge
    tool), so one segment's `t1` and the next one's `t0` are **the same
    instant**. The ramp straddles that instant (half on each side), so the
    two segments add up continuously - fading out only inside a segment would
    leave a one-step dip at the boundary.

    The start of the first segment and the end of the last one have no ramp:
    there is no neighbour to blend with.
    """
    t0, t1 = float(seg["t0"]), float(seg["t1"])
    r = float(ramp or 0.0)
    if r <= 0:
        return 1.0 if t0 <= t <= t1 else 0.0
    lo = t0 - r / 2 if not first else t0 - 1e-9
    hi = t1 + r / 2 if not last else t1 + 1e-9
    if t <= lo or t >= hi:
        return 0.0
    w = 1.0
    if not first and t < t0 + r / 2:
        w = min(w, (t - lo) / r)
    if not last and t > t1 - r / 2:
        w = min(w, (hi - t) / r)
    return max(0.0, min(1.0, w))


def slice_segments(g, t, segments, ramp):
    """Keep only the segments active at this instant; returns (sliced g, per-row weights).

    **Sliced first**, not masked afterwards: the per-point work (`dt`/`op`/
    `motion`) and the depth matrix product for projection all scale with
    **the total count**. The other four segments of a five-segment model
    never appear on screen, so there is no reason to compute them.

    Segments are contiguous row ranges in the PLY, so normally (away from a
    boundary) this is a **zero-copy view**; only during a cross-fade are two
    segments concatenated.

    With no active segment returns (None, None); the caller draws the background.
    """
    act = []
    for i, seg in enumerate(segments):
        k = seg_weight(t, seg, ramp, first=(i == 0),
                       last=(i == len(segments) - 1))
        if k > 0:
            a = int(seg["start"])
            act.append((a, a + int(seg["count"]), float(k)))
    if not act:
        return None, None
    n = sum(b - a for a, b, _ in act)
    out = {}
    for key, v in g.items():
        if torch.is_tensor(v) and v.shape[0] == g["xyz"].shape[0]:
            out[key] = (v[act[0][0]:act[0][1]] if len(act) == 1
                        else torch.cat([v[a:b] for a, b, _ in act]))
        else:
            out[key] = v
    if all(k >= 1.0 for _, _, k in act):
        return out, None                      # all at full weight; no multiplication needed
    w = torch.empty(n, device=g["xyz"].device)
    at = 0
    for a, b, k in act:
        w[at:at + (b - a)] = k
        at += b - a
    return out, w


def render(g, cam, t=0.0, znear=None, zfar=None, bg=0.0, cull_opacity=None,
           segments=None, seg_ramp=0.0):
    """With znear/zfar None, they are **derived from the scene**.

    `cull_opacity`: time-window culling threshold. None = no culling (original behaviour).

    Each STG Gaussian covers only a short span of time (`trbf_center` /
       `trbf_scale`); far from its time centre it is almost fully transparent
       in this frame. Without culling it is still projected, binned, sorted
       and blended - full cost, no contribution. Measured: **86-89% of each
       frame is like this**. Culling took 8-12 fps to 52-66 fps and cut peak
       VRAM by 72%.

       This does not bypass the design; it uses what STG already expresses
       (algorithmically equivalent).

    The threshold is not worth tuning: 0.005 and 0.002 give almost the same
       quality at half the speed, so the error comes from the accumulation of
       the large mass below. Lowering it cannot help, only slow down.

    These two values must follow the scene scale. Hard-coded znear=0.01 /
    zfar=100 can be a thousand times too small in SfM units - a tiny `znear`
    makes the screen radius of near Gaussians explode, the rasterizer's tile
    index overflows, and it fails with `CUDA illegal memory access` (not NaN;
    very hard to trace).
    """
    from diff_gaussian_rasterization_ch3 import (GaussianRasterizationSettings,
                                                 GaussianRasterizer)
    # combined-training models: slice to the segments active now, so everything below only covers those.
    seg_w = None
    if segments:
        g, seg_w = slice_segments(g, t, segments, seg_ramp)
        if g is None:
            return torch.full((3, cam["H"], cam["W"]), float(bg), device="cuda")
    w2c = torch.tensor(getWorld2View2(cam["R"], cam["T"])).transpose(0, 1).cuda()
    depth = (g["xyz"] @ w2c[:3, :3] + w2c[3, :3])[:, 2]
    if znear is None or zfar is None:
        d = depth[depth > 0]
        if d.numel() == 0:
            d = torch.tensor([1.0], device=depth.device)
        hi = torch.quantile(d, 0.999).item()
        med = torch.quantile(d, 0.5).item()
        # znear cannot come from "the nearest Gaussian". Once the camera moves
        #    into the scene, the nearest Gaussian touches the face, znear goes to
        #    almost 0 -> screen radius explodes -> tile index overflow ->
        #    `CUDA illegal memory access`. It is tied to the scene scale instead.
        # The **median depth** is the reference, not the farthest point. With
        #    the view turned away the scene is behind, and only a few Gaussians
        #    sit at the near plane -> radius explodes -> tile overflow -> crash.
        #    Tying znear to the median depth removes those pathological Gaussians too.
        znear = med * 0.15 if znear is None else znear
        zfar = hi * 3.0 if zfar is None else zfar
    proj = getProjectionMatrix(znear, zfar, cam["FoVx"], cam["FoVy"]).transpose(0, 1).cuda()
    full = (w2c.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    campos = w2c.inverse()[3, :3]

    # Cull Gaussians too close or behind the camera. Real renderers do this;
    #    without it, interactive movement always hits the tile overflow above.
    # Besides the near plane, also cull Gaussians with **too large a projected
    #    radius**. With the view turned sideways or back, a few Gaussians sit at
    #    the near plane at grazing angles with screen radii of tens of
    #    thousands of pixels -> tile index overflow -> CUDA crash.
    #    radius ~ f x scale / depth; anything above the image diagonal is dropped.
    _f = cam["W"] / (2.0 * math.tan(cam["FoVx"] / 2))
    _r = _f * g["scale"].max(dim=1).values / depth.clamp(min=1e-6)
    keep = (depth > znear) & (_r < max(cam["W"], cam["H"]))
    xyz, rot = g["xyz"], g["rot"]
    if g["kind"] == "stg":
        dt = t - g.get("trbf_center", torch.zeros_like(g["opacity"]))
        op = g["opacity"] * torch.exp(-(dt / torch.exp(
            g.get("trbf_scale", torch.zeros_like(dt)))) ** 2)
        if "motion" in g:
            m = g["motion"]
            xyz = xyz + m[:, 0:3] * dt + m[:, 3:6] * dt ** 2 + m[:, 6:9] * dt ** 3
        if "omega" in g:
            # Must use F.normalize, not a hand-written rot / rot.norm().
            #
            #    Trained models contain **quaternions with norm exactly 0**
            #    (measured: 13,381 / 12,772 / 5,948 in three models). A
            #    hand-written division gives 0/0 = NaN; a NaN quaternion -> NaN
            #    covariance -> the rasterizer computes tile bounds from NaN ->
            #    tile index out of range -> **CUDA illegal memory access**.
            #
            #    `F.normalize` has an eps in the denominator and safely returns
            #    a zero vector for a zero vector - upstream's
            #    `ourslite.py get_rotation()` uses exactly that
            #    (`self.rotation_activation = torch.nn.functional.normalize`).
            #
            #    This is not a safety net but **a fix for a deviation from the
            #    upstream implementation**. The crash angle varied and
            #    CUDA_LAUNCH_BLOCKING did not catch it, because it only fails
            #    when a NaN Gaussian happens to pass the keep mask into tile binning.
            rot = torch.nn.functional.normalize(rot + g["omega"] * dt, dim=1)
        colors = g["rgb"]
    else:
        op = g["opacity"]
        d = xyz - campos.repeat(xyz.shape[0], 1)
        d = d / d.norm(dim=1, keepdim=True)
        colors = torch.clamp_min(eval_sh(g["deg"], g["sh"].transpose(1, 2), d) + 0.5, 0.0)

    # Cross-fade at segment boundaries (`slice_segments` already removed whole
    # segments outside; this only handles the short time straddling a boundary).
    # Applied to `op` and **before** the time-window culling below: Gaussians
    # pushed down by the ramp are culled as well and cost no rasterization.
    if seg_w is not None:
        op = op * seg_w.reshape(-1, 1)

    # Time-window culling. **Uses the `op` computed above** (including the trbf
    # time window), so nothing is recomputed; the cost is one comparison.
    if cull_opacity is not None:
        keep = keep & (op.reshape(-1) > float(cull_opacity))

    n_keep = int(keep.sum())
    global LAST_SENT
    LAST_SENT = n_keep
    if n_keep == 0:
        # With the view entirely off the scene no Gaussian remains. An empty
        #    tensor passed to the rasterizer causes CUDA illegal memory access -
        #    easy to hit while turning interactively.
        return torch.full((3, cam["H"], cam["W"]), float(bg), device="cuda")
    if n_keep != keep.numel():
        xyz, rot = xyz[keep], rot[keep]
        op, colors = op[keep], colors[keep]
        scales = g["scale"][keep]
    else:
        scales = g["scale"]

    rs = GaussianRasterizationSettings(
        image_height=cam["H"], image_width=cam["W"],
        tanfovx=math.tan(cam["FoVx"] * 0.5), tanfovy=math.tan(cam["FoVy"] * 0.5),
        bg=torch.full((9,), float(bg), device="cuda"), scale_modifier=1.0,
        viewmatrix=w2c, projmatrix=full, sh_degree=0, campos=campos, prefiltered=False)
    # means2D must be a separate tensor **with gradient**. With plain zeros,
    #    the second rasterizer call in the same process fails with CUDA illegal
    #    memory access (the first call and single renders work; only
    #    consecutive renders fail - very hard to trace).
    sp = torch.zeros_like(xyz, requires_grad=True) + 0
    try:
        sp.retain_grad()
    except Exception:
        pass
    out = GaussianRasterizer(raster_settings=rs)(
        means3D=xyz, means2D=sp,
        shs=None, colors_precomp=colors, opacities=op,
        scales=scales, rotations=rot, cov3D_precomp=None)
    img = torch.clamp(out[0], 0, 1).detach().clone()
    del out, sp
    # torch.cuda.empty_cache() used to be called here every time.
    #
    #    It was added against "consecutive renders cause CUDA illegal memory
    #    access" - but the **real cause was NaN from the hand-written
    #    quaternion normalization**, which is fixed. empty_cache only delayed
    #    the symptom.
    #
    #    And it is **a hard sync + allocator flush**, not cheap:
    #        912x512    49.7 -> 46.9 ms (1.06x)
    #        1000x1100 103.7 -> 80.5 ms (**1.29x**)
    #
    #    The caller now calls it **every few dozen frames** instead (see vr_viewer.py).
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--gt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--t", type=float, default=0.0)
    p.add_argument("--res", type=int, default=1,
                   help="resolution divisor; must match training (resolution in the STG cfg)")
    a = p.parse_args()

    with torch.no_grad():
        g = load_ply(a.ply)
        cam = camera_from_colmap(a.model, a.image, a.res)
        print(f"  PLY {os.path.basename(a.ply)}: {g['xyz'].shape[0]:,} Gaussians, kind {g['kind']}")
        print(f"  camera {a.image}: {cam['W']}x{cam['H']}  "
              f"FoV {math.degrees(cam['FoVx']):.1f} deg x {math.degrees(cam['FoVy']):.1f} deg")
        im = render(g, cam, a.t)

        if a.gt:
            import cv2
            gt = cv2.imread(a.gt)
            if a.res != 1:
                gt = cv2.resize(gt, (cam["W"], cam["H"]), interpolation=cv2.INTER_AREA)
            gt = gt[:, :, ::-1].astype(np.float32) / 255.0
            gtt = torch.tensor(gt, dtype=torch.float32).permute(2, 0, 1).cuda()
            mse = ((im - gtt) ** 2).mean().item()
            print(f"  PSNR = {10*math.log10(1/max(mse,1e-12)):.2f} dB")
            if a.out:
                d = torch.clamp((im - gtt).abs() * 4, 0, 1)
                grid = torch.cat([gtt, im, d], dim=1)
                import torchvision
                torchvision.utils.save_image(grid, a.out)
                print(f"  -> {a.out} (top ground truth / middle render / bottom difference x4)")
        elif a.out:
            import torchvision
            torchvision.utils.save_image(im, a.out)
            print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
