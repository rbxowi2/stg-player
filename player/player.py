"""4DGS player - **single entry point** for flat screen and VR.

    python player.py <shooting root>              flat screen
    python player.py <shooting root> --vr         VR (needs a headset)
    python player.py --check-vr                   only report the VR device state

The model and iteration come from `player.json` or the automatic rules (see
`captureset.py`). The older form (a PLY plus the `colmap_N` root) still works
for quick comparisons.

## A machine without VR hardware does not crash because of this program

Before `--vr`, `player_core.vr_status()` is asked; **it does not initialize
OpenVR**, so it can answer on a machine with no VR at all. If VR is not ready
it says what is missing and stops, instead of dying inside `openvr.init()`.

The `vr_viewer` module (with `glfw` / `openvr` / `OpenGL`) is **imported only
when VR is actually used** - a flat-screen machine does not need those three
packages at all.

## The two run() loops are deliberately separate

Flat mode is a cv2 event loop and VR is the OpenVR compositor loop; they are
different in nature. What they share is `player_core.Scene` (scene, geometry,
scale, audio, playback state) and the constants. Forcing the two loops
together would only produce a pile of `if vr:`.

Switching between flat and VR during playback is not implemented; changing
mode means restarting, which reloads the model.
"""
from __future__ import annotations

import argparse
import sys

import captureset as CS
from player_core import vr_status


def build_parser():
    ap = argparse.ArgumentParser(
        prog="player.py",
        description="4DGS player (flat / VR). Giving the shooting root is recommended.")
    ap.add_argument("ply", nargs="?", default=None,
                    metavar="ROOT_OR_PLY")
    ap.add_argument("root", nargs="?", default=None,
                    metavar="COLMAP_ROOT", help="only needed when a PLY is given directly")
    ap.add_argument("--vr", action="store_true",
                    help="play in VR (needs SteamVR and a headset; if not ready, says what is missing)")
    ap.add_argument("--check-vr", action="store_true",
                    help="only check the VR device state and exit, without loading a model")
    ap.add_argument("--model", default=None, help="model to play, for example H or H_out")
    ap.add_argument("--iter", type=int, default=None, help="iteration to play")
    ap.add_argument("--dataset", default=None, help="dataset to use (usually detected)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="render resolution factor. 1.0 = best quality (default); lower for frame rate")
    ap.add_argument("--secs", type=float, default=None,
                    help="clip length in seconds. Default from window.json")
    return ap


def resolve(a):
    """Arguments -> (ply, colmap root). **Moved out of `main()` unchanged.**

    It is separate so a regression test can take **the same path as the
    user**: a test that constructed the viewer directly once missed a bug in
    `main()` (a missing import) - all tests passed, but the player did not start.
    """
    if a.ply.lower().endswith(".ply"):
        if not a.root:
            raise SystemExit("  ERROR when a PLY is given, the second argument must be the colmap_N root")
        return a.ply, a.root
    ply, root, info = CS.resolve(a.ply, model=a.model, it=a.iter,
                                 dataset=a.dataset)
    print(info)
    return ply, root


def report_vr():
    """Print the VR state; return whether it is ready."""
    ok, why = vr_status()
    print(f"  VR device: {'OK ' if ok else 'NOT READY '}{why}")
    if not ok:
        print("      -> flat mode is not affected; play without --vr.")
    return ok


def main(argv=None):
    a = build_parser().parse_args(argv)

    if a.check_vr:
        return 0 if report_vr() else 1

    if a.ply is None:
        build_parser().print_help()
        return 2

    # The VR readiness check comes **before loading the model** - loading
    #    takes tens of seconds, and reporting "no headset" afterwards is the
    #    worst possible order.
    if a.vr and not report_vr():
        print("  ERROR VR is not ready; not starting.")
        return 1

    ply, root = resolve(a)

    if a.vr:
        # Imported only here: `vr_viewer` pulls in glfw / openvr / OpenGL,
        #    and a flat-screen machine should not need them.
        import vr_viewer
        vr_viewer.VRViewer(ply, root, a.scale, a.secs).run()
    else:
        import viewer
        viewer.Viewer(ply, root, a.secs, a.scale).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
