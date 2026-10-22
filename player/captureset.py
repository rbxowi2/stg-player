#!/usr/bin/env python3
"""captureset.py - resolving a shooting root (shared by the flat and VR players)

    from captureset import resolve
    ply, dataset, info = resolve("<shooting root>", model="I", it=100000)


## What a shooting root is

**Everything produced by one shoot lives in one root folder.** Another shoot
is another root.

    my_shoot/                       <- shooting root (rename it freely; no metadata to change)
        rig/                            copy of the calibration used for this shoot (copied by preprocessing)
        rec_.../                        the recording
        stg/  stg240/                   datasets (one shoot may have several, with different step counts)
        A_out/ ... I_out/               models of the same shoot with different training parameters

**Playback only needs this root** - `rig/` is not read; it is an input of
preprocessing. The whole folder can be moved, compressed or handed over.


## `player.json` - the only file written by hand

Placed in the shooting root; it **only records which model is the default**:

```json
{ "default": { "model": "I_out", "iter": 100000 } }
```

Without it the player still works (automatic rules below). **It only stores
what cannot be derived from the disk.**

**No note fields.** Folders hold **results**; explanations belong in
documents. Notes inside data go stale because nobody maintains them, and they
drown the start-up messages.

### Why "which is the default" cannot be derived

Because **the largest iteration is not necessarily the best**. Example:

| iter | test PSNR | saved? |
|---|---|---|
| 70,000 | 30.948 | yes |
| **80,000** | **31.061 <- peak** | no |
| 100,000 | 30.977 | yes |

Which one is best needs an evaluation; the player should not guess. It is
**a judgement** and has to be written down by a person.

Conversely, everything else is on disk and must **not** be written into
`player.json` (it would go out of sync without anyone noticing):

| To know | Read from |
|---|---|
| which models exist | scan `*_out/point_cloud/iteration_*` |
| which iterations exist | same, the folder names |
| which dataset a model was trained on | **`source_path` in `<model>/cfg_args`** |
| number of time steps | `duration` in `cfg_args` |

`cfg_args` is **written by the trainer itself**, not maintained by hand, so
it cannot be wrong.


## Defaults without `player.json`

- **model**: the only one if there is one; otherwise **the most recently modified** (usually the one just trained)
- **iteration**: the largest

`--model` / `--iter` on the command line take precedence over `player.json`.
"""

from __future__ import annotations

import json
import os
import re

PLAYER_JSON = "player.json"


def _iters(mdl_dir):
    """Returns [(iteration, PLY path)], ascending."""
    pc = os.path.join(mdl_dir, "point_cloud")
    out = []
    if os.path.isdir(pc):
        for d in os.listdir(pc):
            m = re.fullmatch(r"iteration_(\d+)", d)
            p = os.path.join(pc, d, "point_cloud.ply")
            if m and os.path.isfile(p):
                out.append((int(m.group(1)), p))
    return sorted(out)


def _dataset_of(mdl_dir, root):
    """Which dataset this model was trained on - read from the trainer's own `cfg_args`.

    `source_path` looks like `.../my_shoot/stg240/colmap_0`; its parent is the dataset.

    **Only the folder name is trusted, not the whole path**: if training ran
       on another machine or path, the absolute path is invalid. The name is
       looked up under **this** shooting root instead, so the result survives
       moving and renaming the whole folder.
    """
    f = os.path.join(mdl_dir, "cfg_args")
    if os.path.isfile(f):
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
            m = re.search(r"source_path\s*=\s*'([^']+)'", txt)
            if m:
                name = os.path.basename(os.path.dirname(m.group(1).replace("\\", "/")))
                cand = os.path.join(root, name)
                if os.path.isdir(os.path.join(cand, "colmap_0")):
                    return cand
        except Exception:                                        # noqa: BLE001
            pass
    return None


def scan(root):
    """Scan a shooting root. Returns (list of datasets, list of models).

    Each model: dict(name, dir, iters=[(n, ply)], dataset, mtime)
    """
    root = os.path.abspath(root)
    datasets = sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, "colmap_0", "sparse", "0")))

    models = []
    for d in sorted(os.listdir(root)):
        md = os.path.join(root, d)
        if not os.path.isdir(md):
            continue
        its = _iters(md)
        if not its:
            continue
        ds = _dataset_of(md, root)
        if ds is None and len(datasets) == 1:
            ds = datasets[0]           # only one dataset: nothing to guess
        models.append({"name": d, "dir": md, "iters": its, "dataset": ds,
                       "mtime": max(os.path.getmtime(p) for _, p in its)})
    return datasets, models


def load_player(root):
    """Read `player.json` in the shooting root. Missing -> empty dict (not an error)."""
    f = os.path.join(root, PLAYER_JSON)
    if not os.path.isfile(f):
        return {}
    try:
        return json.load(open(f, encoding="utf-8"))
    except Exception as e:                                       # noqa: BLE001
        print(f"  ! {PLAYER_JSON} cannot be read ({e}); using the automatic default")
        return {}


def resolve(path, model=None, it=None, dataset=None):
    """Resolve the path given by the user into (ply, dataset, description).

    `path` can be:
      - **a shooting root** <- recommended; reads `player.json` or picks automatically
      - a `.ply` (then `dataset` is required) - older form, kept for compatibility

    Precedence: command line > `player.json` default > automatic rules.
    """
    path = os.path.abspath(path)

    if path.lower().endswith(".ply"):
        if not dataset:
            raise SystemExit("  ERROR a PLY given directly needs the dataset folder as well")
        return path, os.path.abspath(dataset), "(PLY given directly)"

    if not os.path.isdir(path):
        raise SystemExit(f"  ERROR not found: {path}")

    datasets, models = scan(path)
    pj = load_player(path)
    dflt = pj.get("default") or {}
    src = "command line" if (model or it) else ("player.json" if dflt else "automatic")
    model = model or dflt.get("model")
    it = it or dflt.get("iter")
    if not models:
        raise SystemExit(
            f"  ERROR no model found under {os.path.basename(path)}\n"
            f"    a model looks like <name>_out/point_cloud/iteration_N/point_cloud.ply\n"
            f"    to play a dataset directly, give a PLY plus the dataset folder (older form)")

    # -- choose the model
    if model:
        hit = [m for m in models if m["name"] == model] or \
              [m for m in models if m["name"].lower().startswith(model.lower())]
        if not hit:
            raise SystemExit(f"  ERROR no model {model}. Available: "
                             + ", ".join(m["name"] for m in models))
        pick = hit[0]
    elif len(models) == 1:
        pick = models[0]
    else:
        pick = max(models, key=lambda m: m["mtime"])      # most recently modified

    # -- choose the iteration
    if it:
        hit = [p for n, p in pick["iters"] if n == int(it)]
        if not hit:
            raise SystemExit(f"  ERROR {pick['name']} has no iteration_{it}. Available: "
                             + ", ".join(f"{n:,}" for n, _ in pick["iters"]))
        ply, n = hit[0], int(it)
    else:
        n, ply = pick["iters"][-1]                        # the largest

    ds = dataset or pick["dataset"]
    if ds is None:
        raise SystemExit(
            f"  ERROR cannot tell which dataset {pick['name']} was trained on"
            f" (cfg_args unreadable, and there are {len(datasets)} datasets here).\n"
            f"    Choose one with --dataset: " + ", ".join(os.path.basename(d) for d in datasets))

    # -- description for the user
    lines = [f"  shooting root {os.path.basename(path)}"]
    wn = max((len(m["name"]) for m in models), default=8)
    wd = max((len(os.path.basename(m["dataset"] or "?")) for m in models), default=6)
    for m in models:
        mark = ">" if m is pick else " "
        ii = "/".join(f"{x:,}" for x, _ in m["iters"])
        lines.append(f"   {mark} {m['name']:<{wn}}  "
                     f"{os.path.basename(m['dataset']) if m['dataset'] else '?':<{wd}}  "
                     f"iter {ii}")
    lines.append(f"  -> playing {pick['name']} @ {n:,}   <- {src}"
                 + ("" if src != "automatic" else " (most recently modified + largest iteration)"))
    return ply, os.path.abspath(ds), "\n".join(lines)


def provenance(ply, root):
    """Three lines shown by the player: shooting root / dataset / model@iteration. **All read from paths.**

    The shooting root = the common parent of dataset and model. To change a
       name, rename the folder; the player shows the new name immediately,
       with no metadata to edit.
    """
    ds = os.path.abspath(root)
    p = os.path.abspath(ply)

    it = os.path.basename(os.path.dirname(p))                 # iteration_30000
    outdir = os.path.dirname(os.path.dirname(os.path.dirname(p)))
    mdl = os.path.basename(outdir)                            # H_out
    if it.startswith("iteration_"):
        mdl = f"{mdl} @{int(it.split('_')[1]):,}"

    return os.path.basename(os.path.dirname(ds)), os.path.basename(ds), mdl
