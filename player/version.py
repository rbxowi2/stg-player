"""version.py - **single source of the player version.**

The launcher, the self-check and the player all read this constant. Documents
that mention the version point here instead of repeating it.

**This module must not import anything.** The launcher and the self-check
only need a version number and should not pull in torch for it
(`player_core` imports `render_at`, which costs 2 seconds and hundreds of MB).

    1.1.0   plays models from combined training: when window.json has
            segments, each frame shows only the Gaussians of the current
            segment (cross-fade at segment boundaries). Models without
            segments behave exactly as before.
    1.0.2   fixed audio sounding "stepped, electronic": the playhead chased a
            frame time that only advanced every 26-100 ms, so the rate jittered
            by up to 13.9%. Now the picture follows the audio (audio is the
            master clock).
    1.0.1   launcher error messages no longer point into the model folder;
            the field is called "root"
    1.0.0   first release: launcher, remembered paths, start-up self-check,
            ch3 rasterizer for several GPU architectures (sm_75-120 + PTX)

Versioning:

    major 1.x.x   user-visible major change, or old data no longer usable
    minor x.1.x   new feature
    patch x.x.1   bug fix or adjustment without behaviour change
"""

PLAYER_VERSION = "1.1.0"
