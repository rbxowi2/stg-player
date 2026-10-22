#!/usr/bin/env python3
"""audio4d.py - audio for the 4DGS player (shared by flat and VR)

    from audio4d import Audio
    a = Audio(dataset_root, unit_mm)        # without audio every method of the object is a no-op
    a.start()
    per frame: a.update(speed, playing, listener_pos=None, listener_R=None)
               t = a.now_t()      # audio is the master clock; the picture asks it where playback is
    at the end: a.stop()


## Where the data comes from

The recorder stores `audio/audio.wav` + `audio/index.json`; preprocessing
copies them unchanged into the dataset and adds an `audio` block to
`window.json`:

    "audio": { "path": "audio/audio.wav", "t0_offset_s": 0.312 }

`t0_offset_s` is **how many seconds the audio start differs from t=0 (the
first time step)**. The player's `t` is normalized time 0..1; in audio
seconds:

    audio seconds = t x duration_s + t0_offset_s

Speed changes, seeking, pausing and looping then line up automatically.


## Audio is the master clock; the picture asks it

**Audio runs freely; whenever the picture draws a frame it calls `now_t()` to
ask "where are we now".** This is what VLC / mpv / browsers do (mpv's default
is literally `video-sync=audio`), because the cost of correcting each side
differs enormously: a frame drawn slightly late is barely noticeable, while
audio **has no "skip a bit" option** - it is always audible.

4DGS makes this even easier than a video player: **the picture is a function
of time** (`render(cam, t=any real number)`), so no frames need to be dropped
or repeated; it can always be drawn at exactly the instant the audio gives.

This used to be the other way round: every 10.7 ms the audio pulled its
playhead 5% toward the picture time, which only advanced every 26-100 ms -
chasing a signal 3-10x coarser than itself made the playhead rate jitter.
Measured at 10 fps the rate swung between **0.72 and 1.25**, more than +-5%
off 78.7% of the time, heard as a "stepped, electronic" graininess.
Professional players tolerate rate adjustments of **0.1%**.

**Without audio** (no file, no sounddevice, device unavailable, failure
during playback) `ok=False` and the picture follows the wall clock - audio is
always optional.


## Speed change = pitch change (by design)

The playhead advances `speed / sample_rate` seconds per output sample, with
linear interpolation. `speed=2` is double speed and double pitch, like a tape
fast-forward. It is the simplest and most intuitive approach and needs no
time-stretching (a separate project, with artifacts).

Linear interpolation aliases at high speed. The limit is 8x; high
frequencies get slightly metallic - a known and accepted trade-off.


## World-locked virtual stereo

**The window is fixed and the sound comes in from outside it.** When the
viewer moves or turns in front of the window, the sound field outside
**should not turn with them** - a direct consequence of the "window"
definition.

Method: the left and right channels are treated as two **virtual speakers
fixed on the window plane**, at the real positions of the rig's two
microphones. Every frame recomputes the gain and delay of those two speakers
relative to the listener's ears.

    for each (speaker s, ear e):
        distance   d = |speaker position - ear position|
        delay      tau = d / speed of sound        <- ITD, interaural time difference
        gain       g = 1/d x head shadow           <- ILD, interaural level difference

    output[e] = sum_s  g[s][e] x source[s](playhead - tau[s][e])

This is basic ITD + ILD, **not HRTF**. HRTF needs a measured dataset
(spectral colouring by the outer ear) and resolves up/down and front/back.
ITD+ILD only gives left/right, but **left/right is all a window really
needs** - the sound always comes from the direction of the window, never from
above or behind.

> **This is not fake spatial audio.** It honestly expresses "the sound comes
> from the window" without pretending to localize every source outside -
> two microphones cannot record that; it needs at least four (first-order
> ambisonics).

The flat player uses the same model (listener = virtual camera), so **moving
left and right in front of the window with the keyboard shifts the audio
balance**. That is another check of the geometry: if the sound direction
disagrees with the picture, `unit_mm` or the camera positions are wrong.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

# -- constants ----------------------------------------------------------
SOUND_MM_S = 343000.0      # speed of sound 343 m/s in mm/s (scene units are converted via unit_mm)
EAR_MM = 75.0              # ears at +-75 mm from the head centre (adult ear distance about 150 mm)
SHADOW = 0.65              # head shadow: attenuation fully on the far side (0 = fully blocked, 1 = none)
MIN_MM = 60.0              # lower bound for distance attenuation, so gain does not explode next to a speaker
SMOOTH = 0.15              # per-block smoothing of gain/delay, against zipper noise while moving
BLOCK = 512                # audio block size; about 10.7 ms at 48k


def _find_audio(meta, root, mpath):
    """Returns (wav path, t0_offset_s) or (None, 0.0).

    Only the `audio` block of window.json is used; **folders are not
       scanned** - consistent with "the metadata decides" (as in prep_4d.py).
    """
    if not meta or "audio" not in meta:
        return None, 0.0
    a = meta["audio"]
    rel = a.get("path", "audio/audio.wav")
    bases = [os.path.abspath(root)]
    if mpath:
        bases.insert(0, os.path.dirname(os.path.abspath(mpath)))
    for b in bases:
        p = os.path.join(b, rel)
        if os.path.isfile(p):
            return p, float(a.get("t0_offset_s", 0.0))
    print(f"  ERROR window.json names the audio file {rel}, but it does not exist")
    return None, 0.0


def _read_wav(path):
    """Read WAV -> float32 [N, 2], range +-1.

    Only the standard library's `wave`, no soundfile - the format written by
    the recorder is plain uncompressed PCM WAV (S32_LE / S16_LE / S24), so
    another dependency is not needed.
    """
    import wave
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 4:            # S32_LE - the ICS-43434 has 24 significant bits, low 8 bits zero
        a = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif sw == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 3:          # rare packed 24 bit
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        a = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
              | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
             ).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width {sw} bytes")
    a = a.reshape(-1, ch)
    if ch == 1:
        a = np.repeat(a, 2, axis=1)
    return np.ascontiguousarray(a[:, :2]), sr


class Audio:
    """**Audio is always optional and must never block the player from starting.**

    No audio file, no sounddevice, the device busy, a broken WAV - each of
    these only sets `self.ok=False`; every method becomes a no-op and the
    player plays silently. This class **never raises**.

    Everything that can fail happens in `_build()`; `__init__` only catches.
    """

    def __init__(self, meta, root, mpath, unit_mm, duration_s, mic_mm=None,
                 win_ctr=None, win_right=None, win_fwd=None):
        self.ok = False
        self.info = "no audio"
        self.stream = None
        self.sd = None
        try:
            self._build(meta, root, mpath, unit_mm, duration_s, mic_mm,
                        win_ctr, win_right, win_fwd)
        except Exception as e:                                   # noqa: BLE001
            # last resort: each step above handles its own errors; this catches the unexpected
            self.ok = False
            self.info = f"ERROR audio initialization failed (playback unaffected): {e}"
            print(f"  {self.info}")

    def _build(self, meta, root, mpath, unit_mm, duration_s, mic_mm,
               win_ctr, win_right, win_fwd):
        path, self.t0 = _find_audio(meta, root, mpath)
        if path is None:
            return
        try:
            import sounddevice as sd
        except ImportError:
            self.info = "ERROR sounddevice not installed; playing silently (pip install sounddevice)"
            print(f"  {self.info}")
            return

        self.sd = sd
        try:
            self.buf, self.sr = _read_wav(path)
        except Exception as e:                                   # noqa: BLE001
            self.info = f"ERROR reading the audio failed: {e}"
            print(f"  {self.info}")
            return

        # Apply audio.gain_db from window.json.
        #
        #    I2S MEMS microphones deliver linear PCM with full scale reserved
        #    for 120+ dB SPL, so normal speech is about -60 LUFS - **without
        #    gain it is inaudible even at full volume** (37 dB short).
        #    Preprocessing computes the gain with EBU R128 and writes it into
        #    window.json (the recorder stores only raw).
        #
        #    It is multiplied into the buffer, not tied to the volume keys:
        #       volume is the user's preference, gain is a property of this
        #       material, and the two should not be mixed.
        g_db = float((meta.get("audio") or {}).get("gain_db") or 0.0)
        if g_db:
            self.buf = self.buf * (10.0 ** (g_db / 20.0))
        self.gain_db = g_db

        self.n = self.buf.shape[0]
        self.len_s = self.n / self.sr
        self.u = float(unit_mm)
        self.dur = float(duration_s)

        # -- scene positions of the two virtual speakers -----------------
        # position = window centre +- (microphone spacing / 2) along the window's "right".
        # The spacing comes from audio.mic_spacing_mm in window.json (written by the recorder).
        # The fallback 200.0 is the spacing of one particular rig and goes
        #    stale with another rig; the real value always comes from window.json.
        sp = float((meta.get("audio") or {}).get("mic_spacing_mm", mic_mm or 200.0))
        self.spk = None
        if win_ctr is not None and win_right is not None:
            h = (sp / 2.0) / self.u
            self.spk = [np.asarray(win_ctr) - np.asarray(win_right) * h,     # left
                        np.asarray(win_ctr) + np.asarray(win_right) * h]     # right
        self.mic_mm = sp
        self.fwd = None if win_fwd is None else np.asarray(win_fwd)

        # -- playback state (shared by the audio and main threads, guarded by a lock) --
        self.lock = threading.Lock()
        # Playhead in seconds within the audio file. The start is the clip
        # start (t=0 corresponds to t0), not second 0 of the file - recording
        # usually starts a little before the video.
        # The output latency must be added: now_t() returns "the audible
        #    position" = head - lat; without it the first 90 ms report as the
        #    clip end (t~0.995). `lat` is only known once start() has asked the
        #    device, so start() adds it again.
        self.head = self.t0
        self._cb_at = None          # time of the latest mix (for interpolation in now_t)
        self.block_s = BLOCK / float(self.sr)
        self.speed = 1.0
        self.playing = True
        self.vol = 0.7
        self.muted = False
        self.spatial = self.spk is not None
        # gain and delay per (speaker, ear); initially pass-through
        self.gain = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
        self.delay = np.zeros((2, 2), np.float32)
        self.stream = None
        self.lat = 0.0             # output latency (s), reported by the device in start()
        self.ok = True
        self.info = (f"{self.len_s:.2f}s / {self.sr} Hz / "
                     f"{'stereo' if self.buf.shape[1] == 2 else 'mono'}")
        print(f"  audio {os.path.relpath(path, root)}  {self.info}")
        print(f"       t0_offset {self.t0:+.3f}s   virtual speaker spacing {sp:.0f} mm"
              f"{'' if self.spatial else '   ! no window geometry; plain stereo'}")

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if not self.ok or self.stream is not None:
            return
        try:
            self.stream = self.sd.OutputStream(
                samplerate=self.sr, channels=2, dtype="float32",
                blocksize=BLOCK, callback=self._cb)
            self.stream.start()
            # The real output latency differs a lot between devices (WASAPI
            #    shared mode is often 90-180 ms), so the device's value is used.
            self.lat = float(getattr(self.stream, "latency", 0.0) or 0.0)
            # with the latency known, adjust the start, or the first loop would start from the clip end
            with self.lock:
                if abs(self.head - self.t0) < 1e-9:
                    self.head = self.t0 + self.lat * self.speed
            print(f"       output latency {self.lat*1000:.0f} ms (compensated)")
        except Exception as e:                                   # noqa: BLE001
            self.ok = False
            self.info = f"ERROR opening the audio device failed: {e}"
            print(f"  {self.info}")

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:                                    # noqa: BLE001
                pass
            self.stream = None

    # ------------------------------------------------------------ per frame
    def now_t(self):
        """Where playback is now (normalized time 0..1). **The picture asks this for every frame.**

        Returns **the instant being heard**, not the instant generated:
           `head` is how far audio has been generated, and the speakers lag by
           one output buffer (`self.lat`, measured 90-192 ms). The picture
           must match the ears, not the buffer's write position.

        `head` only jumps once per block (10.7 ms); using it directly would
           make the picture stutter at that granularity. So **between two
           callbacks, the elapsed time is interpolated** - audio is still the
           master clock (each callback recalibrates); this only turns the
           steps into a ramp, as mpv does. The interpolation is capped at one
           block, so it cannot run away if callbacks stop.
        """
        with self.lock:
            head, speed, since = self.head, self.speed, self._cb_at
        if since is not None and self.playing:
            gap = min(max(time.perf_counter() - since, 0.0), self.block_s)
            head += gap * speed
        return (((head - self.lat * speed) - self.t0) / self.dur) % 1.0

    def seek_t(self, t_norm):
        """Put the playhead at a time (R key reset, seeking).

        The picture side must **not change `t` on its own** - the audio would
        not know and the two would drift apart.
        """
        with self.lock:
            self.head = self._wrap(float(t_norm) * self.dur + self.t0
                                   + self.lat * self.speed)

    def set_speed(self, speed):
        """Playback speed: only changes the playhead step. The picture follows via `now_t()`."""
        with self.lock:
            self.speed = float(speed)

    def _wrap_arr(self, sec):
        """Vector version of `_wrap` (the sample positions of a whole block)."""
        return self.t0 + np.mod(sec - self.t0, self.dur)

    def _wrap(self, sec):
        """Wrap the playhead into [t0, t0 + dur).

        **Wraps on the clip length, not the file length**: the audio file is
           usually longer than the clip (measured 10.125 vs 9.5414 s; recording
           ends after the video), and wrapping on the file length would play
           those extra 0.6 s at every loop seam.
        """
        return self.t0 + (sec - self.t0) % self.dur

    def update(self, speed, playing, pos=None, R=None):
        """Called once per frame by the picture side. `pos`/`R` are the listener position and orientation (world -> camera rotation).

        This runs inside the render loop and **must never raise and interrupt the player**.
        """
        if not self.ok:
            return
        try:
            self._update(speed, playing, pos, R)
        except Exception as e:                                   # noqa: BLE001
            self.ok = False
            self.info = f"ERROR audio disabled: {e}"
            print(f"  {self.info}")
            self.stop()

    def _update(self, speed, playing, pos, R):
        g, d = self._pan(pos, R)
        with self.lock:
            self.speed, self.playing = float(speed), playing
            if g is not None:
                # smoothing: a sudden gain jump when turning the head is heard as zipper noise
                self.gain += (g - self.gain) * SMOOTH
                self.delay += (d - self.delay) * SMOOTH

    def _pan(self, pos, R):
        """Gain and delay per (speaker, ear). Returns (2x2, 2x2) or (None, None)."""
        if not self.spatial or self.muted or pos is None or R is None:
            return None, None
        pos = np.asarray(pos, np.float64)
        right, fwd = R[0], R[2]                 # camera right and forward (world coordinates)
        ear = [pos - right * (EAR_MM / self.u), pos + right * (EAR_MM / self.u)]
        g = np.zeros((2, 2), np.float32)
        d = np.zeros((2, 2), np.float32)
        for s in range(2):
            for e in range(2):
                v = self.spk[s] - ear[e]
                dist_mm = max(float(np.linalg.norm(v)) * self.u, MIN_MM)
                d[s, e] = dist_mm / SOUND_MM_S
                # head shadow: attenuate when the speaker is on the far side of this ear.
                # cos>0 = same side (not blocked by the head), cos<0 = far side.
                side = 1.0 if e == 1 else -1.0
                cos = float(np.dot(v / max(np.linalg.norm(v), 1e-9), right)) * side
                shade = SHADOW + (1.0 - SHADOW) * (0.5 + 0.5 * cos)
                g[s, e] = (MIN_MM / dist_mm) * shade
        # normalize: total level = 1 straight in front of the window centre; moving only shifts the balance
        g *= 1.0 / max(g.sum(axis=0).mean(), 1e-6)
        return g * self.vol, d

    # ------------------------------------------------------------ audio thread
    def _cb(self, out, frames, time_info, status):               # noqa: ARG002
        """Audio thread. An exception here would make sounddevice abort the
        stream, so it is always swallowed and silence is output - better no
        sound than any effect on the picture."""
        try:
            self._mix(out, frames)
        except Exception:                                        # noqa: BLE001
            out[:] = 0.0

    def _mix(self, out, frames):
        """The playhead is **advanced only by the sample clock** and chases nothing.

        This is the source of the whole time base: each block advances exactly
        `frames * speed / sr` seconds, which is the time the sound card takes
        to consume it. The picture reads it through `now_t()`.

        Pulling `head` toward the picture time used to happen here; that was
        the source of the graininess (see the module docstring).
        """
        with self.lock:
            head, speed, playing = self.head, self.speed, self.playing
            gain, delay = self.gain.copy(), self.delay.copy()
            muted, vol = self.muted, self.vol

        step = speed / self.sr
        with self.lock:
            self._cb_at = time.perf_counter()      # when this block was handed over

        if not playing:
            out[:] = 0.0            # paused: time stops too, and the picture stays put
            return

        if muted:
            # Muted only means silent; **the clock keeps running** - if the
            #    playhead stopped, the picture would freeze (it gets its time from now_t()).
            out[:] = 0.0
            with self.lock:
                self.head = self._wrap(head + frames * step)
            return

        # playhead positions (seconds in the audio file). Linear interpolation -> speed change = pitch change.
        pos_s = self._wrap_arr(head + np.arange(frames, dtype=np.float64) * step)

        src = np.empty((frames, 2), np.float32)
        for c in range(2):
            src[:, c] = self._tap(pos_s, c)

        # world-locked mix: each ear receives both speakers, each with its own delay
        if self.spatial and not np.allclose(delay, 0):
            for e in range(2):
                acc = np.zeros(frames, np.float32)
                for s in range(2):
                    acc += gain[s, e] * self._tap(pos_s - delay[s, e], s)
                out[:, e] = acc
        else:
            out[:, 0] = src[:, 0] * vol
            out[:, 1] = src[:, 1] * vol

        np.clip(out, -1.0, 1.0, out)
        with self.lock:
            self.head = self._wrap(head + frames * step)

    def _tap(self, pos_s, ch):
        """Sample at any position in seconds (fractional, possibly out of range), linear interpolation + looping."""
        x = np.mod(pos_s * self.sr, self.n)
        i0 = x.astype(np.int64)
        f = (x - i0).astype(np.float32)
        i1 = (i0 + 1) % self.n
        b = self.buf[:, ch]
        return b[i0] * (1.0 - f) + b[i1] * f

    # ------------------------------------------------------------ keyboard
    def toggle_mute(self):
        if self.ok:
            self.muted = not self.muted

    def toggle_spatial(self):
        """Diagnostic: switch between "world-locked" and "plain stereo".
        The difference while turning the head is what spatialization actually does."""
        if self.ok and self.spk is not None:
            self.spatial = not self.spatial

    def vol_step(self, k):
        if self.ok:
            self.vol = float(np.clip(self.vol * k, 0.02, 2.0))

    def hud(self):
        if not self.ok:
            return "off"
        if self.muted:
            return "MUTED"
        return f"{self.vol:.2f} {'world' if self.spatial else 'plain'}"
