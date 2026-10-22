#!/usr/bin/env python3
r"""launcher.py - start menu of the player.

    python launcher.py                    open the menu (the last played root is filled in)
    python launcher.py <root>             open the menu with the path filled in

Choose a root -> list every playable model in it -> flat/VR -> scale -> play.

The "root" is **the folder that contains the model folders** (for example
`fast_out\` and `stg301\`), not a model folder itself. The field is labelled
"Root": with a label like "shooting set", people choose `fast_out\`.

    main window (this)  --play-->  player window (another process)
        ^ locked, cannot close            |
        +------- close the player --------+

The main window **stays open**; models can be chosen and played again and
again. Closing the main window ends the program.

## Why the player runs in another process

In the same process Tk's message loop would stop, and after a few seconds
Windows marks the title bar "Not responding" - the "lock" should **look like
playback**, not **look like a hang**.

Also, playing repeatedly in one process loads another copy of the model onto
the GPU each time; without explicit release, the third or fourth run runs out
of GPU memory. **When a process ends the operating system reclaims
everything**, with nothing to manage.

## Why tkinter

It is in the standard library and adds nothing to install. PyQt6 is 214 MB,
and this screen is a five-row form.

## High DPI

Windows defaults to "not DPI aware" and scales the whole window as a bitmap -
the text gets blurry. With a display scale factor of 1.333 it is clearly
visible, so DPI awareness is declared first, before any window is created.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ICON_RGB = (0x2E, 0x7D, 0xD2)      # colour of the blue dot (R,G,B)


def _dot_ico():
    """Build a 32x32 icon: transparent background + one solid dot; returns the file path.

    The bytes are assembled by hand rather than with Pillow - an icon is not
    worth a dependency.

    "No icon at all" was the goal, but Windows always draws something:
       a 1x1 transparent PhotoImage, a fully transparent 32-bit .ico and an
       invisible 1-bit .ico all turned into black squares in the title bar
       and taskbar.

    stg-studio has an identical copy. **It is deliberately not shared** -
       the player is installed independently and must not depend on the
       studio. Change both when changing this.
    """
    import struct
    path = os.path.join(tempfile.gettempdir(), "stg_dot.ico")
    if os.path.isfile(path):
        return path
    n, r = 32, 11
    b, g, rr = ICON_RGB[2], ICON_RGB[1], ICON_RGB[0]
    rows = []
    for y in range(n - 1, -1, -1):                 # DIB rows go bottom-up
        px = bytearray()
        for x in range(n):
            d = (x - (n - 1) / 2) ** 2 + (y - (n - 1) / 2) ** 2
            px += bytes((b, g, rr, 255)) if d <= r * r else b"\x00\x00\x00\x00"
        rows.append(bytes(px))
    xor = b"".join(rows)
    and_mask = b"\x00" * (n * 4)                   # a 32-bit image relies on alpha
    hdr = struct.pack("<IiiHHIIiiII", 40, n, n * 2, 1, 32, 0,
                      len(xor) + len(and_mask), 0, 0, 0, 0)
    img = hdr + xor + and_mask
    ico = struct.pack("<HHH", 0, 1, 1) + struct.pack(
        "<BBBBHHII", n, n, 0, 0, 1, 32, len(img), 22) + img
    with open(path, "wb") as f:
        f.write(ico)
    return path


def _no_icon(win):
    """Replace Tk's feather icon (UI rule: no logo). Failure is ignored; the icon is not a feature."""
    try:
        win.iconbitmap(default=_dot_ico())
    except Exception:                                         # noqa: BLE001
        pass


def _dpi_aware():
    """Must be called before any window is created. Silently skipped on other systems."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


_dpi_aware()

import captureset as CS                                       # noqa: E402
from version import PLAYER_VERSION                            # noqa: E402
from player_core import vr_status                             # noqa: E402

# Every "what is missing, can it run" decision is in `selfcheck.py`; it is
#    not repeated here (the list of modules to check is there too). It runs
#    in a subprocess; see that file for why.
#
# There is only one render() in `render_at.py`, always using ch3; whether a
#    model plays depends only on **which Python runs it** (one without ch3
#    cannot play any model), not on the model.
SCALES = ["0.25", "0.5", "0.75", "1.0"]
SCALE_FLAT, SCALE_VR = "1.0", "0.5"      # the two modes have different defaults
GREY = "#888888"
RED = "#c02020"
SELFCHECK_TIMEOUT_S = 60                 # the self-check takes 2.2 s; the timeout guards against hangs, it is not a threshold


HERE = os.path.dirname(os.path.abspath(__file__))

SETTINGS = "settings.json"


BUILD_INFO = "build_info.txt"


def build_tag():
    """The version text in the lower left of the window.

    With a `build_info.txt` next to the program, shows `v1.0.0 (a3f9c21)` -
    **a screenshot sent with a report then carries the version**.
    Without it, shows `v1.0.0 (source)`.
    """
    tag = f"v{PLAYER_VERSION}"
    f = os.path.join(settings_dir(), BUILD_INFO)
    if not os.path.isfile(f):
        return tag + " (source)"
    try:
        for line in open(f, encoding="utf-8"):
            if line.startswith("git "):
                return f"{tag} ({line.split(None, 1)[1].strip()})"
    except OSError:
        pass
    return tag


def child_cmd(kind, *args):
    """Command line of a subprocess. The only place where source and frozen builds differ.

    `kind` is `"player"` or `"selfcheck"`.

    - from source: `[python.exe, "-u", <kind>.py, ...]`
    - frozen executable: `[<exe>, "--run-<kind>", ...]`
      Frozen, `sys.executable` is the exe and appending a `.py` is
      meaningless, so the exe **calls itself again** with a flag naming the role.

    `-u` = unbuffered: when the process is killed, output still in the buffer
    is lost, and that is usually the last few lines that matter most.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, f"--run-{kind}", *args]
    return [sys.executable, "-u", os.path.join(HERE, f"{kind}.py"), *args]


def settings_dir():
    """Settings live **next to the program**; for a frozen build that is next to the exe.

    A frozen build must use `sys.executable`, **not `__file__`** - the
       bundler unpacks the sources into an internal folder that `__file__`
       points to, which is no place for user settings.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return HERE


def load_last_root():
    """The last played shooting root. None if missing, unreadable or no longer existing.

    **A missing path is not an error**: the folder may have been renamed or
       moved; then there is simply no memory and the user chooses again. No message.
    """
    try:
        with open(os.path.join(settings_dir(), SETTINGS), encoding="utf-8") as f:
            d = json.load(f)
        p = d.get("last_root")
    except Exception:                                        # noqa: BLE001
        return None
    return p if p and os.path.isdir(p) else None


def save_last_root(path):
    """Remember this shooting root. **If it cannot be written, skip silently.**

    Failing to write is not a bug but the environment (installed under
       Program Files, run from inside an archive, a read-only disk, group
       policy). The player still works fully; the user just chooses again
       each time - blocking them with an error window for a convenience
       feature would be backwards.
    """
    try:
        with open(os.path.join(settings_dir(), SETTINGS), "w",
                  encoding="utf-8") as f:
            json.dump({"last_root": os.path.abspath(path)}, f,
                      ensure_ascii=False, indent=2)
    except Exception:                                        # noqa: BLE001
        pass


class Launcher:
    def __init__(self, preset=None):
        self.proc = None                 # the player subprocess while playing
        self.log = None
        self.log_path = ""

        self.root = tk.Tk()
        # UI rule: no logo; the program's own name as the title
        self.root.title("STG player")
        _no_icon(self.root)          # default=... also applies to every window opened later
        try:
            ttk.Style().theme_use("vista")     # looks like a native Windows program
        except Exception:
            pass
        self.root.resizable(False, False)
        # the X in the top right ends the whole program (not only this window)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        f = ttk.Frame(self.root, padding=14)
        f.grid(sticky="nsew")
        f.columnconfigure(1, weight=1)

        # The field is "Root", not "shooting set": with the latter, people
        #    choose a **model** folder such as `fast_out\` - the right choice
        #    is **one level up**.
        ttk.Label(f, text="Root").grid(row=0, column=0, sticky="w", pady=4)
        self.path = tk.StringVar(value=preset or "")
        self.path_ent = ttk.Entry(f, textvariable=self.path, width=46)
        self.path_ent.grid(row=0, column=1, sticky="we", padx=6)
        self.pick_btn = ttk.Button(f, text="Browse...", command=self._pick)
        self.pick_btn.grid(row=0, column=2)

        ttk.Label(f, text="Model").grid(row=1, column=0, sticky="w", pady=4)
        self.model = ttk.Combobox(f, state="readonly", width=44)
        self.model.grid(row=1, column=1, columnspan=2, sticky="we", padx=6)

        ttk.Label(f, text="Mode").grid(row=2, column=0, sticky="w", pady=8)
        box = ttk.Frame(f)
        box.grid(row=2, column=1, columnspan=2, sticky="w", padx=6)
        self.mode = tk.StringVar(value="flat")
        self.flat_rb = ttk.Radiobutton(box, text="Flat", variable=self.mode,
                                       value="flat", command=self._mode_changed)
        self.flat_rb.pack(side="left")
        self.vr_rb = ttk.Radiobutton(box, text="VR", variable=self.mode,
                                     value="vr", command=self._mode_changed)
        self.vr_rb.pack(side="left", padx=(16, 0))

        # Without VR the option is greyed out - but **not hidden**, and the
        # reason is written next to it. Greyed out alone only raises "why can't I click it".
        self.vr_ok, why = vr_status()
        # The label always exists (not only on failure): after the
        # self-check it may need rewriting (VR packages failed to load).
        self.vr_why = ttk.Label(box, text="" if self.vr_ok else why,
                                foreground=GREY)
        self.vr_why.pack(side="left", padx=8)
        if not self.vr_ok:
            self.vr_rb.state(["disabled"])

        ttk.Label(f, text="Scale").grid(row=3, column=0, sticky="w", pady=4)
        self.scale = ttk.Combobox(f, values=SCALES, state="readonly", width=8)
        self.scale.set(SCALE_FLAT)
        self.scale.grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(f, text=f"default: flat {SCALE_FLAT} / VR {SCALE_VR}",
                  foreground=GREY).grid(row=3, column=1, sticky="e")

        ttk.Label(f, text="GPU").grid(row=4, column=0, sticky="w", pady=4)
        self.gpu_lbl = ttk.Label(f, text="", foreground=GREY)
        self.gpu_lbl.grid(row=4, column=1, columnspan=2, sticky="w", padx=6)

        ttk.Separator(f).grid(row=5, column=0, columnspan=3, sticky="we",
                              pady=10)
        self.msg = ttk.Label(f, text="", foreground=GREY)
        self.msg.grid(row=6, column=0, columnspan=2, sticky="w")
        # the version is visible - a screenshot sent with a report carries it
        ttk.Label(f, text=build_tag(), foreground=GREY).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.play_btn = ttk.Button(f, text="Play", width=14,
                                   command=self._play)
        self.play_btn.grid(row=6, column=2, sticky="e")

        self.entries = []                 # (model name, iteration) in the order of the model list
        self.gpu_ok = None                # None = still checking
        self.chk = None
        if preset:
            self._rescan()
        else:
            self._set_ready(False, "Choose a root folder")
        self._start_selfcheck()

    # ------------------------------------------------------------ self-check
    def _start_selfcheck(self):
        """Check the GPU when the menu opens, **in a subprocess, without blocking the window** (see selfcheck.py).

        `import torch` is not done in this process: it creates a CUDA context
           holding hundreds of MB of GPU memory **until the menu closes**, and
           the menu stays open - that memory would be missing from the player.
        """
        self.gpu_lbl.configure(text="Checking the GPU...", foreground=GREY)
        self._set_play_enabled()
        try:
            self.chk = subprocess.Popen(
                child_cmd("selfcheck"),
                cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:                                # noqa: BLE001
            return self._selfcheck_done({
                "ok": False, "msg": "The self-check could not start", "detail": str(e)})
        self.chk_t0 = time.time()
        self._poll_selfcheck()

    def _poll_selfcheck(self):
        """Polled for the same reason as `_poll()`: waiting would stop Tk and the window would be "Not responding"."""
        if self.chk is None:
            return
        if self.chk.poll() is None:
            # If it hangs, the play button must not stay locked forever - a timeout counts as failure, with a clear message.
            if time.time() - self.chk_t0 > SELFCHECK_TIMEOUT_S:
                self.chk.kill()
                self.chk = None
                return self._selfcheck_done({
                    "ok": False, "msg": "The GPU self-check did not respond",
                    "detail": f"It did not finish within {SELFCHECK_TIMEOUT_S} s and was stopped.\n"
                              "The GPU driver may have a problem."})
            self.root.after(200, self._poll_selfcheck)
            return
        out, err = self.chk.communicate()
        self.chk = None
        try:
            # only the last line is JSON (the human-readable report goes to stderr and does not mix in)
            r = json.loads([x for x in out.splitlines() if x.strip()][-1])
        except Exception:                                     # noqa: BLE001
            r = {"ok": False, "msg": "The self-check's report could not be read",
                 "detail": (out or "") + "\n" + (err or "")}
        self._selfcheck_done(r)

    def _selfcheck_done(self, r):
        self.gpu_ok = bool(r.get("ok"))
        # If the VR packages are broken, disable VR and **state the reason** -
        # otherwise the user chooses VR, waits tens of seconds for the model,
        # and only finds the import failure with the headset on.
        vr = r.get("vr_modules")
        if vr and vr != "ok" and self.vr_ok:
            self.vr_ok = False
            self.vr_rb.state(["disabled"])
            self.mode.set("flat")
            self.vr_why.configure(text="VR packages failed to load (flat mode unaffected)")
        # UI rule: no emoji on screen; state is shown by colour
        if self.gpu_ok:
            self.gpu_lbl.configure(text=r.get("msg", ""), foreground=GREY)
        else:
            self.gpu_lbl.configure(text=r.get("msg", "Self-check failed"), foreground=RED)
            # The details go into a window **that can be copied from**, not "go find the log"
            self._show_error("This computer cannot play models",
                             r.get("msg", "") + "\n\n" + r.get("detail", ""))
        self._set_play_enabled()

    def _set_play_enabled(self):
        """The only switch of the play button: a model is chosen **and** the GPU self-check passed.

        While the self-check is still running (`gpu_ok is None`) it cannot be
           pressed either - pressing during those seconds would skip the
           self-check and make it pointless.
        """
        ok = bool(self.entries) and self.gpu_ok is True and not self.proc
        self.play_btn.state(["!disabled"] if ok else ["disabled"])

    # ---------------------------------------------------------------- actions
    def _pick(self):
        d = filedialog.askdirectory(title="Choose the root folder (it contains model folders such as fast_out)",
                                    initialdir=self.path.get() or os.getcwd())
        if d:
            self.path.set(os.path.normpath(d))
            self._rescan()

    def _rescan(self):
        """List **every** playable model and iteration in this shooting root.

        Uses `captureset.scan()` rather than walking folders here - playback
        loads through it too, so the same logic avoids "listed but not loadable".
        """
        root = self.path.get()
        if not os.path.isdir(root):
            return self._set_ready(False, "Folder not found")
        try:
            _datasets, models = CS.scan(root)
        except Exception as e:                                # noqa: BLE001
            return self._set_ready(False, f"Scanning failed: {e}")
        if not models:
            return self._set_ready(False, self._why_no_model(root))

        # newest first: usually the one just trained
        models.sort(key=lambda m: m["mtime"], reverse=True)
        self.entries, labels = [], []
        for m in models:
            for n, _ply in sorted(m["iters"], key=lambda x: -x[0]):
                self.entries.append((m["name"], n))
                labels.append(f"{m['name']} @{n:,}")
        self.model["values"] = labels
        self.model.current(0)
        self._set_ready(True, f"{len(labels)} models found")

    @staticmethod
    def _why_no_model(root):
        """When no model is found, say what happened **this time**, not just the specification.

        A message like "no playable model here (needs point_cloud/iteration_*)"
           sent people to `fast_out\\` (which does contain `point_cloud`) - but
           that is a **model** folder, and the right choice is **one level up**.
           The message led to the wrong place.

        So three cases are told apart first.
        """
        if os.path.isdir(os.path.join(root, "point_cloud")):
            up = os.path.basename(os.path.dirname(root)) or "the parent"
            return (f"This is a model folder, not the root - choose one level up ({up})")

        # if the parent has models, a dataset (stg301) or another subfolder was probably chosen
        parent = os.path.dirname(root)
        try:
            if parent and parent != root and CS.scan(parent)[1]:
                return (f"No models here, but the parent ({os.path.basename(parent)}) has some - "
                        "choose one level up")
        except Exception:                                     # noqa: BLE001
            pass

        return ("No playable model under this root. "
                "The root must directly contain model folders (such as fast_out) and the dataset")

    def _mode_changed(self):
        """Switching mode sets the scale to that mode's default."""
        self.scale.set(SCALE_VR if self.mode.get() == "vr" else SCALE_FLAT)

    def _set_ready(self, ok, why=""):
        """Result of scanning the shooting root. **The play button is not enabled here** -
        it also depends on the self-check, so only `_set_play_enabled()` can enable it."""
        self.msg.configure(text=why)
        if not ok:
            self.entries = []
            self.model.set("")
            self.model["values"] = []
        self._set_play_enabled()

    # ------------------------------------------------------------ playback
    def _play(self):
        i = self.model.current()
        if i < 0 or i >= len(self.entries) or self.proc:
            return
        name, it = self.entries[i]
        cmd = child_cmd("player", self.path.get(), "--model", name,
                        "--iter", str(it), "--scale", self.scale.get())
        if self.mode.get() == "vr":
            cmd.append("--vr")
        # The subprocess output must be captured. Started from a shortcut
        # (pythonw) there is no console, and without capture everything the
        # player prints is lost - on failure the user would only see "nothing happened".
        self.log_path = os.path.join(tempfile.gettempdir(), "stg_player.log")
        try:
            self.log = open(self.log_path, "w", encoding="utf-8",
                            errors="replace")
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=self.log,
                                         stderr=subprocess.STDOUT)
        except Exception as e:                                # noqa: BLE001
            messagebox.showerror("Cannot start", str(e))
            return
        # Remember the path only after the subprocess **really started**. If
        #    opening the log or Popen fails, do not remember it - remembering
        #    a path that does not start only fails again next time.
        save_last_root(self.path.get())
        self._lock(True)
        self._poll()

    def _poll(self):
        """Check every 0.2 s whether the player has ended.

        Polling instead of waiting - waiting would stop Tk's message loop and
        Windows would mark the main window "Not responding", exactly what the
        subprocess is meant to avoid.
        """
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.root.after(200, self._poll)
            return
        rc = self.proc.returncode
        self.proc = None
        try:
            self.log.close()
        except Exception:
            pass
        self._lock(False)
        if rc:
            # On failure, **put the reason in front of the user**, not "go find the log".
            self.msg.configure(text=f"The player did not exit normally (exit code {rc})")
            self._show_error(f"Playback failed (exit code {rc})", self._log_tail(200))

    def _show_error(self, title, body):
        """An error window whose text **can be selected and copied**.

        The text of a `messagebox` cannot be selected or copied - and when
        something fails, what the user most needs is to paste the message to
        someone else.
        """
        w = tk.Toplevel(self.root)
        w.title(title)
        w.transient(self.root)
        fr = ttk.Frame(w, padding=10)
        fr.pack(fill="both", expand=True)
        txt = tk.Text(fr, width=92, height=20, wrap="none")
        ys = ttk.Scrollbar(fr, orient="vertical", command=txt.yview)
        xs = ttk.Scrollbar(fr, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="we")
        fr.rowconfigure(0, weight=1)
        fr.columnconfigure(0, weight=1)
        txt.insert("1.0", body)
        txt.see("end")                    # errors are usually at the end

        bar = ttk.Frame(fr)
        bar.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8, 0))

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(body)
            copied.configure(text="Copied to the clipboard")

        ttk.Button(bar, text="Copy all", command=copy).pack(side="left")
        copied = ttk.Label(bar, text="", foreground=GREY)
        copied.pack(side="left", padx=10)
        ttk.Button(bar, text="Close", command=w.destroy).pack(side="right")
        # the text stays selectable; read-only by swallowing keys, not state=disabled
        txt.bind("<Key>", lambda e: "break" if e.state & 4 == 0 else None)
        w.grab_set()

    def _log_tail(self, n=18):
        """The last lines of the log. Long logs lose their beginning - errors are usually at the end."""
        try:
            lines = open(self.log_path, encoding="utf-8",
                         errors="replace").read().splitlines()
        except OSError:
            return "(no output was left)"
        tail = [x for x in lines if x.strip()][-n:]
        return "\n".join(tail) or "(no output was left)"

    def _lock(self, playing):
        """Lock the whole main window while playing - but **it must look like playback**, not a hang."""
        st = ["disabled"] if playing else ["!disabled"]
        for w in (self.path_ent, self.pick_btn, self.model, self.scale,
                  self.flat_rb):
            w.state(st)
        # if VR is not ready, unlocking must **not** make it selectable
        self.vr_rb.state(st if self.vr_ok else ["disabled"])
        # the play button is not in the list above: unlocking it also depends on the self-check
        self._set_play_enabled()
        self.msg.configure(
            text="Playing... close the player window to return" if playing
            else f"{len(self.entries)} models found")

    def _quit(self):
        # The main window cannot be closed while playing.
        # Close the player window first to come back.
        if self.proc:
            self.root.bell()
            return
        # Closing while the self-check runs would orphan that subprocess (it
        # imports torch and holds GPU memory), and `after` callbacks would hit
        # a destroyed window. **Clean up first.**
        if self.chk is not None:
            try:
                self.chk.kill()
            except Exception:                                 # noqa: BLE001
                pass
            self.chk = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # command line > last played (the command line is the current intent; memory is only a default)
    preset = argv[0] if argv and os.path.isdir(argv[0]) else load_last_root()
    Launcher(preset).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
