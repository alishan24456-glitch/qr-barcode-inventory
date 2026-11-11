#!/usr/bin/env python3
"""
QR / Barcode Inventory -- scan it, count it
================================================

Point the camera (or a video file) at QR codes and EAN-13 barcodes and
the script builds a live inventory: every code it actually decodes is
logged with its type, first-seen time, last-seen time and how many
separate sightings it had. Nothing is counted from a label, a filename
or a product list -- only codes a real decoder read from the pixels.

One decoder reads both code families -- ZXing (``zxing-cpp``) -- because
in testing it was the only one that read everything: OpenCV's own
``barcode`` detector could not read clean synthetic EAN-13 barcodes at
all, and OpenCV's ``QRCodeDetector`` (``detectAndDecodeMulti``) silently
missed a valid, legible QR code whenever a second QR code was in the
same frame, while ZXing found every code in the frame. The inventory
only ever counts what ZXing decoded from the pixels.

A "sighting" is one continuous pass of a code through the frame: a code
seen again after more than 2 seconds away counts as a new sighting
(the product came round again), otherwise it extends the same one.

Simulation mode (``--simulate``) renders a conveyor belt with six
labelled product boxes rolling past. Each label carries a real QR code
(generated with the ``qrcode`` library) and/or a real EAN-13 barcode
(drawn by the encoder in this file -- the same encoder verified
against ZXing's own writer). The inventory only ever counts what the
decoders read from the rendered pixels, exactly as in real mode.
Simulation is an internal logic check; the delivered demo is real
footage (see README).

Examples
--------
    python qr_barcode_inventory.py                 (opens a file picker)
    python qr_barcode_inventory.py --source pick   (same picker, explicit)
    python qr_barcode_inventory.py --source shelf.mp4
    python qr_barcode_inventory.py --source 0      (webcam)
    python qr_barcode_inventory.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the final
inventory table to ``output_inventory.txt`` and the timestamped event
log to ``output_events.log`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 168                     # inventory panel at the bottom
VIDEO_H = H - PANEL_H             # video area below the title bar
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (80, 200, 90)             # QR boxes
CYAN = (220, 180, 60)             # 1D barcode boxes
ORANGE = (60, 140, 255)
SIGHTING_GAP = 2.0                # seconds away = a new sighting

# ---------------------------------------------------------------------------
# EAN-13 encoder (draws a real, decodable barcode)
# ---------------------------------------------------------------------------
_L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001",
      "0101111", "0111011", "0110111", "0001011"]
_G = ["0100111", "0110011", "0011011", "0100001", "0011101", "0111001",
      "0000101", "0010001", "0001001", "0010111"]
_R = ["1110010", "1100110", "1101100", "1000010", "1011100", "1001110",
      "1010000", "1000100", "1001000", "1110100"]
# first digit -> parity pattern of the six left-hand digits.
# Verified empirically against ZXing's own EAN-13 writer (digit 4 is
# LGLLGG -- several published tables print LGLGGG, which is wrong).
_PAR = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
        "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def ean13_check(d12):
    """Check digit for a 12-digit EAN base (positions from the right)."""
    s = sum(int(d) * (3 if (len(d12) - i) % 2 == 1 else 1)
            for i, d in enumerate(d12))
    return (10 - s % 10) % 10


def ean13_bits(code12):
    """Return (95-module bit string, full 13-digit code)."""
    code12 = str(code12)
    assert len(code12) == 12 and code12.isdigit(), code12
    full = code12 + str(ean13_check(code12))
    par = _PAR[int(full[0])]
    bits = "101"
    for i, ch in enumerate(full[1:7]):
        bits += (_L if par[i] == "L" else _G)[int(ch)]
    bits += "01010"
    for ch in full[7:13]:
        bits += _R[int(ch)]
    bits += "101"
    return bits, full


def draw_ean13(code12, module=3, height=64, quiet=10, digits=True):
    """Draw an EAN-13 barcode (BGR image, white background).

    With digits=True the 13-digit number is printed below the bars.
    """
    bits, full = ean13_bits(code12)
    w = (len(bits) + 2 * quiet) * module
    text_h = 24 if digits else 0
    img = np.full((height + text_h, w, 3), 255, np.uint8)
    for i, b in enumerate(bits):
        if b == "1":
            x = (quiet + i) * module
            cv2.rectangle(img, (x, 0), (x + module - 1, height),
                          (0, 0, 0), -1)
    if digits:
        cv2.putText(img, full, (quiet * module, height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return img, full


# ---------------------------------------------------------------------------
# Decoder: ZXing reads QR codes and 1D barcodes from the pixels.
# ---------------------------------------------------------------------------
def detect_codes(frame):
    """Decode every QR / 1D code visible in a BGR frame.

    Returns a list of dicts: {text, format, points} where points is a
    4x2 array of corner coordinates (or None when the decoder gives no
    position). The decoder sees only the pixels.
    """
    found = []
    try:
        import zxingcpp

        for r in zxingcpp.read_barcodes(frame):
            if not r.valid or not r.text:
                continue
            fmt = "QR" if r.format == zxingcpp.BarcodeFormat.QRCode \
                else str(r.format)
            pts = None
            try:
                pos = r.position
                pts = np.array([[pos.top_left.x, pos.top_left.y],
                                [pos.top_right.x, pos.top_right.y],
                                [pos.bottom_right.x, pos.bottom_right.y],
                                [pos.bottom_left.x, pos.bottom_left.y]],
                               dtype=np.float32)
            except Exception:
                pts = None
            found.append({"text": r.text, "format": fmt, "points": pts})
    except Exception:
        pass
    return found


# ---------------------------------------------------------------------------
# Inventory: counts only what the decoders actually read
# ---------------------------------------------------------------------------
class Inventory:
    def __init__(self):
        self.items = {}           # text -> record dict
        self.events = []          # (t, message) in order

    def update(self, found, t):
        """Fold one frame's decodes into the inventory. Returns the
        event lines produced by this frame."""
        new_events = []
        for item in found:
            text, fmt = item["text"], item["format"]
            rec = self.items.get(text)
            if rec is None:
                rec = {"text": text, "format": fmt, "first": t,
                       "last": t, "sightings": 1, "frames": 1}
                self.items[text] = rec
                msg = f"NEW {fmt} {text}"
                self.events.append((t, msg))
                new_events.append((t, msg))
            else:
                rec["frames"] += 1
                if t - rec["last"] > SIGHTING_GAP:
                    rec["sightings"] += 1
                    msg = (f"RESIGHT {fmt} {text} "
                           f"(sighting #{rec['sightings']})")
                    self.events.append((t, msg))
                    new_events.append((t, msg))
                rec["last"] = t
        return new_events

    @property
    def total_sightings(self):
        return sum(r["sightings"] for r in self.items.values())

    def table_lines(self):
        lines = ["CODE                            TYPE        "
                 "SIGHTINGS  FRAMES  FIRST    LAST",
                 "-" * 76]
        for r in sorted(self.items.values(),
                        key=lambda x: x["first"]):
            lines.append(f"{r['text']:<32}{r['format']:<12}"
                         f"{r['sightings']:<11}{r['frames']:<8}"
                         f"{r['first']:>6.1f}s  {r['last']:>6.1f}s")
        return lines


# ---------------------------------------------------------------------------
# Drawing: annotated video + inventory panel
# ---------------------------------------------------------------------------
def draw_detections(frame, found):
    """Draw boxes + labels for the decoded codes, in place."""
    for item in found:
        color = GREEN if item["format"] == "QR" else CYAN
        pts = item["points"]
        if pts is not None and len(pts) == 4:
            cv2.polylines(frame, [pts.astype(np.int32)], True, color, 3)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x, y = 20, 60
        label = f"{item['format']}: {item['text']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,
                                      0.6, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1,
                    cv2.LINE_AA)


def render_frame(frame, found, inv, frame_idx, t, note):
    """Compose the 1280x720 output frame: title, video, inventory panel."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:, :] = BG
    cv2.putText(canvas, "QR / BARCODE INVENTORY", (16, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "every code counted was decoded from the pixels",
                (470, 32), cv2.FONT_HERSHEY_DUPLEX, 0.55, GREY, 1,
                cv2.LINE_AA)
    (ntw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cv2.putText(canvas, note, (W - 16 - ntw, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    # letterbox-fit the video into the video area (keeps the aspect
    # ratio -- a vertical phone clip must not be squashed flat)
    fh, fw = frame.shape[:2]
    scale = min(W / fw, VIDEO_H / fh)
    vw, vh = int(fw * scale), int(fh * scale)
    video = cv2.resize(frame, (vw, vh))
    vx, vy = (W - vw) // 2, TITLE_H + (VIDEO_H - vh) // 2
    canvas[vy:vy + vh, vx:vx + vw] = video
    # inventory panel
    y0 = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, y0), (W, H), PANEL_BG, -1)
    cv2.putText(canvas,
                f"INVENTORY  --  {len(inv.items)} unique codes  ·  "
                f"{inv.total_sightings} sightings  ·  "
                f"{frame_idx + 1} frames scanned",
                (16, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, WHITE, 1,
                cv2.LINE_AA)
    y = y0 + 54
    for line in inv.table_lines()[2:][:4]:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, WHITE, 1, cv2.LINE_AA)
        y += 26
    if len(inv.items) > 4:
        cv2.putText(canvas, f"... and {len(inv.items) - 4} more "
                    "(see inventory file)", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    if not inv.items:
        cv2.putText(canvas, "No codes decoded yet -- hold a QR code or "
                    "barcode in frame.", (16, y0 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1, cv2.LINE_AA)
    return canvas


def summary_frame(inv, note_lines):
    """End-of-run full-screen inventory table."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:, :] = BG
    cv2.putText(canvas, "INVENTORY -- FINAL COUNT", (16, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"{len(inv.items)} unique codes  ·  "
                f"{inv.total_sightings} sightings",
                (16, 78), cv2.FONT_HERSHEY_DUPLEX, 0.7, GREEN, 2,
                cv2.LINE_AA)
    y = 130
    for line in inv.table_lines():
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, WHITE, 1, cv2.LINE_AA)
        y += 30
    y += 16
    for note in note_lines:
        cv2.putText(canvas, note, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, GREY, 1, cv2.LINE_AA)
        y += 26
    return canvas


# ---------------------------------------------------------------------------
# Simulation: a conveyor belt of labelled products. The decoders see only
# the rendered pixels -- the inventory never learns a code any other way.
# ---------------------------------------------------------------------------
SIM_W, SIM_H = 1280, 720
SIM_FPS = 30.0

# (payload, label, kind, ean12-or-None) -- kind: "both", "qr", "barcode"
_SIM_PRODUCTS = [
    ("SKU-1001:WIDGET-A", "WIDGET A", "both", "590123412345"),
    ("SKU-1002:GADGET-B", "GADGET B", "qr", None),
    ("SKU-1003:TOOL-C", "TOOL C", "barcode", "400638133393"),
    ("SKU-1004:PART-D", "PART D", "both", "978020137962"),
    ("SKU-1005:BOX-E", "BOX E", "qr", None),
    ("SKU-1006:ITEM-F", "ITEM F", "barcode", "501234567890"),
]
_BOX_COLORS = [(70, 90, 190), (60, 140, 90), (170, 120, 40),
               (140, 60, 140), (50, 120, 170), (180, 80, 60)]
_BELT_Y0, _BELT_Y1 = 300, 540
_BOX_W, _BOX_H = 260, 200


def _make_qr_image(payload, px=150):
    """Render a QR code payload to a BGR numpy image."""
    import qrcode

    qr = qrcode.QRCode(box_size=max(2, px // 25), border=2)
    qr.add_data(payload)
    qr.make()
    pil_img = qr.make_image(fill_color="black",
                            back_color="white").convert("RGB")
    arr = np.asarray(pil_img)
    arr = cv2.resize(arr, (px, px), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _sim_assets():
    """Pre-render the per-product label art once (QR + barcode images)."""
    assets = []
    for payload, label, kind, ean12 in _SIM_PRODUCTS:
        art = {"label": label, "kind": kind}
        if kind in ("both", "qr"):
            art["qr"] = _make_qr_image(payload, px=88)
            art["qr_small"] = _make_qr_image(payload, px=72)
            art["payload"] = payload
        if kind in ("both", "barcode"):
            art["barcode"], art["ean"] = draw_ean13(
                ean12, module=2, height=64, quiet=8)
            art["barcode_small"], _ = draw_ean13(
                ean12, module=2, height=24, quiet=8, digits=False)
        assets.append(art)
    return assets


def _paste(img, patch, x, y):
    """Paste a patch onto img at (x, y), clipping to the canvas edge
    (boxes roll in from off-screen, so parts of them are out of frame)."""
    ph, pw = patch.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + pw, img.shape[1]), min(y + ph, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    px0, py0 = x0 - x, y0 - y
    img[y0:y1, x0:x1] = patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]


def _draw_box(img, cx, art, color):
    """One product box with its label (QR and/or barcode)."""
    x0, x1 = int(cx - _BOX_W // 2), int(cx + _BOX_W // 2)
    y0, y1 = _BELT_Y0 + 20, _BELT_Y0 + 20 + _BOX_H
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 30), 2)
    # tape stripe + label background
    cv2.rectangle(img, (x0, y0 + 8), (x1, y0 + 22), (255, 255, 255), -1)
    cv2.putText(img, art["label"], (x0 + 12, y0 + 62),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    label_x0, label_y0 = x0 + 14, y0 + 74
    label_w, label_h = _BOX_W - 28, _BOX_H - 88
    cv2.rectangle(img, (label_x0, label_y0),
                  (label_x0 + label_w, label_y0 + label_h),
                  (255, 255, 255), -1)
    if art["kind"] == "qr":
        qr = art["qr"]
        qh, qw = qr.shape[:2]
        qx = label_x0 + (label_w - qw) // 2
        _paste(img, qr, qx, label_y0 + 4)
        cv2.putText(img, art["payload"][:18],
                    (label_x0 + 6, label_y0 + label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
                    cv2.LINE_AA)
    elif art["kind"] == "barcode":
        bc = art["barcode"]
        bh, bw = bc.shape[:2]
        bx = label_x0 + (label_w - bw) // 2
        by = label_y0 + (label_h - bh) // 2
        _paste(img, bc, bx, by)
    else:  # "both": small QR top-left, barcode strip along the bottom
        _paste(img, art["qr_small"], label_x0 + 6, label_y0 + 4)
        bc = art["barcode_small"]
        bh, bw = bc.shape[:2]
        _paste(img, bc, label_x0 + 5, label_y0 + label_h - bh - 4)


def render_sim_frame(frame_idx, assets):
    """Render one simulation frame (1280x720 BGR)."""
    img = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)
    img[0:280, :] = (205, 190, 170)              # wall
    for bx, bw in ((40, 200), (300, 160), (520, 220), (800, 180),
                   (1020, 200)):
        cv2.rectangle(img, (bx, 60), (bx + bw, 280), (120, 120, 130), -1)
        cv2.putText(img, "STOCK", (bx + 18, 170),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (230, 230, 230), 2,
                    cv2.LINE_AA)
    cv2.rectangle(img, (0, 280), (SIM_W, 300), (150, 150, 150), -1)
    cv2.rectangle(img, (0, _BELT_Y0), (SIM_W, _BELT_Y1), (48, 48, 54), -1)
    for x in range(0, SIM_W, 80):                 # belt rollers
        cv2.circle(img, (x + 40, (_BELT_Y0 + _BELT_Y1) // 2), 12,
                   (90, 90, 96), -1)
        cv2.circle(img, (x + 40, (_BELT_Y0 + _BELT_Y1) // 2), 5,
                   (130, 130, 136), -1)
    cv2.rectangle(img, (0, _BELT_Y1), (SIM_W, SIM_H), (150, 150, 150), -1)
    # boxes roll left -> right, spaced 300 px apart, looping; the loop
    # is 24 s at 5 px/frame so every box crosses inside a 30 s run
    span = SIM_W + 2 * _BOX_W + 6 * 300
    for i, art in enumerate(assets):
        x = ((i * 300 + frame_idx * 5.0) % span) - _BOX_W
        if -_BOX_W < x < SIM_W + _BOX_W:
            _draw_box(img, x + _BOX_W // 2, art, _BOX_COLORS[i])
    cv2.putText(img, "SIMULATION -- rendered conveyor belt",
                (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.putText(img, "SCAN ZONE", (SIM_W // 2 - 90, _BELT_Y0 - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 215, 255), 2,
                cv2.LINE_AA)
    return img


class SimCapture:
    """A cv2.VideoCapture look-alike serving rendered simulation frames."""

    def __init__(self, total_frames, fps=SIM_FPS):
        self.total_frames = int(total_frames)
        self.fps = fps
        self.idx = 0
        self.assets = _sim_assets()

    def isOpened(self):
        return True

    def read(self):
        if self.idx >= self.total_frames:
            return False, None
        frame = render_sim_frame(self.idx, self.assets)
        self.idx += 1
        return True, frame

    def release(self):
        pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0


# ---------------------------------------------------------------------------
# Source handling: file picker / path / webcam index
# ---------------------------------------------------------------------------
def pick_file():
    """Tkinter video chooser. Returns a path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception as exc:
        print(f"File picker unavailable ({exc}). "
              "Pass a path instead: --source video.mp4")
        return None


def open_source(source_arg):
    """Resolve --source into (cv2.VideoCapture, is_webcam, label)."""
    if source_arg is None or str(source_arg).lower() == "pick":
        path = pick_file()
        if not path:
            print("No file chosen. Exiting.")
            sys.exit(0)
        source_arg = path
    if str(source_arg).isdigit():
        idx = int(source_arg)
        cap = cv2.VideoCapture(idx)
        return cap, True, f"webcam {idx}"
    cap = cv2.VideoCapture(str(source_arg))
    return cap, False, str(source_arg)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="QR / Barcode Inventory -- decode codes from video "
                    "and count them.")
    ap.add_argument("--source", default=None,
                    help="'pick' (default with no flag: file chooser), a "
                         "video path, or a webcam index such as 0")
    ap.add_argument("--output", default="output.mp4",
                    help="annotated video to write (default output.mp4)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop after this many seconds of footage (0 = all)")
    ap.add_argument("--summary-seconds", type=float, default=4.0,
                    help="seconds of final inventory table at the end")
    ap.add_argument("--no-display", action="store_true",
                    help="do not open a window (headless runs)")
    ap.add_argument("--simulate", action="store_true",
                    help="scan a rendered conveyor belt instead of a "
                         "video/webcam (internal logic check only; the "
                         "delivered demo is always real footage)")
    args = ap.parse_args()

    def reopen_source():
        if args.simulate:
            total = int((args.max_seconds or 30.0) * SIM_FPS)
            return SimCapture(total), False, \
                "simulation (rendered conveyor belt)"
        return open_source(args.source)

    if args.simulate:
        total = int((args.max_seconds or 30.0) * SIM_FPS)
        cap = SimCapture(total)
        is_webcam, label = False, "simulation (rendered conveyor belt)"
    else:
        cap, is_webcam, label = open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open source: {label}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0 or src_fps > 120:
        src_fps = 30.0
    out_fps = src_fps
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (W, H))
        if not writer.isOpened():
            print(f"Could not open output writer for {args.output}")
            return 1

    print(f"[scan] source: {label} @ {src_fps:.2f} fps")
    inv = Inventory()
    frame_idx = 0
    max_frames = int(args.max_seconds * src_fps) if args.max_seconds else None
    quit_now = False
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break
        t = frame_idx / src_fps
        found = detect_codes(frame)
        events = inv.update(found, t)
        for et, msg in events:
            print(f"[{et:6.1f}s] {msg}")
        annotated = frame.copy()
        draw_detections(annotated, found)
        note = f"frame {frame_idx + 1}  ·  {t:5.1f} s"
        canvas = render_frame(annotated, found, inv, frame_idx, t, note)
        if writer:
            writer.write(canvas)
        if not args.no_display:
            cv2.imshow("QR / Barcode Inventory", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                quit_now = True
                break
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[scan] {frame_idx} frames, "
                  f"{len(inv.items)} unique codes")
    cap.release()
    elapsed = time.perf_counter() - t_start
    print(f"[scan] done: {frame_idx} frames in {elapsed:.1f} s, "
          f"{len(inv.items)} unique codes, "
          f"{inv.total_sightings} sightings")

    notes = [
        "Every counted code was decoded from the video pixels by "
        "ZXing (QR + 1D barcodes); nothing was counted "
        "from a label or a list.",
        f"Source: {label}. A sighting is one continuous pass; a code "
        "seen again after > 2 s away counts as a new sighting.",
    ]
    if writer:
        summary = summary_frame(inv, notes)
        for _ in range(int(args.summary_seconds * out_fps)):
            writer.write(summary)
        writer.release()
        stem = os.path.splitext(args.output)[0]
        with open(stem + "_inventory.txt", "w", encoding="utf-8") as fh:
            fh.write("QR / Barcode Inventory -- final count\n")
            fh.write(f"Source: {label}\n\n")
            fh.write("\n".join(inv.table_lines()) + "\n\n")
            fh.write(f"Unique codes: {len(inv.items)}\n")
            fh.write(f"Total sightings: {inv.total_sightings}\n")
            fh.write(f"Frames scanned: {frame_idx}\n")
        with open(stem + "_events.log", "w", encoding="utf-8") as fh:
            for et, msg in inv.events:
                fh.write(f"[{et:6.1f}s] {msg}\n")
        print(f"[done] video: {args.output}")
        print(f"[done] inventory: {stem}_inventory.txt")
        print(f"[done] events: {stem}_events.log")
    if not args.no_display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
