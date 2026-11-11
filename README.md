# QR / Barcode Inventory — scan it, count it

Point the camera (or a video file) at QR codes and EAN-13 barcodes and
the script builds a live inventory: every code it actually decodes is
logged with its type, first-seen time, last-seen time and how many
separate sightings it had. Nothing is counted from a label, a filename
or a product list — only codes a real decoder read from the pixels.

A "sighting" is one continuous pass of a code through the frame: a code
seen again after more than 2 seconds away counts as a new sighting
(the product came round again), otherwise it extends the same one.

## Decoder

One decoder reads both code families — ZXing (`zxing-cpp`) — because
in testing it was the only one that read everything:

- OpenCV's own `barcode` detector could not read clean synthetic
  EAN-13 barcodes at all (verified-correct codes, confirmed against
  ZXing's own writer).
- OpenCV's `QRCodeDetector` (`detectAndDecodeMulti`) silently missed
  a valid, legible QR code whenever a second QR code was in the same
  frame, while ZXing found every code in the frame.

## Demo

**Verified run (11 Nov 2026)** — real mode on free-license Pexels
stock footage ("QR Code on a Smartphone", Leeloo The First,
1080×1920 vertical): the phone screen shows a QR code that the
pipeline decodes in 768 of 807 frames (95%), first seen at 0.0 s and
held to the end of the footage — 1 unique code, 1 sighting.

The source clip is only 9 s long, so the demo loops it 3× (26.9 s of
footage + 4 s final inventory table = 30.9 s video). The loop is the
same real footage repeated, stated here plainly — the code is decoded
fresh from the pixels on every pass, and the inventory counts it as
one continuous sighting because it never leaves the frame.

**Simulation mode (`--simulate`)** renders a conveyor belt with six
labelled product boxes (four QR codes, four EAN-13 barcodes — two
boxes carry both) and runs the same pipeline on those pixels as an
internal logic check only. Verified 30 s run: all 8 codes decoded,
15 sightings (boxes loop round and are re-sighted), 900 frames.
Simulation is not the demo.

Two real bugs were found and fixed during verification: the EAN-13
parity table in this file had digit 4 as `LGLGGG` (several published
tables print this — it is wrong; ZXing's own writer proves it is
`LGLLGG`), and the payload text under a QR-only box overlapped the
bottom row of the code, which stopped it decoding.

## What the numbers mean

One code in the demo does not mean the system only reads one code:
the footage contains exactly one code. The simulation run is the
proof it counts many codes at once (8 in 30 s). On other footage the
count is whatever the camera actually shows — the inventory never
invents a code.

## Run it

```bash
pip install -r requirements.txt

python qr_barcode_inventory.py                 # opens a file picker
python qr_barcode_inventory.py --source pick   # same picker, explicit
python qr_barcode_inventory.py --source shelf.mp4
python qr_barcode_inventory.py --source 0      # webcam
python qr_barcode_inventory.py --source shelf.mp4 --no-display
python qr_barcode_inventory.py --simulate --max-seconds 30 --no-display
```

Press Q to quit. The annotated video is written to `output.mp4`,
the final inventory table to `output_inventory.txt` and the
timestamped event log to `output_events.log` next to it.

Useful flags: `--max-seconds` (stop early), `--summary-seconds`
(final inventory table hold at the end, default 4).

## Files

- `qr_barcode_inventory.py` — the whole project: ZXing decoding,
  inventory + sightings, on-screen table, simulation belt, EAN-13
  encoder (verified against ZXing's writer)
- `requirements.txt` — `opencv-python`, `numpy`, `pillow`,
  `qrcode` (simulation labels only), `zxing-cpp`
- `output_inventory.txt` / `output_events.log` (produced on each run)

Author: Abdul Hanan — Computer Vision & AI Developer
