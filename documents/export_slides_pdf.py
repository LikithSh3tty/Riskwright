"""Render documents/slides.html to the PDF the assignment requires.

The submission instructions ask for a presentation "saved as a PDF inside the
/documents folder". The deck is HTML, so this screenshots each slide at the
authored 1920x1080 and stitches them into a single PDF. Animations and
navigation are not preserved: the PDF is a static snapshot, which is the normal
and expected trade.

The Frontend Slides skill documents a `scripts/export-pdf.sh` for this. That
script is referenced by the skill but not shipped in the repository, so this is
the equivalent in Python against the Playwright already used for the browser
verification. It finds slides by the `.slide` class, as the skill's own exporter
does, and drives the deck through `window.__deckShow`, which the controller
exposes for exactly this.

    python documents/export_slides_pdf.py

Requires playwright (a dev dependency; not in the served image).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "documents" / "slides.html"
PDF = ROOT / "documents" / "project_presentation.pdf"

STAGE_W, STAGE_H = 1920, 1080

# Playwright's own Chromium, already present from the frontend verification.
# Falls back to whatever the install registers if this path moves.
CHROMIUM = Path(
    r"C:\Users\HP\AppData\Local\ms-playwright\chromium-1223\chrome-win64\chrome.exe"
)


def export(html: Path = HTML, pdf: Path = PDF) -> Path:
    from playwright.sync_api import sync_playwright
    from PIL import Image

    if not html.exists():
        raise FileNotFoundError(
            f"{html} is missing. Run `python documents/build_slides.py` first."
        )

    shots: list[Image.Image] = []
    with sync_playwright() as p:
        launch = {"executable_path": str(CHROMIUM)} if CHROMIUM.exists() else {}
        browser = p.chromium.launch(**launch)
        # Captured at 2x so text stays crisp, then downsampled on save. A
        # straight 2x PDF is ~11MB, which is too heavy to commit for a gain
        # nobody reading it would notice.
        page = browser.new_page(
            viewport={"width": STAGE_W, "height": STAGE_H},
            device_scale_factor=2,
        )
        page.goto(html.as_uri(), wait_until="networkidle")
        # Webfonts must be in before the first capture or slide 1 renders in a
        # fallback face while every later slide renders in JetBrains Mono.
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(1200)

        count = page.evaluate("window.__deckCount")
        print(f"{count} slides found")

        for i in range(count):
            page.evaluate(f"window.__deckShow({i})")
            # Let the staggered reveal finish so nothing is captured mid-fade.
            page.wait_for_timeout(700)
            raw = page.locator(".slide.active").screenshot()
            tmp = ROOT / "documents" / f".slide_{i:02d}.png"
            tmp.write_bytes(raw)
            shots.append(Image.open(tmp).convert("RGB"))
            print(f"  captured slide {i + 1:02d}/{count}")

        browser.close()

    # Downsample to 1.25x, still above the authored size so nothing reads softer
    # than the HTML does.
    target = (int(STAGE_W * 1.25), int(STAGE_H * 1.25))
    shots = [im.resize(target, Image.LANCZOS) for im in shots]

    # Pillow's PDF writer flate-encodes RGB and ignores `quality`, which put a
    # 20-slide deck at 11MB. Round-tripping each frame through an in-memory
    # JPEG makes the writer embed DCTDecode instead, which is what the size
    # difference is. Done here rather than by lowering the capture resolution,
    # so the text stays sharp.
    compressed = []
    for frame in shots:
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=82, optimize=True)
        buffer.seek(0)
        compressed.append(Image.open(buffer))

    compressed[0].save(pdf, save_all=True, append_images=compressed[1:],
                       resolution=144.0)
    for leftover in (ROOT / "documents").glob(".slide_*.png"):
        leftover.unlink()

    print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB, {len(shots)} pages)")
    return pdf


if __name__ == "__main__":
    try:
        export()
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        print("install with: pip install playwright pillow && playwright install chromium",
              file=sys.stderr)
        sys.exit(1)
