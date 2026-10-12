"""False-colour rendering of index tiles (R = NDBI, G = NDVI, B = MNDWI)."""

from __future__ import annotations

import base64
import io

import numpy as np

BG = "#0d1117"


def false_colour(*tiles: np.ndarray) -> list[np.ndarray]:
    """RGB images (H, W, 3 in 0-1) of each tile, stretched with one shared 2-98 % scale."""
    out_channels = []
    for band in (1, 0, 2):
        stacked = np.concatenate([t[band].ravel() for t in tiles])
        lo, hi = np.nanpercentile(stacked, [2, 98])
        out_channels.append([np.clip((t[band] - lo) / (hi - lo + 1e-9), 0, 1) for t in tiles])
    return [np.nan_to_num(np.stack([c[i] for c in out_channels], axis=-1)) for i in range(len(tiles))]


def png_base64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def draw_boxes(ax, clusters, linewidth: float = 1.6) -> None:
    import matplotlib.patches as patches

    top = max((c.score for c in clusters), default=0.0)
    for c in clusters:
        r0, c0, r1, c1 = c.box
        ax.add_patch(
            patches.Rectangle(
                (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, linewidth=linewidth, edgecolor=c.colour(top), facecolor="none"
            )
        )


def tile_png(tile: np.ndarray, title: str, clusters=(), reference: np.ndarray | None = None) -> str:
    """One false-colour tile as a base64 PNG (used by the web app)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = false_colour(tile, reference)[0] if reference is not None else false_colour(tile)[0]
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
    ax.imshow(rgb, interpolation="nearest")
    draw_boxes(ax, clusters)
    ax.set_title(title, color="white", fontsize=11, pad=4)
    ax.axis("off")
    return png_base64(fig)


def panel_png(result) -> str:
    """Before | after with clusters | NDVI change, as one base64 PNG (used by the site report)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair = result.pair
    before_rgb, after_rgb = false_colour(pair.before, pair.after)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), facecolor=BG)
    axes[0].imshow(before_rgb, interpolation="nearest")
    axes[0].set_title(f"Before ({pair.before_year})", color="#c9d1d9", fontsize=13)
    axes[1].imshow(after_rgb, interpolation="nearest")
    draw_boxes(axes[1], result.clusters, linewidth=2)
    axes[1].set_title(f"After ({pair.after_year}) with change clusters", color="#c9d1d9", fontsize=13)
    im = axes[2].imshow(pair.after[0] - pair.before[0], cmap="RdYlGn", vmin=-0.5, vmax=0.5, interpolation="nearest")
    draw_boxes(axes[2], result.clusters)
    axes[2].set_title("NDVI change (red = vegetation lost)", color="#c9d1d9", fontsize=13)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.045, pad=0.02)
    cbar.ax.tick_params(colors="#8b949e")
    for ax in axes:
        ax.axis("off")
    fig.text(
        0.5,
        0.02,
        "False colour: R = NDBI, G = NDVI, B = MNDWI, shared scale, 10 m pixels",
        ha="center",
        color="#8b949e",
        fontsize=10,
    )
    return png_base64(fig)
