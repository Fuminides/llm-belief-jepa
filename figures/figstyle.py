"""Shared figure style: publication defaults (sans-serif, 7-9 pt, thin axes) and the Okabe-Ito colours."""
from itertools import combinations
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

# Okabe-Ito: distinguishable under all common colour-vision deficiencies.
BLUE, VERMILLION, GREEN, ORANGE, SKY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"
GREY = "#7f7f7f"
DOUBLE = 7.2  # double-column width, inches (183 mm)


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "figure.constrained_layout.use": True, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def panel_labels(axes):
    for letter, ax in zip(ascii_uppercase, axes):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _data_points(ax, renderer):
    """Display coordinates of what is plotted in an axes: line points (densified), bar boxes, collection vertices."""
    pts = []
    for line in ax.lines:
        xy = line.get_transform().transform(line.get_xydata())
        xy = xy[np.isfinite(xy).all(1)]
        if len(xy) > 1:
            t = np.linspace(0, 1, 20)[:, None, None]
            xy = (xy[:-1] * (1 - t) + xy[1:] * t).reshape(-1, 2)
        pts.append(xy)
    for patch in ax.patches:
        b = patch.get_window_extent(renderer)
        pts.append(np.array([[b.x0, b.y0], [b.x1, b.y1], [b.x0, b.y1], [b.x1, b.y0], [(b.x0 + b.x1) / 2, b.y1]]))
    for coll in ax.collections:
        for path in coll.get_paths():
            pts.append(coll.get_transform().transform(path.vertices))
    return np.concatenate(pts) if pts else np.empty((0, 2))


def overlaps(fig):
    """Pairs of texts that overlap, and legends or in-axes texts drawn over data."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = set()  # tick labels outside the view limits exist but are not drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 * abs(hi - lo) <= tick.get_loc() <= hi + 1e-9 * abs(hi - lo):
                    hidden.update((tick.label1, tick.label2))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in hidden]
    boxes = [t.get_window_extent(renderer).expanded(0.95, 0.8) for t in texts]
    issues = [f"text {a.get_text()[:30]!r} overlaps {b.get_text()[:30]!r}"
              for (a, ba), (b, bb) in combinations(zip(texts, boxes), 2) if ba.overlaps(bb)]
    for ax in fig.axes:
        pts = _data_points(ax, renderer)
        floating = ([ax.get_legend()] if ax.get_legend() else []) + list(ax.texts)
        for item in floating:
            b = item.get_window_extent(renderer)
            if ((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)).any():
                issues.append(f"{type(item).__name__} {getattr(item, 'get_text', lambda: '')()[:30]!r} covers data")
    return issues


def save(fig, stem):
    """Vector PDF for the paper, PNG for quick viewing. Warns about overlapping text."""
    for issue in overlaps(fig):
        print("WARNING overlap:", issue)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}")
    print("saved", f"{stem}.pdf/.png")
